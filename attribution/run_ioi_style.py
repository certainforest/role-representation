"""
IOI-style speaker binding circuit analysis for Qwen/Qwen3-8B (or Llama).

Produces for each experiment:
  Phase 1 (attribution):
    1_logit_lens.png           – logit-lens: model answer quality by layer
    2_per_layer_attribution.png – per-layer contribution to logit diff
    3_per_head_attribution.png  – per-head direct logit contribution
    4_attn_analysis.html        – circuitsvis attention for top-3 pos/neg heads

  Phase 2 (patching):
    5_patch_resid_pre.png       – resid_pre activation patching
    6_patch_all_blocks.png      – resid / attn-out / mlp-out patching
    7_patch_heads_all_pos.png   – per-head patching (all positions)
    8_patch_qkv.png             – Q/K/V/Pattern patching

Plus aggregate comparison plots in _comparison/.

Usage:
  python run_ioi_style.py --model qwen --gpu 0
  python run_ioi_style.py --model llama --gpu 1 --exp ex1_1_1
"""
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model", choices=["llama", "qwen"], default="qwen")
parser.add_argument("--gpu",   default="0")
parser.add_argument("--exp",   default="all",
                    help="Experiment key (e.g. ex1_1_1) or 'all'")
parser.add_argument("--phase", choices=["1", "2", "all"], default="all",
                    help="Which phase to run (default: all)")
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import einops
from transformer_lens import HookedTransformer, patching
from utils import MODEL_CONFIGS, EXPERIMENTS

device = "cuda"
cfg = MODEL_CONFIGS[args.model]
BASE_OUTDIR = (f"/mnt/ssd/aryawu/role-representation/attribution"
               f"/circuit_results_{args.model}")

# ── Select experiments ────────────────────────────────────────────────────────
if args.exp == "all":
    exps_to_run = EXPERIMENTS
else:
    exps_to_run = [e for e in EXPERIMENTS if e["key"] == args.exp]
    assert exps_to_run, f"Unknown experiment key: {args.exp}"

# ── Load model ────────────────────────────────────────────────────────────────
print(f"Loading {cfg['name']}...")
model = HookedTransformer.from_pretrained(
    cfg["name"], dtype=torch.bfloat16, cache_dir=cfg["cache"])
model.eval().to(device)
print(f"Loaded: {model.cfg.n_layers}L {model.cfg.n_heads}H d_model={model.cfg.d_model}")


# ══════════════════════════════════════════════════════════════════════════════
# CORE ANALYSIS FUNCTION
# ══════════════════════════════════════════════════════════════════════════════
def run_experiment(exp):
    key    = exp["key"]
    outdir = os.path.join(BASE_OUTDIR, key)
    os.makedirs(outdir, exist_ok=True)
    print(f"\n{'='*70}")
    print(f"[{exp['kind']}] {key} | query={exp['query_type']} swap={exp['swap_type']}")
    print(f"{'='*70}")

    # ── Tokenize ─────────────────────────────────────────────────────────────
    src_id  = model.tokenizer.encode(exp["source_answer"], add_special_tokens=False)[0]
    base_id = model.tokenizer.encode(exp["base_answer"],   add_special_tokens=False)[0]

    clean_tokens     = model.to_tokens(exp["source_prompt"], prepend_bos=True)
    corrupted_tokens = model.to_tokens(exp["base_prompt"],   prepend_bos=True)
    assert clean_tokens.shape == corrupted_tokens.shape, \
        f"Token length mismatch: {clean_tokens.shape[1]} vs {corrupted_tokens.shape[1]}"
    seq_len = clean_tokens.shape[1]

    clean_ids     = clean_tokens[0].tolist()
    corrupted_ids = corrupted_tokens[0].tolist()
    diff_pos      = [i for i, (c, d) in enumerate(zip(clean_ids, corrupted_ids)) if c != d]
    tok_strs      = [model.tokenizer.decode(t) for t in clean_ids]   # readable
    tok_repr      = [repr(model.tokenizer.decode(t)) for t in clean_ids]  # repr for axis

    print(f"  seq_len={seq_len}, diff_pos={diff_pos} {[tok_repr[p] for p in diff_pos]}")

    # ── Forward passes ────────────────────────────────────────────────────────
    with torch.no_grad():
        clean_logits_raw     = model(clean_tokens)
        corrupted_logits_raw = model(corrupted_tokens)

    assert clean_logits_raw[0, -1].argmax().item() == src_id, \
        f"SOURCE predicts wrong: {model.tokenizer.decode(clean_logits_raw[0,-1].argmax().item())!r}"
    assert corrupted_logits_raw[0, -1].argmax().item() == base_id, \
        f"BASE predicts wrong: {model.tokenizer.decode(corrupted_logits_raw[0,-1].argmax().item())!r}"

    clean_ld     = (clean_logits_raw[0, -1, src_id] - clean_logits_raw[0, -1, base_id]).item()
    corrupted_ld = (corrupted_logits_raw[0, -1, src_id] - corrupted_logits_raw[0, -1, base_id]).item()
    print(f"  SOURCE LD={clean_ld:.3f}, BASE LD={corrupted_ld:.3f}")

    def binding_metric(logits):
        ld = logits[0, -1, src_id] - logits[0, -1, base_id]
        return (ld - corrupted_ld) / (clean_ld - corrupted_ld)

    with torch.no_grad():
        clean_logits, clean_cache     = model.run_with_cache(clean_tokens)
        _,            corrupted_cache = model.run_with_cache(corrupted_tokens)

    # logit-diff direction in d_model space
    ld_dir = (model.W_U[:, src_id] - model.W_U[:, base_id]).float().detach()

    def resid_to_ld(resid_stack, cache):
        """Project a residual stack to logit-diff scalar."""
        scaled = cache.apply_ln_to_stack(resid_stack, layer=-1, pos_slice=-1)
        return einops.einsum(scaled.float(), ld_dir,
                             "... batch d_model, d_model -> ... batch").squeeze(-1).detach()

    results = {}

    # ═══════════════════════════════════════════════════════════════════════
    # PHASE 1
    # ═══════════════════════════════════════════════════════════════════════
    if args.phase in ("1", "all"):

        # ── 1. Logit lens ─────────────────────────────────────────────────
        # accumulated_resid returns the residual stream after each layer (embedding + layers)
        # shape: [n_components, batch, d_model]  (at pos_slice=-1)
        acc_resid, acc_labels = clean_cache.accumulated_resid(
            layer=-1, incl_mid=False, pos_slice=-1, return_labels=True)
        logit_lens_ld = resid_to_ld(acc_resid, clean_cache).cpu().numpy()

        fig, ax = plt.subplots(figsize=(12, 3.5))
        ax.plot(logit_lens_ld, marker='o', markersize=3, linewidth=1.5, color='steelblue')
        ax.axhline(clean_ld,     color='green',    linestyle='--', alpha=0.7, label=f"SOURCE LD={clean_ld:.2f}")
        ax.axhline(corrupted_ld, color='firebrick', linestyle='--', alpha=0.7, label=f"BASE LD={corrupted_ld:.2f}")
        ax.axhline(0, color='gray', linewidth=0.5)
        ax.fill_between(range(len(logit_lens_ld)), corrupted_ld, logit_lens_ld,
                        where=(logit_lens_ld > corrupted_ld), alpha=0.15, color='green')
        ax.set_xticks(range(len(acc_labels)))
        ax.set_xticklabels(acc_labels, rotation=45, ha='right', fontsize=6)
        ax.set_xlabel("Layer"); ax.set_ylabel("Logit Diff (source − base token)")
        ax.set_title(f"Logit Lens [{exp['kind']}] {key} (query={exp['query_type']}, swap={exp['swap_type']})")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{outdir}/1_logit_lens.png", dpi=150, bbox_inches='tight')
        plt.close()
        print("  Saved 1_logit_lens.png")

        # ── 2. Per-layer contribution ──────────────────────────────────────
        per_layer_resid, layer_labels = clean_cache.decompose_resid(
            layer=-1, pos_slice=-1, return_labels=True)
        per_layer_ld = resid_to_ld(per_layer_resid, clean_cache).cpu().numpy()

        fig, ax = plt.subplots(figsize=(14, 3.5))
        colors = ['steelblue' if v >= 0 else 'firebrick' for v in per_layer_ld]
        ax.bar(range(len(per_layer_ld)), per_layer_ld, color=colors, width=0.8)
        ax.set_xticks(range(len(layer_labels)))
        ax.set_xticklabels(layer_labels, rotation=45, ha='right', fontsize=6)
        ax.axhline(0, color='black', linewidth=0.5)
        ax.set_xlabel("Component"); ax.set_ylabel("Logit Diff Contribution")
        ax.set_title(f"Per-Layer Contribution [{exp['kind']}] {key}")
        ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig(f"{outdir}/2_per_layer_attribution.png", dpi=150, bbox_inches='tight')
        plt.close()
        print("  Saved 2_per_layer_attribution.png")

        # ── 3. Per-head direct contribution ───────────────────────────────
        per_head_resid, _ = clean_cache.stack_head_results(
            layer=-1, pos_slice=-1, return_labels=True)
        per_head_resid = einops.rearrange(
            per_head_resid,
            "(layer head) batch d_model -> layer head batch d_model",
            layer=model.cfg.n_layers)
        per_head_ld = resid_to_ld(per_head_resid, clean_cache).cpu().numpy()

        vmax = max(abs(per_head_ld).max(), 0.1)
        fig, ax = plt.subplots(figsize=(model.cfg.n_heads * 0.45, model.cfg.n_layers * 0.3))
        im = ax.imshow(per_head_ld, cmap='RdBu_r', aspect='auto', vmin=-vmax, vmax=vmax)
        plt.colorbar(im, ax=ax, label="Logit Diff Contribution")
        ax.set_xlabel("Head"); ax.set_ylabel("Layer")
        ax.set_title(f"Per-Head Direct Contribution [{exp['kind']}] {key}")
        ax.set_xticks(range(model.cfg.n_heads))
        ax.set_yticks(range(model.cfg.n_layers))
        plt.tight_layout()
        plt.savefig(f"{outdir}/3_per_head_attribution.png", dpi=150, bbox_inches='tight')
        plt.close()
        print("  Saved 3_per_head_attribution.png")

        # ── 4. Attention diff for top 3 pos / top 3 neg heads ─────────────
        # Each head: 3 columns = SOURCE | BASE | SOURCE−BASE
        flat_ld = per_head_ld.flatten()
        top_pos_idx = np.argsort(flat_ld)[-3:][::-1]
        top_neg_idx = np.argsort(flat_ld)[:3]
        focus_heads = [(divmod(i, model.cfg.n_heads), flat_ld[i]) for i in
                       list(top_pos_idx) + list(top_neg_idx)]
        focus_heads = [((int(l), int(h)), score) for (l, h), score in focus_heads]

        print(f"  Top 3 positive: {[(l,h,f'{s:.3f}') for (l,h),s in focus_heads[:3]]}")
        print(f"  Top 3 negative: {[(l,h,f'{s:.3f}') for (l,h),s in focus_heads[3:]]}")

        n_heads_show = len(focus_heads)  # 6
        fig, axes = plt.subplots(n_heads_show, 3,
                                 figsize=(seq_len * 0.35 * 3, n_heads_show * seq_len * 0.22 + 0.5))
        if n_heads_show == 1:
            axes = axes[np.newaxis, :]

        for row, ((layer, head), score) in enumerate(focus_heads):
            src_attn  = clean_cache["pattern",     layer][0, head].float().cpu().numpy()
            base_attn = corrupted_cache["pattern", layer][0, head].float().cpu().numpy()
            diff_attn = src_attn - base_attn
            sign  = "+" if score > 0 else ""
            title_prefix = f"L{layer}H{head}  LD={sign}{score:.2f}"

            for col, (data, cmap, subtitle) in enumerate([
                (src_attn,  "Blues",  "SOURCE"),
                (base_attn, "Blues",  "BASE"),
                (diff_attn, "RdBu_r", "SOURCE − BASE"),
            ]):
                ax = axes[row, col]
                vmax = max(abs(data).max(), 0.01)
                im = ax.imshow(data, cmap=cmap, aspect="auto",
                               vmin=-vmax if col == 2 else 0, vmax=vmax)
                plt.colorbar(im, ax=ax, shrink=0.6)
                ax.set_xticks(range(seq_len))
                ax.set_xticklabels(tok_strs, rotation=90, fontsize=6)
                ax.set_yticks(range(seq_len))
                ax.set_yticklabels(tok_strs, fontsize=6)
                for pos in diff_pos:
                    ax.axvline(x=pos - 0.5, color="orange", alpha=0.5, lw=1.2)
                    ax.axvline(x=pos + 0.5, color="orange", alpha=0.5, lw=1.2)
                ax.set_title(f"{title_prefix} | {subtitle}", fontsize=8)

        fig.suptitle(f"Attn Diff: Top 3 Pos + Top 3 Neg Heads by Direct Logit Attribution\n"
                     f"[{exp['kind']}] {key}  —  orange = differing positions", fontsize=9)
        plt.tight_layout()
        plt.savefig(f"{outdir}/4_attn_analysis.png", dpi=130, bbox_inches='tight')
        plt.close()
        print("  Saved 4_attn_analysis.png")

        results["per_head_ld"] = per_head_ld
        results["per_layer_ld"] = per_layer_ld
        results["layer_labels"] = layer_labels

    # ═══════════════════════════════════════════════════════════════════════
    # PHASE 2
    # ═══════════════════════════════════════════════════════════════════════
    if args.phase in ("2", "all"):

        def _mark_diff(ax, diff_pos):
            for pos in diff_pos:
                ax.axvline(x=pos - 0.5, color='orange', alpha=0.5, linewidth=1.5)
                ax.axvline(x=pos + 0.5, color='orange', alpha=0.5, linewidth=1.5)

        def _xticklabels(ax, tok_repr, diff_pos, fontsize=5):
            ax.set_xticks(range(len(tok_repr)))
            xlabels = ax.set_xticklabels(tok_repr, rotation=90, ha='center', fontsize=fontsize)
            for i in diff_pos:
                xlabels[i].set_color('red')
                xlabels[i].set_fontweight('bold')
            return xlabels

        # ── 5. resid_pre patching ──────────────────────────────────────────
        print(f"  resid_pre patching ({seq_len * model.cfg.n_layers} passes)...")
        act_patch_resid = patching.get_act_patch_resid_pre(
            model=model, corrupted_tokens=corrupted_tokens,
            clean_cache=clean_cache, patching_metric=binding_metric)

        fig, ax = plt.subplots(figsize=(max(14, seq_len * 0.35), model.cfg.n_layers * 0.3))
        im = ax.imshow(act_patch_resid.float().cpu().numpy(), aspect='auto',
                       cmap='RdBu_r', origin='lower', vmin=0, vmax=1)
        plt.colorbar(im, ax=ax, label="Metric (1=SOURCE, 0=BASE)")
        _mark_diff(ax, diff_pos)
        _xticklabels(ax, tok_repr, diff_pos)
        ax.set_yticks(range(model.cfg.n_layers))
        ax.set_xlabel("Token Position"); ax.set_ylabel("Layer")
        ax.set_title(f"resid_pre Patching [{exp['kind']}] {key}")
        plt.tight_layout()
        plt.savefig(f"{outdir}/5_patch_resid_pre.png", dpi=150, bbox_inches='tight')
        plt.close()
        print("  Saved 5_patch_resid_pre.png")

        # ── 6. All blocks patching (resid / attn_out / mlp_out) ───────────
        print(f"  block patching (3 × {seq_len * model.cfg.n_layers} passes)...")
        act_patch_blocks = patching.get_act_patch_block_every(
            model=model, corrupted_tokens=corrupted_tokens,
            clean_cache=clean_cache, metric=binding_metric)

        block_labels = ["Residual Stream", "Attn Output", "MLP Output"]
        fig, axes = plt.subplots(1, 3, figsize=(max(14, seq_len * 0.35) * 3,
                                                 model.cfg.n_layers * 0.3))
        for ax, data, blabel in zip(axes, act_patch_blocks, block_labels):
            im = ax.imshow(data.float().cpu().numpy(), aspect='auto',
                           cmap='RdBu_r', origin='lower', vmin=0, vmax=1)
            plt.colorbar(im, ax=ax)
            _mark_diff(ax, diff_pos)
            _xticklabels(ax, tok_repr, diff_pos)
            ax.set_yticks(range(0, model.cfg.n_layers, 4))
            ax.set_yticklabels(range(0, model.cfg.n_layers, 4))
            ax.set_xlabel("Token Position"); ax.set_ylabel("Layer")
            ax.set_title(f"{blabel} [{exp['kind']}] {key}")
        fig.suptitle(f"Activation Patching: All Blocks [{exp['kind']}] {key}", fontsize=11)
        plt.tight_layout()
        plt.savefig(f"{outdir}/6_patch_all_blocks.png", dpi=150, bbox_inches='tight')
        plt.close()
        print("  Saved 6_patch_all_blocks.png")

        # ── 7. Per-head patching (all positions) ──────────────────────────
        print(f"  per-head patching ({model.cfg.n_layers * model.cfg.n_heads} passes)...")
        act_patch_heads = patching.get_act_patch_attn_head_out_all_pos(
            model=model, corrupted_tokens=corrupted_tokens,
            clean_cache=clean_cache, patching_metric=binding_metric)

        flat_patch = act_patch_heads.float().cpu().numpy().flatten()
        top_heads = []
        for idx in np.argsort(flat_patch)[-10:][::-1]:
            l, h = divmod(idx, model.cfg.n_heads)
            top_heads.append((int(l), int(h)))
        print(f"  Top heads: {top_heads[:5]}  max={flat_patch.max():.4f}")

        fig, ax = plt.subplots(figsize=(model.cfg.n_heads * 0.45, model.cfg.n_layers * 0.3))
        im = ax.imshow(act_patch_heads.float().cpu().numpy(), aspect='auto',
                       cmap='RdBu_r', origin='lower', vmin=0, vmax=1)
        plt.colorbar(im, ax=ax, label="Metric (1=SOURCE, 0=BASE)")
        ax.set_xlabel("Head"); ax.set_ylabel("Layer")
        ax.set_title(f"Per-Head Patching (All Positions) [{exp['kind']}] {key}")
        ax.set_xticks(range(model.cfg.n_heads))
        ax.set_yticks(range(model.cfg.n_layers))
        plt.tight_layout()
        plt.savefig(f"{outdir}/7_patch_heads_all_pos.png", dpi=150, bbox_inches='tight')
        plt.close()
        print("  Saved 7_patch_heads_all_pos.png")

        # ── 8. Q / K / V / Pattern patching ──────────────────────────────
        print(f"  Q/K/V/Pattern patching (5 × {model.cfg.n_layers * model.cfg.n_heads} passes)...")
        act_patch_qkvo = patching.get_act_patch_attn_head_all_pos_every(
            model=model, corrupted_tokens=corrupted_tokens,
            clean_cache=clean_cache, metric=binding_metric)

        comp_labels = ["Output", "Query", "Key", "Value", "Pattern"]
        fig, axes = plt.subplots(1, 5, figsize=(model.cfg.n_heads * 0.45 * 5,
                                                 model.cfg.n_layers * 0.3))
        for ax, data, clabel in zip(axes, act_patch_qkvo, comp_labels):
            im = ax.imshow(data.float().cpu().numpy(), aspect='auto',
                           cmap='RdBu_r', origin='lower', vmin=0, vmax=1)
            plt.colorbar(im, ax=ax)
            ax.set_xlabel("Head"); ax.set_ylabel("Layer")
            ax.set_title(f"{clabel}")
            ax.set_xticks(range(model.cfg.n_heads))
            ax.set_yticks(range(0, model.cfg.n_layers, 4))
            ax.set_yticklabels(range(0, model.cfg.n_layers, 4))
        fig.suptitle(f"Q/K/V/Pattern Patching (All Positions) [{exp['kind']}] {key}", fontsize=11)
        plt.tight_layout()
        plt.savefig(f"{outdir}/8_patch_qkv.png", dpi=150, bbox_inches='tight')
        plt.close()
        print("  Saved 8_patch_qkv.png")

        results["patch_heads"]  = act_patch_heads.float().cpu().numpy()
        results["patch_resid"]  = act_patch_resid.float().cpu().numpy()
        results["flat_patch"]   = flat_patch
        results["top_heads"]    = top_heads
        results["diff_pos"]     = diff_pos
        results["tok_repr"]     = tok_repr

    print(f"  → {outdir}/")
    return results


# ══════════════════════════════════════════════════════════════════════════════
# RUN ALL EXPERIMENTS
# ══════════════════════════════════════════════════════════════════════════════
all_results = {}
for exp in exps_to_run:
    all_results[exp["key"]] = run_experiment(exp)


# ══════════════════════════════════════════════════════════════════════════════
# AGGREGATE COMPARISON PLOTS (only when running all 8)
# ══════════════════════════════════════════════════════════════════════════════
if args.exp == "all" and args.phase in ("2", "all"):
    CMP_DIR = os.path.join(BASE_OUTDIR, "_comparison")
    os.makedirs(CMP_DIR, exist_ok=True)
    print(f"\n{'='*70}")
    print("AGGREGATE COMPARISON PLOTS")
    print(f"{'='*70}")

    K = 10  # top-K heads per experiment for voting

    # ── A. 2×4 per-head patching grid ─────────────────────────────────────
    vmax_g = max(all_results[e["key"]]["flat_patch"].max() for e in EXPERIMENTS)
    fig = plt.figure(figsize=(model.cfg.n_heads * 0.4 * 4, model.cfg.n_layers * 0.3 * 2 + 2))
    gs  = gridspec.GridSpec(2, 4, figure=fig, hspace=0.5, wspace=0.3)

    for row, example in enumerate(["ex1", "ex2"]):
        for col, exp in enumerate([e for e in EXPERIMENTS if e["example"] == example]):
            ax = fig.add_subplot(gs[row, col])
            data = all_results[exp["key"]]["patch_heads"]
            im = ax.imshow(data, aspect='auto', cmap='RdBu_r', origin='lower',
                           vmin=0, vmax=vmax_g)
            ax.set_xticks(range(0, model.cfg.n_heads, 4))
            ax.set_xticklabels(range(0, model.cfg.n_heads, 4), fontsize=6)
            ax.set_yticks(range(0, model.cfg.n_layers, 4))
            ax.set_yticklabels(range(0, model.cfg.n_layers, 4), fontsize=6)
            ax.set_xlabel("Head", fontsize=7); ax.set_ylabel("Layer", fontsize=7)
            bc = 'firebrick' if exp['kind'] == 'BINDING' else 'steelblue'
            for spine in ax.spines.values():
                spine.set_edgecolor(bc); spine.set_linewidth(2.5)
            ax.set_title(exp["label"], fontsize=7, color=bc)

    fig.suptitle("Per-Head Patching: All 8 Experiments\n"
                 "(red border=BINDING, blue=CONTROL)", fontsize=10)
    plt.colorbar(im, ax=fig.axes, shrink=0.4, label="Metric (1=SOURCE)")
    plt.savefig(f"{CMP_DIR}/A_patch_heads_grid.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved A_patch_heads_grid.png")

    # ── B. 2×4 resid_pre patching grid ────────────────────────────────────
    fig, axes = plt.subplots(2, 4, figsize=(model.cfg.n_heads * 0.35 * 4,
                                             model.cfg.n_layers * 0.25 * 2 + 2))
    for row, example in enumerate(["ex1", "ex2"]):
        for col, exp in enumerate([e for e in EXPERIMENTS if e["example"] == example]):
            ax   = axes[row, col]
            r    = all_results[exp["key"]]
            im   = ax.imshow(r["patch_resid"], aspect='auto', cmap='RdBu_r',
                             origin='lower', vmin=0, vmax=1)
            for pos in r["diff_pos"]:
                ax.axvline(x=pos-0.5, color='orange', alpha=0.6, lw=1.5)
                ax.axvline(x=pos+0.5, color='orange', alpha=0.6, lw=1.5)
            ax.set_yticks(range(0, model.cfg.n_layers, 8))
            ax.set_yticklabels(range(0, model.cfg.n_layers, 8), fontsize=6)
            ax.set_xticks([]); ax.set_xlabel("Tokens", fontsize=6); ax.set_ylabel("Layer", fontsize=6)
            bc = 'firebrick' if exp['kind'] == 'BINDING' else 'steelblue'
            for spine in ax.spines.values():
                spine.set_edgecolor(bc); spine.set_linewidth(2.5)
            ax.set_title(exp["label"], fontsize=7, color=bc)
    fig.suptitle("resid_pre Patching: All 8 Experiments (orange=diff positions)\n"
                 "red=BINDING, blue=CONTROL", fontsize=10)
    plt.tight_layout()
    plt.savefig(f"{CMP_DIR}/B_patch_resid_grid.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved B_patch_resid_grid.png")

    # ── C. Per-layer contribution comparison ──────────────────────────────
    if "per_layer_ld" in all_results[EXPERIMENTS[0]["key"]]:
        fig, axes = plt.subplots(2, 4, figsize=(14 * 4 // 3, 3.5 * 2))
        for row, example in enumerate(["ex1", "ex2"]):
            exps_row = [e for e in EXPERIMENTS if e["example"] == example]
            ymax = max(abs(all_results[e["key"]]["per_layer_ld"]).max() for e in exps_row) * 1.1
            for col, exp in enumerate(exps_row):
                ax   = axes[row, col]
                r    = all_results[exp["key"]]
                vals = r["per_layer_ld"]
                colors = ['steelblue' if v >= 0 else 'firebrick' for v in vals]
                ax.bar(range(len(vals)), vals, color=colors, width=0.8)
                ax.set_ylim(-ymax, ymax)
                ax.axhline(0, color='black', linewidth=0.5)
                ax.set_xticks(range(0, len(vals), 4))
                ax.set_xticklabels(r["layer_labels"][::4], rotation=45, ha='right', fontsize=5)
                bc = 'firebrick' if exp['kind'] == 'BINDING' else 'steelblue'
                for spine in ax.spines.values():
                    spine.set_edgecolor(bc); spine.set_linewidth(2.5)
                ax.set_title(exp["label"], fontsize=7, color=bc)
                ax.grid(True, alpha=0.2, axis='y')
        fig.suptitle("Per-Layer Contribution (all 8 experiments)\nred=BINDING, blue=CONTROL",
                     fontsize=10)
        plt.tight_layout()
        plt.savefig(f"{CMP_DIR}/C_per_layer_grid.png", dpi=150, bbox_inches='tight')
        plt.close()
        print("  Saved C_per_layer_grid.png")

    # ── D. Head voting: all 8 experiments ─────────────────────────────────
    head_vote = {}
    print("\nTop-10 heads per experiment:")
    for exp in EXPERIMENTS:
        r = all_results[exp["key"]]
        print(f"  {exp['key']:12s} [{exp['kind']:7s}]: {r['top_heads'][:5]}")
        for lh in r["top_heads"][:K]:
            head_vote[lh] = head_vote.get(lh, 0) + 1

    vote_map = np.zeros((model.cfg.n_layers, model.cfg.n_heads))
    for (l, h), cnt in head_vote.items():
        vote_map[l, h] = cnt

    fig, ax = plt.subplots(figsize=(model.cfg.n_heads * 0.45, model.cfg.n_layers * 0.3))
    im = ax.imshow(vote_map, aspect='auto', cmap='YlOrRd', origin='lower', vmin=0, vmax=8)
    plt.colorbar(im, ax=ax, label=f"# experiments in top-{K} (out of 8)")
    ax.set_xlabel("Head"); ax.set_ylabel("Layer")
    ax.set_title(f"Head Voting: All 8 Experiments (top-{K})")
    ax.set_xticks(range(model.cfg.n_heads)); ax.set_yticks(range(model.cfg.n_layers))
    for (l, h), cnt in head_vote.items():
        if cnt >= 4:
            ax.text(h, l, str(cnt), ha='center', va='center', fontsize=7,
                    color='white' if cnt >= 6 else 'black', fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{CMP_DIR}/D_head_voting_all.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved D_head_voting_all.png")

    # ── E. BINDING vs CONTROL head voting ─────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(model.cfg.n_heads * 0.9, model.cfg.n_layers * 0.3))
    binding_set, control_set = set(), set()
    for ax, kind, title in zip(axes, ["BINDING", "CONTROL"],
                               ["BINDING (4 exps)", "CONTROL (4 exps)"]):
        vote_m = np.zeros((model.cfg.n_layers, model.cfg.n_heads))
        for exp in EXPERIMENTS:
            if exp["kind"] != kind:
                continue
            for l, h in all_results[exp["key"]]["top_heads"][:K]:
                vote_m[l, h] += 1
                (binding_set if kind == "BINDING" else control_set).add((l, h))
        im = ax.imshow(vote_m, aspect='auto', cmap='YlOrRd', origin='lower', vmin=0, vmax=4)
        plt.colorbar(im, ax=ax, label="# exps in top-10")
        ax.set_xlabel("Head"); ax.set_ylabel("Layer"); ax.set_title(title)
        ax.set_xticks(range(model.cfg.n_heads)); ax.set_yticks(range(model.cfg.n_layers))
        for l in range(model.cfg.n_layers):
            for h in range(model.cfg.n_heads):
                if vote_m[l, h] >= 3:
                    ax.text(h, l, str(int(vote_m[l, h])), ha='center', va='center',
                            fontsize=7, color='white', fontweight='bold')
    fig.suptitle(f"Head Voting: BINDING vs CONTROL (top-{K})", fontsize=11)
    plt.tight_layout()
    plt.savefig(f"{CMP_DIR}/E_head_voting_binding_vs_control.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved E_head_voting_binding_vs_control.png")

    # ── F. Entity-swap vs Attr-swap ────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(model.cfg.n_heads * 0.9, model.cfg.n_layers * 0.3))
    entity_set, attr_set = set(), set()
    for ax, swap_type, title in zip(axes, ["entity", "attr"],
                                    ["Entity-Swap (4 exps)", "Attr-Swap (4 exps)"]):
        vote_m = np.zeros((model.cfg.n_layers, model.cfg.n_heads))
        for exp in EXPERIMENTS:
            if exp["swap_type"] != swap_type:
                continue
            for l, h in all_results[exp["key"]]["top_heads"][:K]:
                vote_m[l, h] += 1
                (entity_set if swap_type == "entity" else attr_set).add((l, h))
        im = ax.imshow(vote_m, aspect='auto', cmap='Blues', origin='lower', vmin=0, vmax=4)
        plt.colorbar(im, ax=ax, label="# exps in top-10")
        ax.set_xlabel("Head"); ax.set_ylabel("Layer"); ax.set_title(title)
        ax.set_xticks(range(model.cfg.n_heads)); ax.set_yticks(range(model.cfg.n_layers))
        for l in range(model.cfg.n_layers):
            for h in range(model.cfg.n_heads):
                if vote_m[l, h] >= 3:
                    ax.text(h, l, str(int(vote_m[l, h])), ha='center', va='center',
                            fontsize=7, color='white', fontweight='bold')
    fig.suptitle(f"Head Voting: Entity-Swap vs Attr-Swap (top-{K})", fontsize=11)
    plt.tight_layout()
    plt.savefig(f"{CMP_DIR}/F_head_voting_entity_vs_attr.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved F_head_voting_entity_vs_attr.png")

    # ── Print summary ──────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    binding_only = binding_set - control_set
    control_only = control_set - binding_set
    shared_bc    = binding_set & control_set
    print(f"Binding-only heads (top-{K}): {sorted(binding_only)}")
    print(f"Control-only heads (top-{K}): {sorted(control_only)}")
    print(f"Shared binding+control:        {sorted(shared_bc)}")
    entity_only = entity_set - attr_set
    attr_only   = attr_set - entity_set
    print(f"Entity-swap-only heads:        {sorted(entity_only)}")
    print(f"Attr-swap-only heads:          {sorted(attr_only)}")
    print(f"Shared entity+attr:            {sorted(entity_set & attr_set)}")
    print(f"\nAll plots → {BASE_OUTDIR}/")

    # ── Save raw top-K data for post-hoc analysis ─────────────────────────
    import json
    raw = {
        "model": args.model,
        "K": K,
        "experiments": [
            {
                "key":        e["key"],
                "kind":       e["kind"],
                "example":    e["example"],
                "query_type": e["query_type"],
                "swap_type":  e["swap_type"],
                "top_heads":  all_results[e["key"]]["top_heads"],   # top-K
                "per_head_patch_flat": all_results[e["key"]]["flat_patch"].tolist(),
            }
            for e in EXPERIMENTS
        ],
    }
    json_path = os.path.join(CMP_DIR, "raw_top_heads.json")
    with open(json_path, "w") as f:
        json.dump(raw, f, indent=2)
    print(f"  Saved raw_top_heads.json  (use for post-hoc K re-analysis)")
