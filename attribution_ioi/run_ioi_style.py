"""
IOI-style speaker binding circuit analysis for Qwen/Qwen3-8B (or "meta-llama/Llama-3.1-8B-Instruct").

Output folder structure:
  results/{model}/{example}/{type}/{key}/
    1_logit_lens.png            – logit-lens: model answer quality by layer
    2_per_layer_attribution.png – per-layer contribution to logit diff
    3_per_head_attribution.png  – per-head direct logit contribution
    4_attn_analysis.png         – attention diff for top-3 pos/neg heads
    5_patch_resid_pre.png       – resid_pre activation patching
    6_patch_all_blocks.png      – resid / attn-out / mlp-out patching
    7_patch_heads_all_pos.png   – per-head patching (all positions)
    8_patch_qkv.png             – Q/K/V/Pattern patching

  results/{model}/_comparison/     (only when --exp all, phase 2)
    A_patch_heads_grid.png, B_patch_resid_grid.png, C_per_layer_grid.png

  results/{model}/F_raw_top_heads.json   (incremental, shared across all runs)

Flags:
  (none)       standard Swap experiments
  --null       NULL experiments (Claire/Bob)
  --hi-swap    Hi-greeting Swap experiments
  --hi-null    Hi-greeting Null experiments
  --reverse    reverse patching: corrupt SOURCE with BASE activations
  --exp KEY    run a single experiment key instead of all
  --phase 1|2  attribution only (1), patching only (2), or both (all)

Usage:
  # Standard forward patching
  python run_ioi_style.py --model qwen  --gpu 0
  python run_ioi_style.py --model llama --gpu 1

  # Null / Hi variants
  python run_ioi_style.py --model qwen --gpu 0 --null
  python run_ioi_style.py --model qwen --gpu 0 --hi-swap
  python run_ioi_style.py --model qwen --gpu 0 --hi-null

  # Reverse patching (corrupt SOURCE with BASE activations)
  python run_ioi_style.py --model qwen --gpu 0 --reverse
  python run_ioi_style.py --model qwen --gpu 0 --reverse --null

  # Single experiment, attribution only
  python run_ioi_style.py --model qwen --gpu 0 --exp Q_name_bind --phase 1

  # Post-hoc voting from accumulated JSON
  python run_ioi_posthoc.py results/qwen/F_raw_top_heads.json --K 5 --type Swap
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
parser.add_argument("--null", action="store_true",
                    help="Run NULL experiments (Claire/David) instead of standard Alice/Bob")
parser.add_argument("--hi-swap", action="store_true",
                    help="Run HI_SWAP experiments (greeting 'Hi Bob!', swap)")
parser.add_argument("--hi-null", action="store_true",
                    help="Run HI_NULL experiments (greeting 'Hi Bob!', null)")
parser.add_argument("--all-types", action="store_true",
                    help="Run Swap + Null + HiSwap + HiNull all in one go (model loaded once)")
parser.add_argument("--reverse", action="store_true",
                    help="Reverse patching: corrupt SOURCE with BASE activations (shows where model breaks)")
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import einops
from transformer_lens import HookedTransformer, patching
from utils import MODEL_CONFIGS, SWAP_EXPERIMENTS, NULL_EXPERIMENTS, HI_SWAP_EXPERIMENTS, HI_NULL_EXPERIMENTS

device = "cuda"
cfg = MODEL_CONFIGS[args.model]
BASE_OUTDIR = "/mnt/ssd/aryawu/role-representation/attribution_ioi/results"
# In reverse mode metric=1 means BROKEN (toward base), so flip colormap so red=broken
PATCH_CMAP = "RdBu" if args.reverse else "RdBu_r"

# ── Select experiments ────────────────────────────────────────────────────────
if args.all_types:
    EXP_POOL = SWAP_EXPERIMENTS + NULL_EXPERIMENTS + HI_SWAP_EXPERIMENTS + HI_NULL_EXPERIMENTS
elif args.null:
    EXP_POOL = NULL_EXPERIMENTS
elif args.hi_swap:
    EXP_POOL = HI_SWAP_EXPERIMENTS
elif args.hi_null:
    EXP_POOL = HI_NULL_EXPERIMENTS
else:
    EXP_POOL = SWAP_EXPERIMENTS

if args.exp == "all":
    exps_to_run = EXP_POOL
else:
    exps_to_run = [e for e in EXP_POOL if e["key"] == args.exp]
    assert exps_to_run, f"Unknown experiment key: {args.exp}"

# ── Reverse: swap source ↔ base so we corrupt SOURCE with BASE activations ────
if args.reverse:
    def _flip(exp):
        e = dict(exp)
        e["source_prompt"], e["base_prompt"]   = exp["base_prompt"],   exp["source_prompt"]
        e["source_answer"], e["base_answer"]   = exp["base_answer"],   exp["source_answer"]
        e["type"] = "reverse_" + exp["type"]
        return e
    exps_to_run = [_flip(e) for e in exps_to_run]

# ── Load model ────────────────────────────────────────────────────────────────
print(f"Loading {cfg['name']}...")
model = HookedTransformer.from_pretrained(
    cfg["name"], dtype=torch.bfloat16, cache_dir=cfg["cache"])
model.eval().to(device)
print(f"Loaded: {model.cfg.n_layers}L {model.cfg.n_heads}H d_model={model.cfg.d_model}")

# ── Incremental JSON setup (shared across swap/null/hi/reverse runs) ──────────
import json as _json
_json_path = os.path.join(BASE_OUTDIR, args.model, "F_raw_top_heads.json")
os.makedirs(os.path.dirname(_json_path), exist_ok=True)
if os.path.exists(_json_path):
    with open(_json_path) as _f:
        _raw = _json.load(_f)
else:
    _raw = {"model": args.model, "n_layers": model.cfg.n_layers,
            "n_heads": model.cfg.n_heads, "experiments": []}
# uid → index for overwrite-on-rerun
_existing_keys = {e["key"] + e["type"]: i for i, e in enumerate(_raw["experiments"])}


# ══════════════════════════════════════════════════════════════════════════════
# CORE ANALYSIS FUNCTION
# ══════════════════════════════════════════════════════════════════════════════
def run_experiment(exp):
    key    = exp["key"]
    outdir = os.path.join(BASE_OUTDIR, args.model,
                          exp["example"], exp["type"], key)
    os.makedirs(outdir, exist_ok=True)
    print(f"\n{'='*70}")
    print(f"[{exp['kind']}] {key} | type={exp['type']}")
    print(f"{'='*70}")

    # ── Tokenize ─────────────────────────────────────────────────────────────
    clean_tokens     = model.to_tokens(exp["source_prompt"], prepend_bos=True)
    corrupted_tokens = model.to_tokens(exp["base_prompt"],   prepend_bos=True)
    assert clean_tokens.shape == corrupted_tokens.shape, \
        f"Token length mismatch: {clean_tokens.shape[1]} vs {corrupted_tokens.shape[1]}"
    seq_len = clean_tokens.shape[1]

    clean_ids     = clean_tokens[0].tolist()
    corrupted_ids = corrupted_tokens[0].tolist()
    diff_pos      = [i for i, (c, d) in enumerate(zip(clean_ids, corrupted_ids)) if c != d]
    tok_strs           = [model.tokenizer.decode(t) for t in clean_ids]
    tok_repr           = [repr(model.tokenizer.decode(t)) for t in clean_ids]
    corrupted_tok_repr = [repr(model.tokenizer.decode(t)) for t in corrupted_ids]

    print(f"  seq_len={seq_len}, diff_pos={diff_pos} {[tok_repr[p] for p in diff_pos]}")

    # ── Forward passes ────────────────────────────────────────────────────────
    with torch.no_grad():
        clean_logits_raw     = model(clean_tokens)
        corrupted_logits_raw = model(corrupted_tokens)

    # answer=None means undetermined — use model's top pred (handles both null and reverse-null)
    if exp["source_answer"] is None:
        src_id = clean_logits_raw[0, -1].argmax().item()
        print(f"  SOURCE answer=None; using model top pred: "
              f"{model.tokenizer.decode(src_id)!r}")
    else:
        src_id = model.tokenizer.encode(exp["source_answer"], add_special_tokens=False)[0]
        src_actual = model.tokenizer.decode(clean_logits_raw[0, -1].argmax().item())
        print(f"  SOURCE answer={exp['source_answer']!r}  model_pred={src_actual!r}")

    if exp["base_answer"] is None:
        base_id = corrupted_logits_raw[0, -1].argmax().item()
        print(f"  BASE answer=None; using model top pred as foil: "
              f"{model.tokenizer.decode(base_id)!r}")
    else:
        base_id = model.tokenizer.encode(exp["base_answer"], add_special_tokens=False)[0]
        base_actual = model.tokenizer.decode(corrupted_logits_raw[0, -1].argmax().item())
        print(f"  BASE foil={exp['base_answer']!r}  model_pred={base_actual!r}")

    clean_ld     = (clean_logits_raw[0, -1, src_id] - clean_logits_raw[0, -1, base_id]).item()
    corrupted_ld = (corrupted_logits_raw[0, -1, src_id] - corrupted_logits_raw[0, -1, base_id]).item()
    print(f"  SOURCE LD={clean_ld:.3f}, BASE LD={corrupted_ld:.3f}")

    def binding_metric(logits):
        ld    = logits[0, -1, src_id] - logits[0, -1, base_id]
        denom = clean_ld - corrupted_ld
        if abs(denom) < 0.01:
            return ld / (abs(clean_ld) + 1e-6)
        return (ld - corrupted_ld) / denom

    with torch.no_grad():
        clean_logits, clean_cache     = model.run_with_cache(clean_tokens)
        _,            corrupted_cache = model.run_with_cache(corrupted_tokens)

    src_str  = model.tokenizer.decode(src_id)
    base_str = model.tokenizer.decode(base_id)
    pred_label = f"SOURCE→{src_str!r}  BASE→{base_str!r}"

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
        ax.set_title(f"Logit Lens [{exp['kind']}] {key} (type={exp['type']})")
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
        im = ax.imshow(per_head_ld, cmap='RdBu_r', aspect='auto', origin='lower', vmin=-vmax, vmax=vmax)
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
            # At diff positions show: src_tok (red) with (base_tok) appended in black
            labels = list(tok_repr)
            for i in diff_pos:
                labels[i] = tok_repr[i] + '\n(' + corrupted_tok_repr[i] + ')'
            ax.set_xticks(range(len(labels)))
            xlabels = ax.set_xticklabels(labels, rotation=90, ha='center', fontsize=fontsize)
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
                       cmap=PATCH_CMAP, origin='lower', vmin=0, vmax=1)
        plt.colorbar(im, ax=ax, label="Metric (1=SOURCE, 0=BASE)")
        _mark_diff(ax, diff_pos)
        _xticklabels(ax, tok_repr, diff_pos)
        ax.set_yticks(range(model.cfg.n_layers))
        ax.set_xlabel("Token Position"); ax.set_ylabel("Layer")
        ax.set_title(f"resid_pre Patching [{exp['kind']}] {key}\n{pred_label}")
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
                           cmap=PATCH_CMAP, origin='lower', vmin=0, vmax=1)
            plt.colorbar(im, ax=ax)
            _mark_diff(ax, diff_pos)
            _xticklabels(ax, tok_repr, diff_pos)
            ax.set_yticks(range(0, model.cfg.n_layers, 4))
            ax.set_yticklabels(range(0, model.cfg.n_layers, 4))
            ax.set_xlabel("Token Position"); ax.set_ylabel("Layer")
            ax.set_title(f"{blabel} [{exp['kind']}] {key}")
        fig.suptitle(f"Activation Patching: All Blocks [{exp['kind']}] {key}  —  {pred_label}", fontsize=11)
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
                       cmap=PATCH_CMAP, origin='lower', vmin=0, vmax=1)
        plt.colorbar(im, ax=ax, label="Metric (1=SOURCE, 0=BASE)")
        ax.set_xlabel("Head"); ax.set_ylabel("Layer")
        ax.set_title(f"Per-Head Patching (All Positions) [{exp['kind']}] {key}\n{pred_label}")
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
                           cmap=PATCH_CMAP, origin='lower', vmin=0, vmax=1)
            plt.colorbar(im, ax=ax)
            ax.set_xlabel("Head"); ax.set_ylabel("Layer")
            ax.set_title(f"{clabel}")
            ax.set_xticks(range(model.cfg.n_heads))
            ax.set_yticks(range(0, model.cfg.n_layers, 4))
            ax.set_yticklabels(range(0, model.cfg.n_layers, 4))
        fig.suptitle(f"Q/K/V/Pattern Patching (All Positions) [{exp['kind']}] {key}  —  {pred_label}", fontsize=11)
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

    # ── Incremental JSON save ─────────────────────────────────────────────
    if "flat_patch" in results:
        entry = {
            "key":                 exp["key"],
            "kind":                exp["kind"],
            "example":             exp["example"],
            "type":                exp["type"],
            "top_heads":           results["top_heads"],
            "per_head_patch_flat": results["flat_patch"].tolist(),
        }
        uid = entry["key"] + entry["type"]
        if uid in _existing_keys:
            _raw["experiments"][_existing_keys[uid]] = entry
        else:
            _existing_keys[uid] = len(_raw["experiments"])
            _raw["experiments"].append(entry)
        with open(_json_path, "w") as _f:
            _json.dump(_raw, _f, indent=2)
        print(f"  Saved to F_raw_top_heads.json ({len(_raw['experiments'])} experiments total)")

    print(f"  → {outdir}/")
    return results


# ══════════════════════════════════════════════════════════════════════════════
# RUN ALL EXPERIMENTS
# ══════════════════════════════════════════════════════════════════════════════
all_results = {}
for exp in exps_to_run:
    all_results[(exp["key"], exp["type"])] = run_experiment(exp)


print(f"\nDone. Head voting → python run_ioi_posthoc.py {_json_path}")
