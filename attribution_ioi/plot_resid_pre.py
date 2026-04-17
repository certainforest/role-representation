"""
Standalone paper-quality resid_pre activation patching plots.

Edit CASES at the bottom to pick experiments.
Pass --load-npy to skip re-running patching (uses .npy saved by run_ioi_style.py).

Usage:
  python plot_resid_pre.py --model qwen  --gpu 0
  python plot_resid_pre.py --model llama --gpu 1
  python plot_resid_pre.py --model qwen  --gpu 0 --load-npy
  python plot_resid_pre.py --model qwen  --gpu 0 --out my_paper_figs/
"""

import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model", choices=["llama", "meta-llama/Llama-3.1-8B-Instruct"], default="llama")
parser.add_argument("--gpu",   default="0")
parser.add_argument("--load-npy", action="store_true",
                    help="Load precomputed .npy arrays instead of re-running patching")
parser.add_argument("--out", default=None,
                    help="Output directory root (default: same as run_ioi_style.py results)")
parser.add_argument("--cmap", default="RdBu_r",
                    help="Colormap (RdBu_r=forward, RdBu=reverse)")
parser.add_argument("--prob-diff", action="store_true",
                    help="Plot P(src) - P(base) instead of normalized LD recovery")
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

from utils import (MODEL_CONFIGS,
                   SWAP_EXPERIMENTS, NULL_EXPERIMENTS,
                   HI_SWAP_EXPERIMENTS, HI_NULL_EXPERIMENTS)

# ── Paper figure style ────────────────────────────────────────────────────────
FONT_SIZE_LABEL  = 10   # axis labels + colorbar label
FONT_SIZE_TICKS  = 10   # y-axis layer numbers
FONT_SIZE_XTICKS = 10   # token tick labels
DPI              = 300
FIG_WIDTH        = 10   # fixed figure width in inches
FIG_HEIGHT       = 6    # fixed figure height in inches

# Prefix text to hide from the x-axis (everything before the dialogue).
# The display starts at the first token AFTER this string.
SKIP_PREFIX = 'This is the transcript of a conversation.\n'

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BASE_OUTDIR = "/mnt/ssd/aryawu/role-representation/attribution_ioi/results"

model = None


def _load_model():
    global model
    if model is not None:
        return
    import torch
    from transformer_lens import HookedTransformer
    cfg = MODEL_CONFIGS[args.model]
    print(f"Loading {cfg['name']}...")
    model = HookedTransformer.from_pretrained(
        cfg["name"], dtype=torch.bfloat16, cache_dir=cfg["cache"],
        n_devices=cfg.get("n_devices", 1))
    model.eval()
    if cfg.get("n_devices", 1) == 1:
        model.to("cuda")
    print(f"Loaded: {model.cfg.n_layers}L {model.cfg.n_heads}H d_model={model.cfg.d_model}")


def _plot(data, tok_repr, corrupted_tok_repr, diff_pos,
          src_str, base_str, n_skip, outpath, vmin, vmax, cbar_label):
    # Crop to displayed region
    data_show  = data[:, n_skip:]
    tok_show   = tok_repr[n_skip:]
    ctok_show  = corrupted_tok_repr[n_skip:]
    diff_show  = [p - n_skip for p in diff_pos if p >= n_skip]

    n_layers, seq_show = data_show.shape
    fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT))
    im = ax.imshow(data_show, aspect='auto', cmap=args.cmap,
                   origin='lower', vmin=vmin, vmax=vmax)

    cbar = plt.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label(cbar_label, fontsize=FONT_SIZE_LABEL)
    cbar.ax.tick_params(labelsize=FONT_SIZE_TICKS)

    # orange bands at diff positions
    for pos in diff_show:
        ax.axvline(x=pos - 0.5, color='orange', alpha=0.6, linewidth=1.5)
        ax.axvline(x=pos + 0.5, color='orange', alpha=0.6, linewidth=1.5)

    # x-tick labels: src token; at diff positions append (base token) in red
    labels = list(tok_show)
    for i in diff_show:
        labels[i] = tok_show[i] + '\n(' + ctok_show[i] + ')'
    ax.set_xticks(range(seq_show))
    xlabels = ax.set_xticklabels(labels, rotation=90, ha='center',
                                  fontsize=FONT_SIZE_XTICKS)
    for i in diff_show:
        xlabels[i].set_color('red')
        xlabels[i].set_fontweight('bold')

    ax.set_yticks(range(n_layers))
    ax.tick_params(axis='y', labelsize=FONT_SIZE_TICKS)
    ax.set_xlabel("Token Position", fontsize=FONT_SIZE_LABEL)
    ax.set_ylabel("Layer",          fontsize=FONT_SIZE_LABEL)

    plt.tight_layout()
    plt.savefig(outpath, dpi=DPI, bbox_inches='tight')
    plt.close()
    print(f"  Saved {outpath}")


def run_case(exp):
    import torch
    from transformer_lens import patching

    key         = exp["key"]
    src_answer  = exp["source_answer"]
    base_answer = exp["base_answer"]

    outdir = os.path.join(BASE_OUTDIR, args.model,
                          exp["example"], exp["type"], key)
    if args.out:
        outdir = os.path.join(args.out, args.model,
                              exp["example"], exp["type"], key)
    os.makedirs(outdir, exist_ok=True)

    npy_path = os.path.join(
        BASE_OUTDIR, args.model, exp["example"], exp["type"], key,
        "5_patch_resid_pre.npy")
    suffix  = "_probdiff" if args.prob_diff else ""
    out_png = os.path.join(outdir, f"5_patch_resid_pre_paper{suffix}.pdf")

    print(f"\n{'='*70}")
    print(f"[{exp['kind']}] {key}  type={exp['type']}")

    _load_model()

    # Tokenise both prompts (needed for tick labels regardless of --load-npy)
    clean_tokens     = model.to_tokens(exp["source_prompt"], prepend_bos=True)
    corrupted_tokens = model.to_tokens(exp["base_prompt"],   prepend_bos=True)
    assert clean_tokens.shape == corrupted_tokens.shape, \
        f"Token length mismatch: {clean_tokens.shape[1]} vs {corrupted_tokens.shape[1]}"

    clean_ids     = clean_tokens[0].tolist()
    corrupted_ids = corrupted_tokens[0].tolist()
    diff_pos           = [i for i, (c, d) in enumerate(zip(clean_ids, corrupted_ids)) if c != d]
    tok_repr           = [repr(model.tokenizer.decode(t)) for t in clean_ids]
    corrupted_tok_repr = [repr(model.tokenizer.decode(t)) for t in corrupted_ids]

    # How many leading tokens to hide (BOS + SKIP_PREFIX)
    prefix_ids = model.tokenizer.encode(SKIP_PREFIX, add_special_tokens=False)
    n_skip = 1 + len(prefix_ids)   # +1 for BOS

    # ── Load or compute patching array ───────────────────────────────────
    if args.load_npy and os.path.exists(npy_path):
        data = np.load(npy_path)
        print(f"  Loaded precomputed array from {npy_path}")
        src_str  = src_answer.strip() if src_answer else "?"
        base_str = base_answer.strip() if base_answer else "?"
    else:
        with torch.no_grad():
            clean_logits_raw     = model(clean_tokens)
            corrupted_logits_raw = model(corrupted_tokens)

        if src_answer is None:
            src_id = clean_logits_raw[0, -1].argmax().item()
        else:
            src_id = model.tokenizer.encode(src_answer, add_special_tokens=False)[0]

        if base_answer is None:
            base_id = corrupted_logits_raw[0, -1].argmax().item()
        else:
            base_id = model.tokenizer.encode(base_answer, add_special_tokens=False)[0]

        src_str  = model.tokenizer.decode(src_id).strip()
        base_str = model.tokenizer.decode(base_id).strip()

        clean_ld     = (clean_logits_raw[0, -1, src_id] - clean_logits_raw[0, -1, base_id]).item()
        corrupted_ld = (corrupted_logits_raw[0, -1, src_id] - corrupted_logits_raw[0, -1, base_id]).item()
        print(f"  SOURCE LD={clean_ld:.3f}  BASE LD={corrupted_ld:.3f}")

        if args.prob_diff:
            import torch.nn.functional as F
            def metric(logits):
                probs = F.softmax(logits[0, -1].float(), dim=-1)
                return probs[src_id] - probs[base_id]
        else:
            def metric(logits):
                ld    = logits[0, -1, src_id] - logits[0, -1, base_id]
                denom = clean_ld - corrupted_ld
                if abs(denom) < 0.01:
                    return ld / (abs(clean_ld) + 1e-6)
                return (ld - corrupted_ld) / denom

        seq_len = clean_tokens.shape[1]
        print(f"  resid_pre patching ({seq_len * model.cfg.n_layers} passes)...")
        _, clean_cache = model.run_with_cache(clean_tokens)
        with torch.no_grad():
            act_patch = patching.get_act_patch_resid_pre(
                model=model, corrupted_tokens=corrupted_tokens,
                clean_cache=clean_cache, patching_metric=metric)
        data = act_patch.float().cpu().numpy()
        del clean_cache

    if args.prob_diff:
        vabs = max(abs(data).max(), 0.01)
        vmin, vmax = -vabs, vabs
        cbar_label = f"P({src_str}) − P({base_str})"
    else:
        vmin, vmax = 0, 1
        cbar_label = f"Logit diff recovery (0 = LD_dst, 1 = LD_src)"

    _plot(data, tok_repr, corrupted_tok_repr, diff_pos,
          src_str, base_str, n_skip, out_png, vmin, vmax, cbar_label)


# ══════════════════════════════════════════════════════════════════════════════
# CASES — edit this list to choose which experiments get paper plots
# ══════════════════════════════════════════════════════════════════════════════
def _get(pool, key):
    matches = [e for e in pool if e["key"] == key]
    assert matches, f"Key {key!r} not found"
    return matches[0]

CASES = [
    # _get(SWAP_EXPERIMENTS,    "Q_country_bind"),     # Swap
    _get(NULL_EXPERIMENTS,    "Q_name_bind"),     # Null
    # _get(HI_SWAP_EXPERIMENTS, "Q_country_hi_bind"),  # HiSwap
    _get(HI_NULL_EXPERIMENTS, "Q_name_hi_bind"),  # HiNull
]

for exp in CASES:
    run_case(exp)

print("\nDone.")
