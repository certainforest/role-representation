"""
Paper-quality resid_pre activation patching for the hi-confused case.

SOURCE: Alice speaks first, greeted as "Hi Alice!" → model should predict France
BASE:   Bob speaks first, greeted as "Hi Bob!"    → model predicts Thailand

Usage:
  python plot_resid_pre_onecase.py --model llama --gpu 0
  python plot_resid_pre_onecase.py --model llama --gpu 0 --prob-diff
"""

import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model", choices=["llama","llamabase", "qwen"], default="llama")
parser.add_argument("--gpu",   default="0")
parser.add_argument("--prob-diff", action="store_true",
                    help="Plot P(src) - P(base) instead of normalized LD recovery")
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch.nn.functional as F
from transformer_lens import HookedTransformer, patching

import sys
sys.path.insert(0, os.path.dirname(__file__))
from utils import MODEL_CONFIGS

# ── Case definition ───────────────────────────────────────────────────────────
PROMPT_SOURCE = (
    'This is the transcript of a conversation.\n'
    '"I am Alice."\n"I am Bob."\n'
    '"Hi Alice!"\n'
    '"I live in France. What about you?"\n'
    '"I live in Thailand."\n'
    'Question: Where does Alice live? Answer: Alice lives in'
)
PROMPT_BASE = (
    'This is the transcript of a conversation.\n'
    '"I am Bob."\n"I am Alice."\n'
    '"Hi Bob!"\n'
    '"I live in France. What about you?"\n'
    '"I live in Thailand."\n'
    'Question: Where does Alice live? Answer: Alice lives in'
)
SOURCE_ANSWER = " France"
BASE_ANSWER   = " Thailand"

SKIP_PREFIX = 'This is the transcript of a conversation.\n'

# ── Paper figure style ────────────────────────────────────────────────────────
FONT_SIZE_LABEL  = 10
FONT_SIZE_TICKS  = 10
FONT_SIZE_XTICKS = 10
DPI              = 300
FIG_WIDTH        = 10
FIG_HEIGHT       = 6

BASE_OUTDIR = "/mnt/ssd/aryawu/role-representation/attribution_ioi/results"


def _plot(data, tok_repr, corrupted_tok_repr, diff_pos,
          n_skip, outpath, vmin, vmax, cbar_label):
    data_show  = data[:, n_skip:]
    tok_show   = tok_repr[n_skip:]
    ctok_show  = corrupted_tok_repr[n_skip:]
    diff_show  = [p - n_skip for p in diff_pos if p >= n_skip]

    fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT))
    im = ax.imshow(data_show, aspect='auto', cmap='RdBu_r',
                   origin='lower', vmin=vmin, vmax=vmax)

    cbar = plt.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label(cbar_label, fontsize=FONT_SIZE_LABEL)
    cbar.ax.tick_params(labelsize=FONT_SIZE_TICKS)

    for pos in diff_show:
        ax.axvline(x=pos - 0.5, color='orange', alpha=0.6, linewidth=1.5)
        ax.axvline(x=pos + 0.5, color='orange', alpha=0.6, linewidth=1.5)

    labels = list(tok_show)
    for i in diff_show:
        labels[i] = tok_show[i] + '\n(' + ctok_show[i] + ')'
    ax.set_xticks(range(len(labels)))
    xlabels = ax.set_xticklabels(labels, rotation=90, ha='center',
                                  fontsize=FONT_SIZE_XTICKS)
    for i in diff_show:
        xlabels[i].set_color('red')
        xlabels[i].set_fontweight('bold')

    ax.set_yticks(range(data_show.shape[0]))
    ax.tick_params(axis='y', labelsize=FONT_SIZE_TICKS)
    ax.set_xlabel("Token Position", fontsize=FONT_SIZE_LABEL)
    ax.set_ylabel("Layer",          fontsize=FONT_SIZE_LABEL)

    plt.tight_layout()
    plt.savefig(outpath, dpi=DPI, bbox_inches='tight')
    plt.close()
    print(f"  Saved {outpath}")


# ── Load model ────────────────────────────────────────────────────────────────
cfg = MODEL_CONFIGS[args.model]
print(f"Loading {cfg['name']}...")
model = HookedTransformer.from_pretrained(
    cfg["name"], dtype=torch.bfloat16, cache_dir=cfg["cache"])
model.eval().to("cuda")
print(f"Loaded: {model.cfg.n_layers}L {model.cfg.n_heads}H")

# ── Tokenize ──────────────────────────────────────────────────────────────────
clean_tokens     = model.to_tokens(PROMPT_SOURCE, prepend_bos=True)
corrupted_tokens = model.to_tokens(PROMPT_BASE,   prepend_bos=True)
assert clean_tokens.shape == corrupted_tokens.shape, \
    f"Token length mismatch: {clean_tokens.shape[1]} vs {corrupted_tokens.shape[1]}"

clean_ids     = clean_tokens[0].tolist()
corrupted_ids = corrupted_tokens[0].tolist()
diff_pos           = [i for i, (c, d) in enumerate(zip(clean_ids, corrupted_ids)) if c != d]
tok_repr           = [repr(model.tokenizer.decode(t)) for t in clean_ids]
corrupted_tok_repr = [repr(model.tokenizer.decode(t)) for t in corrupted_ids]

prefix_ids = model.tokenizer.encode(SKIP_PREFIX, add_special_tokens=False)
n_skip = 1 + len(prefix_ids)

# ── Token IDs for metric ──────────────────────────────────────────────────────
with torch.no_grad():
    clean_logits_raw     = model(clean_tokens)
    corrupted_logits_raw = model(corrupted_tokens)

src_id  = model.tokenizer.encode(SOURCE_ANSWER, add_special_tokens=False)[0]
base_id = model.tokenizer.encode(BASE_ANSWER,   add_special_tokens=False)[0]
src_str  = model.tokenizer.decode(src_id).strip()
base_str = model.tokenizer.decode(base_id).strip()

clean_ld     = (clean_logits_raw[0, -1, src_id] - clean_logits_raw[0, -1, base_id]).item()
corrupted_ld = (corrupted_logits_raw[0, -1, src_id] - corrupted_logits_raw[0, -1, base_id]).item()
print(f"SOURCE LD={clean_ld:.3f}  BASE LD={corrupted_ld:.3f}")

# ── Metric ────────────────────────────────────────────────────────────────────
if args.prob_diff:
    def metric(logits):
        probs = F.softmax(logits[0, -1].float(), dim=-1)
        return probs[src_id] - probs[base_id]
    vabs = 0.5
    vmin, vmax = -vabs, vabs
    cbar_label = f"P({src_str}) − P({base_str})"
    suffix = "_probdiff"
else:
    def metric(logits):
        ld    = logits[0, -1, src_id] - logits[0, -1, base_id]
        denom = clean_ld - corrupted_ld
        if abs(denom) < 0.01:
            return ld / (abs(clean_ld) + 1e-6)
        return (ld - corrupted_ld) / denom
    vmin, vmax = 0, 1
    cbar_label = f"Logit diff recovery (0 = LD_dst, 1 = LD_src)"
    suffix = ""

# ── Patching ──────────────────────────────────────────────────────────────────
seq_len = clean_tokens.shape[1]
print(f"resid_pre patching ({seq_len * model.cfg.n_layers} passes)...")
_, clean_cache = model.run_with_cache(clean_tokens)
with torch.no_grad():
    act_patch = patching.get_act_patch_resid_pre(
        model=model, corrupted_tokens=corrupted_tokens,
        clean_cache=clean_cache, patching_metric=metric)
data = act_patch.float().cpu().numpy()
del clean_cache

# ── Save ──────────────────────────────────────────────────────────────────────
outdir = os.path.join(BASE_OUTDIR, args.model, "hi_confused")
os.makedirs(outdir, exist_ok=True)
out_pdf = os.path.join(outdir, f"resid_pre_hi_confused{suffix}.pdf")

_plot(data, tok_repr, corrupted_tok_repr, diff_pos,
      n_skip, out_pdf, vmin, vmax, cbar_label)

print("Done.")
