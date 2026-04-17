"""
counter_token_heatmap_all.py — Binding-ID probe score heatmap over ALL tokens.

Shows P(△) at every token position × every layer for two fixed prompts:
  - Normal (OOD):  "Hi Bob!" → Alice=△=France, Bob=□=Thailand
  - Confused:      "Hi Alice!" → Alice=△=France (model may get confused)

Unlike counter_token_heatmap.py, no tokens are masked — all positions shown.
Key tokens (entity names, "I", countries) are bolded on the y-axis.

Usage:
  python counter_token_heatmap_all.py --model llama --gpu 0
  python counter_token_heatmap_all.py --model llama --gpu 0 --hi-only
"""
import os, sys, argparse
import numpy as np
import torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize

parser = argparse.ArgumentParser()
parser.add_argument("--model",   choices=["llama","qwen"], default="llama")
parser.add_argument("--gpu",     default="0")
parser.add_argument("--n-probe", type=int, default=100)
parser.add_argument("--hi-only", action="store_true", help="use hi-only probe")
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR  = os.path.dirname(SCRIPT_DIR)
OUTDIR      = SCRIPT_DIR

sys.path.insert(0, PARENT_DIR)
sys.path.insert(0, os.path.join(PARENT_DIR, "../../attribution_ioi"))

from probe_io import cache_path, cache_path_hi_only, load_probes
from utils import MODEL_CONFIGS
from transformer_lens import HookedTransformer

cfg = MODEL_CONFIGS[args.model]
print(f"Loading {cfg['name']}...")
model = HookedTransformer.from_pretrained(cfg["name"], dtype=torch.bfloat16, cache_dir=cfg["cache"])
model.eval().to("cuda")
NL = model.cfg.n_layers
print(f"Loaded: {NL} layers")

# ── Fixed prompts ──────────────────────────────────────────────────────────────
# Normal (OOD): Alice=△ speaks France, Bob=□ speaks Thailand
PROMPT_NORMAL = (
    'This is the transcript of a conversation.\n'
    '"I am Alice."\n'
    '"I am Bob."\n'
    '"Hi Bob!"\n'
    '"Where do you live?"\n'
    '"I live in France. What about you?"\n'
    '"I live in Thailand."\n'
    'Question: Where does Alice live? Answer: Alice lives in'
)

# Confused: "Hi Alice!" flips who speaks what — Alice=△ should speak France
# but model gets confused by the "Hi Alice!" greeting pattern
PROMPT_CONFUSED = (
    'This is a transcript of a conversation.\n'
    '"I am Alice."\n'
    '"I am Bob."\n'
    '"Hi Alice!"\n'
    '"I live in Thailand. What about you?"\n'
    '"I live in France."\n'
    'Question: Where does Alice live?\nAnswer: Alice lives in'
)

KEY_WORDS = {"Alice", "Bob", "France", "Thailand"}

# ── Load probes ────────────────────────────────────────────────────────────────
_cache = (cache_path_hi_only(PARENT_DIR, args.model, args.n_probe)
          if args.hi_only else cache_path(PARENT_DIR, args.model, args.n_probe))
if not os.path.exists(_cache):
    print(f"No cached probes at {_cache}")
    sys.exit(1)
probes = load_probes(_cache)
probe_label = "hi-only probe" if args.hi_only else "hi+hi-confused probe"
print(f"Using {probe_label}")

# ── Helpers ────────────────────────────────────────────────────────────────────
def tok_label(t):
    s = model.tokenizer.decode([t])
    return repr(s)[1:-1][:14]   # escape special chars, cap length

def is_key_token(tok_str):
    clean = tok_str.strip().strip('"').strip("'")
    return any(kw.lower() in clean.lower() for kw in KEY_WORDS) or clean in ("I", "i")

@torch.no_grad()
def run_prompt(prompt):
    ids = model.tokenizer.encode(prompt)
    tok = torch.tensor([ids], device="cuda")
    _, cache = model.run_with_cache(tok, names_filter=lambda n: "hook_resid_post" in n)
    seq_len = len(ids)
    # all_acts: (NL, seq_len, d_model)
    all_acts = np.stack([
        cache[f"blocks.{L}.hook_resid_post"][0, :seq_len].cpu().float().numpy()
        for L in range(NL)
    ])
    # mat: (NL, seq_len) — probe score P(△) at each position
    mat = np.zeros((NL, seq_len))
    for L in range(NL):
        mat[L] = probes[L].predict_proba(all_acts[L])[:, 1]
    labels = [tok_label(t) for t in ids]
    return mat, labels, ids

# ── Run both prompts ───────────────────────────────────────────────────────────
print("Running normal prompt...")
mat_n, labels_n, ids_n = run_prompt(PROMPT_NORMAL)
print(f"  seq_len={len(ids_n)}")

print("Running confused prompt...")
mat_c, labels_c, ids_c = run_prompt(PROMPT_CONFUSED)
print(f"  seq_len={len(ids_c)}")

# Print model predictions
def top_token(prompt):
    ids = model.tokenizer.encode(prompt)
    tok = torch.tensor([ids], device="cuda")
    with torch.no_grad():
        logits = model(tok)[0, -1].cpu().float().numpy()
    top_id = int(np.argmax(logits))
    return model.tokenizer.decode([top_id])

print(f"\nModel prediction — Hi:   '{top_token(PROMPT_NORMAL).strip()}'")
print(f"Model prediction — Hi-reset: '{top_token(PROMPT_CONFUSED).strip()}'")

# ── Plot ───────────────────────────────────────────────────────────────────────
CMAP = plt.cm.RdBu_r
RED   = "#d7191c"
BLUE  = "#2166ac"

def draw_panel(ax, mat, labels, title):
    seq_len = len(labels)
    mat_h = mat.T   # (seq_len, NL)

    im = ax.imshow(mat_h, aspect="auto", cmap=CMAP, vmin=0, vmax=1,
                   origin="upper", interpolation="nearest")
    ax.set_xlim(-0.5, NL - 0.5)
    ax.set_ylim(seq_len - 0.5, -0.5)

    ax.set_xticks(range(0, NL, 2))
    ax.set_xticklabels([str(l) for l in range(0, NL, 2)], fontsize=8)
    ax.set_xlabel("Layer", fontsize=10)

    ax.set_yticks(range(seq_len))
    ax.set_yticklabels(labels, fontsize=7, family="monospace")
    for tick, lbl in zip(ax.get_yticklabels(), labels):
        key = is_key_token(lbl)
        tick.set_fontweight("bold" if key else "normal")
        tick.set_fontsize(8 if key else 6.5)

    # Light vertical grid every 4 layers
    for l in range(0, NL, 4):
        ax.axvline(l - 0.5, color="white", lw=0.4, alpha=0.5, zorder=3)

    ax.set_title(title, fontsize=10, fontweight="bold")
    return im

seq_n = len(labels_n)
seq_c = len(labels_c)
row_h = 0.28
panel_w = max(5, NL * 0.18)
cbar_w = 0.4

fig_w = 2 * panel_w + cbar_w + 1.2
fig_h = max(seq_n, seq_c) * row_h + 1.2

fig, axes = plt.subplots(1, 2, figsize=(fig_w, fig_h))

# Pad shorter sequence heatmap so both panels have same y extent
max_seq = max(seq_n, seq_c)

im = draw_panel(axes[0], mat_n, labels_n,
                'Normal: "Hi Bob!"\n(Alice=△=France, Bob=□=Thailand)')
draw_panel(axes[1], mat_c, labels_c,
           'Confused: "Hi Alice!"\n(Alice=△=France, Bob=□=Thailand)')

# Shared colorbar
fig.subplots_adjust(right=0.88, wspace=0.35)
cbar_ax = fig.add_axes([0.90, 0.08, 0.018, 0.84])
sm = plt.cm.ScalarMappable(cmap=CMAP, norm=Normalize(0, 1))
sm.set_array([])
cbar = fig.colorbar(sm, cax=cbar_ax)
cbar.set_label("P(△)  Binding-ID probe score", fontsize=10, labelpad=8)
cbar.ax.tick_params(labelsize=8)

fig.suptitle(
    f"Binding-ID probe score — ALL tokens  ({cfg['name']})\n"
    f"{probe_label}  |  Bold y-labels = key tokens (names, countries, 'I')\n"
    "Red = P(△)=1 (△/Alice bound), Blue = P(△)=0 (□/Bob bound)",
    fontsize=10, y=1.01)

_suffix = "_hi_only" if args.hi_only else ""
out = os.path.join(OUTDIR, f"counter_token_heatmap_all_{args.model}{_suffix}.png")
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.close()
print(f"\nSaved {out}")
print("Done.")
