"""
probe_score_heard.py — Probe binding-ID score at every token of interest.

Tests probe generalization to a naturalistic "heard" transcript:

  I am Alice.
  I am Bob.
  Hi Bob. I heard that you live in Thailand.
  Hi Alice. Where do you live?
  I live in France.
  Where does Alice live?

Ground truth binding:
  Alice → France  (△, queried entity)
  Bob   → Thailand (□, other entity)

For each occurrence of Alice, Bob, Thailand, France in the prompt,
we extract the residual stream across all layers and compute the probe's
P(△) score. If the probe direction generalizes, Alice/France tokens
should score high (△-like) and Bob/Thailand should score low (□-like).

Outputs:
  probe_score_heard_{model}.png  — P(△) vs layer, one line per token occurrence
  probe_score_heatmap_{model}.png — tokens × layers heatmap

Usage:
  python probe_score_heard.py --model llama --gpu 0
"""
import os, sys, argparse
import numpy as np
import torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("--model",   choices=["llama","qwen"], default="llama")
parser.add_argument("--gpu",     default="0")
parser.add_argument("--n-probe", type=int, default=100)
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR  = os.path.dirname(SCRIPT_DIR)   # visualize_counter/
OUTDIR      = SCRIPT_DIR

sys.path.insert(0, PARENT_DIR)
sys.path.insert(0, os.path.join(PARENT_DIR, "../../attribution_ioi"))

from probe_io import cache_path, load_probes
from utils import MODEL_CONFIGS
from transformer_lens import HookedTransformer

cfg = MODEL_CONFIGS[args.model]
print(f"Loading {cfg['name']}...")
model = HookedTransformer.from_pretrained(cfg["name"], dtype=torch.bfloat16, cache_dir=cfg["cache"])
model.eval().to("cuda")
NL = model.cfg.n_layers
print(f"Loaded: {NL} layers")

# ── Prompt ─────────────────────────────────────────────────────────────────────
PROMPT_PREFIX = 'This is a transcript of a conversation between two speakers.\n'
# PROMPT = (
#     PROMPT_PREFIX +
#     '"I am Alice."\n'
#     '"I am Bob."\n'
#     '"Hi Bob. I heard that you live in France."\n'
#     '"Hi Alice! No, i live in Thailand. Where do you live?"\n'
#     '"I live in France. I love soccer. What sport do you like?"\n'
#     '"I like basketball."\n'
#     'Question: Where does Alice live? Answer: Alice lives in'
# )
PROMPT = (
    PROMPT_PREFIX +
    '"I am Alice."\n'
    '"I am Bob."\n'
    '"Hi Bob!"\n'
    '"I live in Thailand. What about you?"\n'
    '"I live in France."\n'
    'Question: Where does Alice live? Answer: Alice lives in'
)

PROMPT_CONFUSED = (
    PROMPT_PREFIX +
    '"I am Alice."\n'
    '"I am Bob."\n'
    '"Hi Alice!"\n'
    '"I live in Thailand. What about you?"\n'
    '"I live in France."\n'
    'Question: Where does Alice live? Answer: Alice lives in'
)

print("\nPrompt:")
print(PROMPT)
print()

# ── Tokenize and find positions of interest ────────────────────────────────────
ids = model.tokenizer.encode(PROMPT)
tokens_str = [model.tokenizer.decode([t]) for t in ids]

# Number of prefix tokens to hide from the heatmap (still fed to the model)
display_start = len(model.tokenizer.encode(PROMPT_PREFIX))

print("Tokens:")
for i, (tid, ts) in enumerate(zip(ids, tokens_str)):
    print(f"  {i:3d}: {repr(ts)}")

def find_all(ids, word):
    """Return all positions where ' {word}' or '{word}' tokenizes to a match."""
    # Try with leading space first (mid-sentence), then without (start of sentence)
    candidates = []
    for prefix in [f" {word}", word]:
        toks = model.tokenizer.encode(prefix, add_special_tokens=False)
        if len(toks) == 1:
            tid = toks[0]
            positions = [i for i, t in enumerate(ids) if t == tid]
            if positions:
                candidates = positions
                break
        else:
            # Multi-token word: find first token of the word in sequence
            tid = toks[0]
            positions = [i for i, t in enumerate(ids) if t == tid]
            if positions:
                print(f"  NOTE: '{word}' is multi-token {toks}, tracking first token only")
                candidates = positions
                break
    return candidates

WORDS_OF_INTEREST = {
    "Alice":   {"color": "#2166ac", "ground_truth": "△"},   # △-entity
    "Bob":     {"color": "#d73027", "ground_truth": "□"},   # □-entity
    "France":  {"color": "#74add1", "ground_truth": "△"},   # △-country
    "Thailand":{"color": "#f46d43", "ground_truth": "□"},   # □-country
    "soccer": {"color": "#95bad0", "ground_truth": "△"},   # △-sport
    "basketball": {"color": "#f29174", "ground_truth": "□"},   # □-sport
}

positions = {}   # word → list of (pos, label)
for word, meta in WORDS_OF_INTEREST.items():
    pos_list = find_all(ids, word)
    positions[word] = pos_list
    print(f"  {word:10s} ({meta['ground_truth']}): positions {pos_list}")

# ── Load probes ────────────────────────────────────────────────────────────────
_cache = cache_path(PARENT_DIR, args.model, args.n_probe)
if not os.path.exists(_cache):
    print(f"No cached probes at {_cache} — run train_probes.py first")
    sys.exit(1)
print("\nLoading cached probes...")
probes = load_probes(_cache)

directions = np.stack([
    probes[L].coef_[0] / np.linalg.norm(probes[L].coef_[0])
    for L in range(NL)
])   # [NL, d_model]

def probe_score(resid_np, L):
    """P(△) from logistic regression probe at layer L."""
    return float(probes[L].predict_proba(resid_np[L:L+1])[0][1])

def raw_score(resid_np, L):
    """Signed dot product with probe direction (positive = △-like)."""
    return float(resid_np[L] @ directions[L])

# ── Run model with cache ───────────────────────────────────────────────────────
print("\nRunning model with cache...")
tok = torch.tensor([ids], device="cuda")
with torch.no_grad():
    _, cache = model.run_with_cache(tok, names_filter=lambda n: "hook_resid_post" in n)

# Extract residual stream for all positions: shape [seq_len, NL, d_model]
seq_len = len(ids)
all_resid = np.stack([
    np.stack([
        cache[f"blocks.{L}.hook_resid_post"][0, pos].cpu().float().numpy()
        for L in range(NL)
    ])
    for pos in range(seq_len)
])   # [seq_len, NL, d_model]

# Compute probe scores for all positions of interest
# scores[word][occurrence_idx] = np.array of shape [NL] with P(△) per layer
scores_prob = {}
scores_raw  = {}

for word, pos_list in positions.items():
    scores_prob[word] = []
    scores_raw[word]  = []
    for pos in pos_list:
        resid = all_resid[pos]   # [NL, d_model]
        layer_probs = np.array([probe_score(resid, L) for L in range(NL)])
        layer_raw   = np.array([raw_score(resid, L)   for L in range(NL)])
        scores_prob[word].append(layer_probs)
        scores_raw[word].append(layer_raw)

# ── Also compute model's actual answer ────────────────────────────────────────
final_logits = model(tok)[0, -1].detach().cpu().float().numpy()
top5 = np.argsort(final_logits)[-5:][::-1]
print("\nModel's top-5 completions for 'Where does Alice live?':")
for t in top5:
    print(f"  {repr(model.tokenizer.decode([t]))}: {final_logits[t]:.2f}")

# ── Figure: full token-sequence heatmap (reference style) ─────────────────────
# Rows = every token in the sequence, in order.
# Columns = layers.
# Colour = P(△) for words of interest; grey mask for everything else.
# Bold y-tick labels for words of interest.

# Build a set of (word, position) pairs that are "in distribution"
interest_pos = {}   # pos → (word, ground_truth)
for word, meta in WORDS_OF_INTEREST.items():
    for pos in positions[word]:
        interest_pos[pos] = (word, meta["ground_truth"])

# Only display tokens after the prefix — model still sees the full sequence
disp_positions = list(range(display_start, seq_len))
n_disp = len(disp_positions)

# Build display matrix: NaN where not a word of interest
mat = np.full((n_disp, NL), np.nan)
for di, pos in enumerate(disp_positions):
    if pos in interest_pos:
        resid = all_resid[pos]
        mat[di] = np.array([probe_score(resid, L) for L in range(NL)])

# Figure dimensions
row_h   = 0.32
fig_h   = max(8, n_disp * row_h + 2)
fig_w   = 12
fig, ax = plt.subplots(figsize=(fig_w, fig_h))

# Grey background
grey_mat = np.full((n_disp, NL), 0.5)
ax.imshow(grey_mat, aspect="auto", cmap="Greys", vmin=0, vmax=1,
          origin="upper", alpha=0.25)

# Overlay coloured cells for words of interest
import matplotlib.cm as cm
import matplotlib.colors as mcolors
cmap = plt.cm.RdYlGn
for di, pos in enumerate(disp_positions):
    if pos not in interest_pos:
        continue
    for L in range(NL):
        v = mat[di, L]
        if not np.isnan(v):
            ax.add_patch(plt.Rectangle(
                (L - 0.5, di - 0.5), 1, 1,
                color=cmap(v), zorder=2))

# Colorbar
sm = cm.ScalarMappable(cmap=cmap, norm=mcolors.Normalize(0, 1))
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax, shrink=0.4, pad=0.01)
cbar.set_label("P(△)", fontsize=10)

# Axes
ax.set_xlim(-0.5, NL - 0.5)
ax.set_ylim(n_disp - 0.5, -0.5)
ax.set_xticks(range(0, NL, 2))
ax.set_xticklabels([str(l) for l in range(0, NL, 2)], fontsize=8)
ax.set_xlabel("Layer", fontsize=11)

# Y-tick labels for displayed tokens only
ax.set_yticks(range(n_disp))
ytick_labels = [tokens_str[pos].replace("\n", "\\n") for pos in disp_positions]
ax.set_yticklabels(ytick_labels, fontsize=7.5)

# Bold and colour labels for words of interest
GT_COLOR = {"△": "#1a6faf", "□": "#b22222"}
for di, tick in enumerate(ax.get_yticklabels()):
    pos = disp_positions[di]
    if pos in interest_pos:
        word, gt = interest_pos[pos]
        tick.set_fontweight("bold")
        tick.set_fontsize(9)
        tick.set_color(GT_COLOR[gt])
    else:
        tick.set_color("#aaaaaa")

# Horizontal grid lines
for di in range(n_disp - 1):
    pos = disp_positions[di]
    pos_next = disp_positions[di + 1]
    lw    = 0.8 if (pos in interest_pos or pos_next in interest_pos) else 0.2
    alpha = 0.5 if lw > 0.5 else 0.2
    ax.axhline(di + 0.5, color="white", lw=lw, alpha=alpha, zorder=3)

ax.set_title(
    f"Binding-ID probe score  P(△)  — 'heard' transcript  ({cfg['name']})\n"
    "Alice→France (△, blue labels)  ·  Bob→Thailand (□, red labels)\n"
    "Green = △-like  ·  Red = □-like  ·  Grey = not an entity/country token",
    fontsize=10, pad=10)

plt.tight_layout()
out = os.path.join(OUTDIR, f"probe_score_heatmap_{args.model}o.png")
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.close()
print(f"\nSaved {out}")
print("Done.")
