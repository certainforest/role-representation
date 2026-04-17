"""
probe_score_heatmap_v2.py — Compact transposed heatmap of binding-ID probe scores.

Layout:
  - Rows    = layers (0 → N, bottom to top)
  - Columns = entity/attribute token occurrences only (no grey filler rows)
  - Top annotation bar: utterance label, colored by speaker
  - Column header: token label (Alice₁, Bob₁, ...), colored blue (△) or red (□)
  - Vertical separators between utterance groups

Usage:
  python probe_score_heatmap_v2.py --model llama --gpu 0
"""

import os, sys, argparse
import numpy as np
import torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize
from matplotlib import cm

parser = argparse.ArgumentParser()
parser.add_argument("--model",   choices=["llama", "qwen"], default="llama")
parser.add_argument("--gpu",     default="0")
parser.add_argument("--n-probe",  type=int, default=100)
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
model = HookedTransformer.from_pretrained(
    cfg["name"], dtype=torch.bfloat16, cache_dir=cfg["cache"]
)
model.eval().to("cuda")
NL = model.cfg.n_layers
print(f"Loaded: {NL} layers")

# ── Prompt ────────────────────────────────────────────────────────────────────
PROMPT_PREFIX = 'This is a transcript of a conversation between two speakers.\n'
PROMPT = (
    PROMPT_PREFIX +
    '"I am Alice."\n'
    '"I am Bob."\n'
    '"Hi Bob. I heard that you live in France."\n'
    '"Hi Alice! No, i live in Thailand. Where do you live?"\n'
    '"I live in France. I love soccer. What sport do you like?"\n'
    '"I like basketball."\n'
    'Question: What sport does Alice like? Answer: Alice likes'
)

print("\nPrompt:\n", PROMPT, "\n")

ids       = model.tokenizer.encode(PROMPT)
tokens_str = [model.tokenizer.decode([t]) for t in ids]

print("Tokens:")
for i, ts in enumerate(tokens_str):
    print(f"  {i:3d}: {repr(ts)}")

# ── Find all occurrences of each word of interest ─────────────────────────────
def find_all(ids, word):
    for prefix in [f" {word}", word]:
        toks = model.tokenizer.encode(prefix, add_special_tokens=False)
        tid  = toks[0]
        hits = [i for i, t in enumerate(ids) if t == tid]
        if hits:
            if len(toks) > 1:
                print(f"  NOTE: '{word}' is multi-token, tracking first token only")
            return hits
    return []

# ground_truth per word (default, used when no per-occurrence override applies)
# "tri" = △ (Speaker 1 / Alice), "sq" = □ (Speaker 2 / Bob)
WORDS = {
    "Alice":      "tri",
    "Bob":        "sq",
    "France":     "tri",
    "Thailand":   "sq",
    "soccer":     "tri",
    "basketball": "sq",
}

# Per-occurrence ground truth overrides: (word, 1-based occurrence index) → gt
# France₁ appears in Alice's utterance quoting Bob ("I heard you live in France")
# — it is being attributed to Bob in that context, so it is □-like.
# France₂ appears in Alice's own statement ("I live in France") — △-like.
# All other words follow WORDS defaults.
GT_OVERRIDE = {
    ("France", 1): "sq",   # France₁ — misattributed to Bob
    ("France", 2): "tri",  # France₂ — Alice's actual country
}

word_positions = {}
for w, gt in WORDS.items():
    word_positions[w] = find_all(ids, w)
    print(f"  {w:10s} ({gt}): {word_positions[w]}")

# ── Load probes ───────────────────────────────────────────────────────────────
_cache = (cache_path_hi_only(PARENT_DIR, args.model, args.n_probe)
          if args.hi_only else cache_path(PARENT_DIR, args.model, args.n_probe))
if not os.path.exists(_cache):
    print(f"No cached probes at {_cache} — run train_probes{'_hi_only' if args.hi_only else ''}.py first")
    sys.exit(1)
probes = load_probes(_cache)

def probe_score(resid_np, L):
    return float(probes[L].predict_proba(resid_np[L:L+1])[0][1])

# ── Run model ─────────────────────────────────────────────────────────────────
print("\nRunning model...")
tok = torch.tensor([ids], device="cuda")
with torch.no_grad():
    _, cache = model.run_with_cache(
        tok, names_filter=lambda n: "hook_resid_post" in n
    )

seq_len  = len(ids)
all_resid = np.stack([
    np.stack([
        cache[f"blocks.{L}.hook_resid_post"][0, pos].cpu().float().numpy()
        for L in range(NL)
    ])
    for pos in range(seq_len)
])  # [seq_len, NL, d_model]

# ── Build column list ─────────────────────────────────────────────────────────
# Each column = one token occurrence, ordered by position in the prompt.
# Label: Alice₁, Alice₂, ..., Bob₁, Thailand₁, Thailand₂, France, etc.
# We also tag each occurrence with its utterance index.

# Define utterance boundaries by character search in prompt
# (simpler: assign utterance by checking which sentence the token falls in)

# Map token positions to utterance index by finding the sentence boundaries
# We'll do this by tokenizing each utterance substring and tracking cumulative pos
utt_strings = [
    PROMPT_PREFIX,
    '"I am Alice."\n',
    '"I am Bob."\n',
    '"Hi Bob. I heard that you live in France."\n',
    '"Hi Alice! No, i live in Thailand. Where do you live?"\n',
    '"I live in France. I love soccer. What sport do you like?"\n',
    '"I like basketball."\n',
    'Question: Where does Alice live? Answer: Alice lives in',
]

utt_labels = [
    "prefix",
    '"I am Alice."',
    '"I am Bob."',
    '"Hi Bob. \nI heard that you live in France."',
    '"Hi Alice! \nNo, I live in Thailand."',
    '"I live in France.\nI love soccer."',
    '"I like basketball."',
    "Question: Where does Alice live?\nAnswer: Alice lives in",
]

utt_speaker = [0, 1, 2, 1, 2, 1, 2, 0]   # 0=narrator, 1=spk1(△/Alice), 2=spk2(□/Bob)

# Build pos→utterance_idx map
pos_to_utt = {}
cursor = 0
for ui, s in enumerate(utt_strings):
    toks_u = model.tokenizer.encode(s, add_special_tokens=False)
    for j in range(len(toks_u)):
        pos_to_utt[cursor + j] = ui
    cursor += len(toks_u)

# Build ordered column list
subscript = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
occurrence_count = {w: 0 for w in WORDS}

columns = []  # list of dicts: pos, word, gt, label, utt_idx, spk
for pos in range(seq_len):
    for word, default_gt in WORDS.items():
        if pos in word_positions[word]:
            occurrence_count[word] += 1
            occ = occurrence_count[word]
            n = str(occ).translate(subscript)
            label = f"{word}{n}" if len(word_positions[word]) > 1 else word
            utt_idx = pos_to_utt.get(pos, 0)
            gt = GT_OVERRIDE.get((word, occ), default_gt)
            columns.append(dict(
                pos=pos, word=word, gt=gt,
                label=label, utt_idx=utt_idx,
                spk=utt_speaker[utt_idx]
            ))

columns.sort(key=lambda c: c["pos"])
print(f"\nColumns ({len(columns)}):")
for c in columns:
    print(f"  {c['label']:12s}  pos={c['pos']:3d}  utt={c['utt_idx']}  spk={c['spk']}")

# ── Compute probe scores for each column ──────────────────────────────────────
N_COL = len(columns)
mat   = np.zeros((NL, N_COL))  # rows=layers, cols=token occurrences

for ci, col in enumerate(columns):
    resid = all_resid[col["pos"]]  # [NL, d_model]
    mat[:, ci] = np.array([probe_score(resid, L) for L in range(NL)])

# ── Plot ───────────────────────────────────────────────────────────────────────
# New layout:
#   - rows    = token occurrences (ordered by position in prompt, top→bottom)
#   - columns = layers (x-axis, 0 → NL-1)
#   - left panel: transcript text (colored words inline)
#   - right: colorbar
#   - top: legend

CMAP = plt.cm.RdBu_r
GT_COLORS = {"tri": "#d7191c", "sq": "#2166ac"}   # green=△, red=□
UTT_BG    = {
    0: "#eeeeee",   # narrator
    1: "#FCEBEB",   # speaker 1 (△) — light red
    2: "#E6F1FB",   # speaker 2 (□) — light blue
}
SEP_COLOR = "white"

N_ROW = N_COL   # one row per token occurrence
# mat is currently [NL, N_COL]; we need [N_ROW, NL] for imshow
mat_h = mat.T   # [N_ROW, NL]  — rows=tokens, cols=layers

# ── Figure dimensions ─────────────────────────────────────────────────────────
row_h      = 0.38   # inches per token row
label_w    = 2.5    # inches for left transcript panel (narrower)
heatmap_w  = max(4, NL * 0.18)
cbar_w     = 0.35
leg_h_in   = 0.55

fig_w = label_w + heatmap_w + cbar_w + 0.6
fig_h = N_ROW * row_h + leg_h_in + 0.6

fig = plt.figure(figsize=(fig_w, fig_h))

gs = fig.add_gridspec(
    2, 3,
    height_ratios=[leg_h_in, N_ROW * row_h],
    width_ratios=[label_w, heatmap_w, cbar_w + 0.3],
    hspace=0.02, wspace=0.02,
    left=0.01, right=0.99,
    top=0.99, bottom=0.04
)

ax_leg   = fig.add_subplot(gs[0, :])
ax_label = fig.add_subplot(gs[1, 0])
ax_heat  = fig.add_subplot(gs[1, 1])
ax_cbar  = fig.add_subplot(gs[1, 2])

# ── Legend ────────────────────────────────────────────────────────────────────
ax_leg.axis("off")
legend_elements = [
    mpatches.Patch(facecolor=GT_COLORS["tri"], label="Speaker 1 (▲) binding ID"),
    mpatches.Patch(facecolor=GT_COLORS["sq"],  label="Speaker 2 (■) binding ID"),
    mpatches.Patch(facecolor=UTT_BG[1],        label="Speaker 1 utterance"),
    mpatches.Patch(facecolor=UTT_BG[2],        label="Speaker 2 utterance"),
    mpatches.Patch(facecolor=UTT_BG[0],        label="Narrator / query"),
]
ax_leg.legend(
    handles=legend_elements,
    loc="center", ncol=5,
    fontsize=10, frameon=False,
    handlelength=1.2, handleheight=1.0, columnspacing=1.0
)

# ── Heatmap ───────────────────────────────────────────────────────────────────
im = ax_heat.imshow(
    mat_h, aspect="auto", cmap=CMAP, vmin=0, vmax=1,
    origin="upper", interpolation="nearest"
)
ax_heat.set_xlim(-0.5, NL - 0.5)
ax_heat.set_ylim(N_ROW - 0.5, -0.5)

# X axis: layer numbers
ax_heat.set_xticks(range(0, NL, 2))
ax_heat.set_xticklabels([str(l) for l in range(0, NL, 2)], fontsize=9)
ax_heat.set_xlabel("Layer", fontsize=12)

# No y-tick labels on heatmap — transcript panel on the left is the label
ax_heat.set_yticks(range(N_ROW))
ax_heat.set_yticklabels([""] * N_ROW)
ax_heat.yaxis.set_tick_params(length=0)

# Horizontal separators between utterance groups
prev_utt = columns[0]["utt_idx"]
for ri, col in enumerate(columns[1:], 1):
    if col["utt_idx"] != prev_utt:
        ax_heat.axhline(ri - 0.5, color=SEP_COLOR, lw=2.0, zorder=5)
        prev_utt = col["utt_idx"]

# Light vertical grid every 4 layers
for l in range(0, NL, 4):
    ax_heat.axvline(l - 0.5, color="white", lw=0.4, alpha=0.4, zorder=3)

# ── Colorbar ──────────────────────────────────────────────────────────────────
ax_cbar.set_visible(False)
sm = plt.cm.ScalarMappable(cmap=CMAP, norm=Normalize(0, 1))
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax_cbar, fraction=0.9, pad=0.05)
cbar.set_label("Binding ID probe score  P(▲)", fontsize=11, labelpad=8)
cbar.ax.tick_params(labelsize=9)

# ── Left transcript panel ─────────────────────────────────────────────────────
# x: 0..1, y: row index (0=top, N_ROW-1=bottom), matching heatmap rows
ax_label.set_xlim(0, 1)
ax_label.set_ylim(N_ROW - 0.5, -0.5)
ax_label.axis("off")

# Build utterance spans (row index ranges)
utt_spans = {}
for ri, col in enumerate(columns):
    ui = col["utt_idx"]
    if ui not in utt_spans:
        utt_spans[ui] = [ri, ri]
    else:
        utt_spans[ui][1] = ri

# Background color bands
for ui, (ri_start, ri_end) in sorted(utt_spans.items()):
    spk   = utt_speaker[ui]
    ax_label.add_patch(mpatches.Rectangle(
        (0, ri_start - 0.48), 1.0, ri_end - ri_start + 0.96,
        facecolor=UTT_BG[spk], edgecolor="none", zorder=1
    ))

# White separators
prev_utt = columns[0]["utt_idx"]
for ri, col in enumerate(columns[1:], 1):
    if col["utt_idx"] != prev_utt:
        ax_label.axhline(ri - 0.5, color="white", lw=2.0, zorder=5)
        prev_utt = col["utt_idx"]

# Also draw matching separators on heatmap background
prev_utt = columns[0]["utt_idx"]
for ri, col in enumerate(columns[1:], 1):
    if col["utt_idx"] != prev_utt:
        ax_heat.axhline(ri - 0.5, color="white", lw=2.0, zorder=5)
        prev_utt = col["utt_idx"]

# ── helper: word→gt color for a given utterance ───────────────────────────────
def word_color_in_utt(w_clean, ui):
    for col in columns:
        if col["utt_idx"] == ui and col["word"].lower() == w_clean.lower():
            return GT_COLORS[col["gt"]]
    return None

# ── Transcript text rendering — two-pass for accurate inline coloring ──────────
# Pass 1: render all words in black to get their actual bounding boxes.
# Pass 2: re-color key words by setting their color property directly.
# This avoids all character-width estimation.

FONTSIZE    = 9.0

def word_color_in_utt(w_clean, ui):
    for col in columns:
        if col["utt_idx"] == ui and col["word"].lower() == w_clean.lower():
            return GT_COLORS[col["gt"]]
    return None

drawn_utts  = set()
all_word_texts = []   # list of (text_obj, color) for pass-2 recoloring

for ui, (ri_start, ri_end) in sorted(utt_spans.items()):
    if ui in drawn_utts:
        continue
    drawn_utts.add(ui)

    # Split on explicit newlines — no char-count wrapping
    lines_out = [line.split() for line in utt_labels[ui].split("\n") if line.strip()]

    n_lines    = len(lines_out)
    # Each line aligns with one strip row: line i → row (ri_start + i)
    # This makes text sit centered in its own strip band.

    for li, line_words in enumerate(lines_out):
        ly = ri_start + li   # line i sits at row ri_start+i in data coords

        # Render the full line left-aligned from x=0.02, black.
        # We place each word individually with ha='left' so we can
        # measure each word's width and advance x precisely.
        # Use a temporary joined string to find the starting x for centering.
        # Step 1: place all words at x=0 temporarily to measure widths.
        temp_texts = []
        for w in line_words:
            t = ax_label.text(
                0, ly, w + " ",   # include trailing space for spacing
                ha='left', va='center',
                fontsize=FONTSIZE, color='black', fontweight='normal',
                zorder=4, clip_on=True
            )
            temp_texts.append(t)

        # Draw canvas to get bounding boxes
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()

        # Measure each word's pixel width
        word_pxw = []
        for t in temp_texts:
            bb = t.get_window_extent(renderer)
            word_pxw.append(bb.width)
        # Remove temporaries
        for t in temp_texts:
            t.remove()

        # Convert px widths to ax_label data coords
        ax_bb   = ax_label.get_window_extent(renderer)
        px2data = 1.0 / ax_bb.width   # ax xlim is 0..1

        total_w_data = sum(w * px2data for w in word_pxw)
        x_data = 0.5 - total_w_data / 2.0

        # Pass 1: place every word at its correct x, in black
        placed = []
        for wi, w in enumerate(line_words):
            ww_data = word_pxw[wi] * px2data
            t = ax_label.text(
                x_data, ly, w,
                ha='left', va='center',
                fontsize=FONTSIZE, color='black', fontweight='normal',
                zorder=4, clip_on=True
            )
            w_clean = w.strip(".,!?\"'")
            col_c   = word_color_in_utt(w_clean, ui)
            placed.append((t, col_c))
            x_data += ww_data

        # Pass 2: recolor key words
        for t, col_c in placed:
            if col_c:
                t.set_color(col_c)
                t.set_fontweight('bold')

_suffix = "_hi_only" if args.hi_only else ""
out = os.path.join(OUTDIR, f"probe_score_heatmap_v2_{args.model}{_suffix}.png")
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.close()
print(f"\nSaved {out}")
print("Done.")