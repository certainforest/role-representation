"""
probe_score_heatmap_v2_reset_with_I.py — Same as probe_score_heatmap_v2_reset.py
but also tracks the "I" token in "I live in Thailand" and "I live in France",
colored the same as the country token that follows it.

Usage:
  python probe_score_heatmap_v2_reset_with_I.py --model llama --gpu 0
"""

import os, sys, argparse
import numpy as np
import torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize

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

# ── Prompts ────────────────────────────────────────────────────────────────────
PROMPT_PREFIX = 'This is a transcript of a conversation between two speakers.\n'

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

# ── Utterance structure ────────────────────────────────────────────────────────
# Turn-taking: prefix(0), Alice(1), Bob(2), Alice(1), Alice(1), Bob(2), narrator(0)
UTT_SPEAKER = [0, 1, 2, 1, 1, 2, 0]

UTT_STRINGS = {
    "normal": [
        PROMPT_PREFIX,
        '"I am Alice."\n',
        '"I am Bob."\n',
        '"Hi Bob!"\n',
        '"I live in Thailand. What about you?"\n',
        '"I live in France."\n',
        'Question: Where does Alice live? Answer: Alice lives in',
    ],
    "confused": [
        PROMPT_PREFIX,
        '"I am Alice."\n',
        '"I am Bob."\n',
        '"Hi Alice!"\n',
        '"I live in Thailand. What about you?"\n',
        '"I live in France."\n',
        'Question: Where does Alice live? Answer: Alice lives in',
    ],
}

# ── Words of interest ──────────────────────────────────────────────────────────
# GT: tri=△=Speaker1=Alice (red), sq=□=Speaker2=Bob (blue)
WORDS = {
    "Alice":    "tri",
    "Bob":      "sq",
    "Thailand": "tri",   # Alice's country (turn 4, speaker 1)
    "France":   "sq",    # Bob's country   (turn 5, speaker 2)
}

HIGHLIGHT_WORDS = {"Thailand", "France"}
HIGHLIGHT_COLOR = "#E69F00"   # amber

# ── Colors ─────────────────────────────────────────────────────────────────────
RED   = "#d7191c"   # tri / Alice / Speaker 1
BLUE  = "#2166ac"   # sq  / Bob   / Speaker 2
BLACK = "black"

CMAP      = plt.cm.RdBu_r
GT_COLORS = {"tri": RED, "sq": BLUE}
UTT_BG    = {
    0: "#eeeeee",
    1: "#FCEBEB",   # speaker 1 (Alice)
    2: "#E6F1FB",   # speaker 2 (Bob)
}
SEP_COLOR = "white"
FONTSIZE  = 9.0

# ── Load probes ────────────────────────────────────────────────────────────────
_cache = (cache_path_hi_only(PARENT_DIR, args.model, args.n_probe)
          if args.hi_only else cache_path(PARENT_DIR, args.model, args.n_probe))
if not os.path.exists(_cache):
    print(f"No cached probes at {_cache} — run train_probes{'_hi_only' if args.hi_only else ''}.py first")
    sys.exit(1)
probes = load_probes(_cache)

def probe_score(resid_np, L):
    return float(probes[L].predict_proba(resid_np[L:L+1])[0][1])

def find_I_before(ids, country_pos):
    """Return position of the 'I' token that starts 'I live in ...' before country_pos."""
    for look_back in range(1, 8):
        pos = country_pos - look_back
        if pos < 0:
            break
        tok_str = model.tokenizer.decode([ids[pos]])
        # Match standalone 'I' (possibly with a leading quote)
        if tok_str.strip().lstrip('"').lstrip("'") == 'I':
            return pos
    return None


def find_all(ids, word):
    for prefix_str in [f" {word}", word]:
        toks = model.tokenizer.encode(prefix_str, add_special_tokens=False)
        tid  = toks[0]
        hits = [i for i, t in enumerate(ids) if t == tid]
        if hits:
            if len(toks) > 1:
                print(f"  NOTE: '{word}' is multi-token, tracking first token only")
            return hits
    return []

def process_prompt(prompt, utt_strings, variant_name):
    print(f"\n--- {variant_name} ---")
    ids = model.tokenizer.encode(prompt)

    word_positions = {}
    for w in WORDS:
        word_positions[w] = find_all(ids, w)
        print(f"  {w:10s}: {word_positions[w]}")

    pos_to_utt = {}
    cursor = 0
    for ui, s in enumerate(utt_strings):
        toks_u = model.tokenizer.encode(s, add_special_tokens=False)
        for j in range(len(toks_u)):
            pos_to_utt[cursor + j] = ui
        cursor += len(toks_u)

    subscript = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
    occurrence_count = {w: 0 for w in WORDS}
    seq_len = len(ids)

    columns = []
    for pos in range(seq_len):
        for word, gt in WORDS.items():
            if pos in word_positions[word]:
                occurrence_count[word] += 1
                occ = occurrence_count[word]
                n   = str(occ).translate(subscript)
                lbl = f"{word}{n}" if len(word_positions[word]) > 1 else word
                utt_idx = pos_to_utt.get(pos, 0)
                columns.append(dict(
                    pos=pos, word=word, gt=gt,
                    label=lbl, utt_idx=utt_idx,
                    spk=UTT_SPEAKER[utt_idx],
                ))

    # Add "I" tokens before Thailand and France, colored same as the country
    for country in ["Thailand", "France"]:
        gt = WORDS[country]
        for cpos in word_positions.get(country, []):
            i_pos = find_I_before(ids, cpos)
            if i_pos is not None and not any(c["pos"] == i_pos for c in columns):
                utt_idx = pos_to_utt.get(i_pos, 0)
                columns.append(dict(
                    pos=i_pos, word=f"I({country[:3]})", gt=gt,
                    label=f"I\n{country[:3]}", utt_idx=utt_idx,
                    spk=UTT_SPEAKER[utt_idx],
                ))

    columns.sort(key=lambda c: c["pos"])
    print(f"Columns ({len(columns)}):")
    for c in columns:
        print(f"  {c['label']:12s}  pos={c['pos']:3d}  utt={c['utt_idx']}  spk={c['spk']}")

    print("Running model...")
    tok = torch.tensor([ids], device="cuda")
    with torch.no_grad():
        _, cache = model.run_with_cache(
            tok, names_filter=lambda n: "hook_resid_post" in n
        )

    all_resid = np.stack([
        np.stack([
            cache[f"blocks.{L}.hook_resid_post"][0, pos].cpu().float().numpy()
            for L in range(NL)
        ])
        for pos in range(seq_len)
    ])

    N_COL = len(columns)
    mat   = np.zeros((NL, N_COL))
    for ci, col in enumerate(columns):
        resid      = all_resid[col["pos"]]
        mat[:, ci] = np.array([probe_score(resid, L) for L in range(NL)])

    return columns, mat

# ── Run both prompts ───────────────────────────────────────────────────────────
columns_n, mat_n = process_prompt(PROMPT,         UTT_STRINGS["normal"],   'Hi')
columns_c, mat_c = process_prompt(PROMPT_CONFUSED, UTT_STRINGS["confused"], 'Hi-reset')

# ── Shared-transcript helpers ──────────────────────────────────────────────────

def struck(text):
    """Return text with Unicode combining strikethrough on every character."""
    return "".join(c + "\u0336" for c in text)

def render_segments_line(fig, ax, segments, y):
    """Place a list of (text, color, fontweight) segments centered at data-y=y.
    Handles measurement → centering → placement in two passes.
    """
    segs = [(t, c, fw) for t, c, fw in segments if t]

    # Pass 0: measure pixel widths (append space for inter-word gap)
    temp = [
        ax.text(0, y, t + " ", ha="left", va="center",
                fontsize=FONTSIZE, color=c, fontweight=fw,
                zorder=4, clip_on=True)
        for t, c, fw in segs
    ]
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    widths   = [t.get_window_extent(renderer).width for t in temp]
    for t in temp:
        t.remove()

    ax_bb    = ax.get_window_extent(renderer)
    px2data  = 1.0 / ax_bb.width
    x        = 0.5 - sum(w * px2data for w in widths) / 2.0

    # Pass 1: place for real (no trailing space)
    for (t, c, fw), w in zip(segs, widths):
        ax.text(x, y, t, ha="left", va="center",
                fontsize=FONTSIZE, color=c, fontweight=fw,
                zorder=4, clip_on=True)
        x += w * px2data

def build_utt_spans(columns):
    spans = {}
    for ri, col in enumerate(columns):
        ui = col["utt_idx"]
        if ui not in spans:
            spans[ui] = [ri, ri]
        else:
            spans[ui][1] = ri
    return spans

def draw_shared_label(fig, ax, columns):
    """Draw the single shared diff-transcript column."""
    N_ROW = len(columns)
    spans = build_utt_spans(columns)

    ax.set_xlim(0, 1)
    ax.set_ylim(N_ROW - 0.5, -0.5)
    ax.axis("off")

    # Background bands
    for ui, (ri_start, ri_end) in sorted(spans.items()):
        spk = UTT_SPEAKER[ui]
        ax.add_patch(mpatches.Rectangle(
            (0, ri_start - 0.48), 1.0, ri_end - ri_start + 0.96,
            facecolor=UTT_BG[spk], edgecolor="none", zorder=1,
        ))

    # Utterance separators
    prev_utt = columns[0]["utt_idx"]
    for ri, col in enumerate(columns[1:], 1):
        if col["utt_idx"] != prev_utt:
            ax.axhline(ri - 0.5, color="white", lw=2.0, zorder=5)
            prev_utt = col["utt_idx"]

    # Per-utterance text segments
    # utt index → list-of-lines, each line = list of (text, color, fontweight)
    utt_content = {
        # utt1: "I am Alice."
        1: [[
            ('"I am ', BLACK, "normal"),
            ("Alice", RED, "bold"),
            ('."',    BLACK, "normal"),
        ]],
        # utt2: "I am Bob."
        2: [[
            ('"I am ', BLACK, "normal"),
            ("Bob",   BLUE,  "bold"),
            ('."',    BLACK, "normal"),
        ]],
        # utt3: diff greeting — "Hi Bob!" → "Hi Alice!"
        3: [[
            ('"Hi ',          BLACK, "normal"),
            ("Bob!",          BLUE,  "bold"),
            ('" \u2192 "Hi ', BLACK, "normal"),
            ("Alice!",        RED,   "bold"),
            ('"',             BLACK, "normal"),
        ]],
        # utt4: 2 rows — row1=I token, row2=country token
        4: [
            [('"I live in',              BLACK, "normal")],
            [(struck("Thailand"),        BLUE,  "bold"),
             (" ",                       BLACK, "normal"),
             ("Thailand",                RED,   "bold")],
        ],
        # utt5: 2 rows — row1=I token, row2=country token
        5: [
            [('"I live in',              BLACK, "normal")],
            [(struck("France"),          RED,   "bold"),
             (" ",                       BLACK, "normal"),
             ("France",                  BLUE,  "bold"),
             ('."',                      BLACK, "normal")],
        ],
        # utt6: query (2 rows, 2 lines — normal distribution)
        6: [
            [
                ("Question: Where does ", BLACK, "normal"),
                ("Alice",                 RED,   "bold"),
                (" live?",                BLACK, "normal"),
            ],
            [
                ("Answer: ",  BLACK, "normal"),
                ("Alice",     RED,   "bold"),
                (" lives in", BLACK, "normal"),
            ],
        ],
    }

    drawn = set()
    for ui, (ri_start, ri_end) in sorted(spans.items()):
        if ui in drawn or ui not in utt_content:
            drawn.add(ui)
            continue
        drawn.add(ui)

        lines   = utt_content[ui]
        n_lines = len(lines)
        n_rows  = ri_end - ri_start + 1

        if n_lines <= n_rows:
            line_ys    = [float(ri_start + li) for li in range(n_lines)]
            font_scale = 1.0
        else:
            # Clamp: distribute around strip midpoint, shrink font
            center_y   = (ri_start + ri_end) / 2.0
            spacing    = n_rows / n_lines
            start_y    = center_y - (n_lines - 1) / 2.0 * spacing
            line_ys    = [start_y + li * spacing for li in range(n_lines)]
            font_scale = n_rows / n_lines

        global FONTSIZE
        orig_fs  = FONTSIZE
        FONTSIZE = orig_fs * font_scale

        for segs, ly in zip(lines, line_ys):
            render_segments_line(fig, ax, segs, ly)

        FONTSIZE = orig_fs

def draw_heatmap(fig, ax, columns, mat, title):
    """Draw heatmap panel with row highlights for Thailand/France."""
    N_ROW = len(columns)
    mat_h = mat.T   # [N_ROW, NL]
    spans = build_utt_spans(columns)

    ax.set_title(title, fontsize=11, fontweight="bold", pad=5)

    ax.imshow(mat_h, aspect="auto", cmap=CMAP, vmin=0, vmax=1,
              origin="upper", interpolation="nearest")
    ax.set_xlim(-0.5, NL - 0.5)
    ax.set_ylim(N_ROW - 0.5, -0.5)

    ax.set_xticks(range(0, NL, 2))
    ax.set_xticklabels([str(l) for l in range(0, NL, 2)], fontsize=9)
    ax.set_xlabel("Layer", fontsize=11)
    ax.set_yticks(range(N_ROW))
    ax.set_yticklabels([""] * N_ROW)
    ax.yaxis.set_tick_params(length=0)

    # Utterance separators
    prev_utt = columns[0]["utt_idx"]
    for ri, col in enumerate(columns[1:], 1):
        if col["utt_idx"] != prev_utt:
            ax.axhline(ri - 0.5, color=SEP_COLOR, lw=2.0, zorder=5)
            prev_utt = col["utt_idx"]

    # Light vertical grid every 4 layers
    for l in range(0, NL, 4):
        ax.axvline(l - 0.5, color="white", lw=0.4, alpha=0.4, zorder=3)

    # Row highlights for all country-related rows (I and country tokens)
    for ri, col in enumerate(columns):
        if col["word"] in HIGHLIGHT_WORDS or col["word"].startswith("I("):
            ax.add_patch(mpatches.FancyBboxPatch(
                (-0.5, ri - 0.48), NL, 0.96,
                boxstyle="square,pad=0",
                linewidth=2.2, edgecolor=HIGHLIGHT_COLOR,
                facecolor="none", zorder=10,
            ))

# ── Figure layout ──────────────────────────────────────────────────────────────
# heat_n | shared-label | heat_c | cbar
N_ROW     = len(columns_n)
row_h     = 0.38
label_w   = 2.8
heatmap_w = max(4, NL * 0.18)
cbar_w    = 0.35
leg_h_in  = 0.55

fig_w = 2 * heatmap_w + label_w + cbar_w + 1.0
fig_h = N_ROW * row_h + leg_h_in + 0.9

fig = plt.figure(figsize=(fig_w, fig_h))

gs = fig.add_gridspec(
    2, 4,
    height_ratios=[leg_h_in, N_ROW * row_h],
    width_ratios=[heatmap_w, label_w, heatmap_w, cbar_w + 0.3],
    hspace=0.04, wspace=0.02,
    left=0.01, right=0.99,
    top=0.97, bottom=0.05,
)

ax_leg    = fig.add_subplot(gs[0, :])
ax_heat_n = fig.add_subplot(gs[1, 0])
ax_label  = fig.add_subplot(gs[1, 1])
ax_heat_c = fig.add_subplot(gs[1, 2])
ax_cbar   = fig.add_subplot(gs[1, 3])

# ── Legend ─────────────────────────────────────────────────────────────────────
ax_leg.axis("off")
legend_elements = [
    mpatches.Patch(facecolor=GT_COLORS["tri"], label="Speaker 1 (▲) binding"),
    mpatches.Patch(facecolor=GT_COLORS["sq"],  label="Speaker 2 (■) binding"),
    mpatches.Patch(facecolor=UTT_BG[1],        label="Speaker 1 utterance"),
    mpatches.Patch(facecolor=UTT_BG[2],        label="Speaker 2 utterance"),
    mpatches.Patch(facecolor=UTT_BG[0],        label="Narrator / query"),
    # mpatches.Patch(facecolor="none", edgecolor=HIGHLIGHT_COLOR,
    #                linewidth=2, label="Highlighted (country tokens)"),
]
ax_leg.legend(
    handles=legend_elements,
    loc="center", ncol=6,
    fontsize=10, frameon=False,
    handlelength=1.2, handleheight=1.0, columnspacing=1.0,
)

# ── Draw ───────────────────────────────────────────────────────────────────────
draw_heatmap(fig, ax_heat_n, columns_n, mat_n, 'Normal: "Hi Bob!" Model Answer: France')
draw_heatmap(fig, ax_heat_c, columns_c, mat_c, 'Confused: "Hi Alice!" Model Answer: Thailand')
draw_shared_label(fig, ax_label, columns_n)   # row structure same for both

# ── Shared colorbar ────────────────────────────────────────────────────────────
ax_cbar.set_visible(False)
sm = plt.cm.ScalarMappable(cmap=CMAP, norm=Normalize(0, 1))
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax_cbar, fraction=0.9, pad=0.05)
cbar.set_label("Binding ID probe score  P(▲)", fontsize=11, labelpad=8)
cbar.ax.tick_params(labelsize=9)

# ── Save ───────────────────────────────────────────────────────────────────────
_suffix = "_hi_only" if args.hi_only else ""
out_path = os.path.join(OUTDIR, f"probe_score_heatmap_v2_reset_with_I_{args.model}{_suffix}.pdf")
plt.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"\nSaved {out_path}")
print("Done.")
