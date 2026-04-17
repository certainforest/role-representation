"""
binding_id_swap_output_exact.py — Single-example SWAP visualization

Fixes e0=Alice, e1=Bob, a0=Thailand, a1=France.
For each condition (base, hi-e1, hi-confused) × layer × alpha:
  shows the exact token the model outputs, colored by outcome.

The full prompt for each condition is printed alongside the grid.

Usage:
  python binding_id_swap_output_exact.py --model llama --gpu 3
"""
import os, sys, argparse
import numpy as np
import torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.linear_model import LogisticRegression
from probe_io import cache_path, load_probes, save_probes

parser = argparse.ArgumentParser()
parser.add_argument("--model",     choices=["llama","qwen"], default="llama")
parser.add_argument("--gpu",       default="3")
parser.add_argument("--n-probe",   type=int, default=100)
parser.add_argument("--basehi",    action="store_true", help="use base+hi probe")
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../attribution_ioi"))
from utils import MODEL_CONFIGS
from transformer_lens import HookedTransformer

OUTDIR = os.path.dirname(os.path.abspath(__file__))
cfg    = MODEL_CONFIGS[args.model]

print(f"Loading {cfg['name']}...")
model = HookedTransformer.from_pretrained(cfg["name"], dtype=torch.bfloat16, cache_dir=cfg["cache"])
model.eval().to("cuda")
NL = model.cfg.n_layers

# ── Fixed example ──────────────────────────────────────────────────────────────
E0, E1, A0, A1 = "Alice", "Bob", "Canada", "Mexico"

# ── Prompts ────────────────────────────────────────────────────────────────────
def make_base(e0, e1, a0, a1):
    return (
        'This is the transcript of a conversation.\n'
        f'"I am {e0}."\n"I am {e1}."\n'
        f'"I live in {a0}."\n"I live in {a1}."\n'
        f'Question: Where does {e0} live? Answer: {e0} lives in'
    )

def make_hi(e0, e1, a0, a1, greeted):
    return (
        'This is the transcript of a conversation.\n'
        f'"I am {e0}."\n"I am {e1}."\n'
        f'"Hi {greeted}!"\n'
        f'"I live in {a0}. What about you?"\n'
        f'"I live in {a1}."\n'
        f'Question: Where does {e0} live? Answer: {e0} lives in'
    )

CONDS = [
    # (label, prompt, correct_country, wrong_country)
    ("base",
     make_base(E0, E1, A0, A1),
     A0, A1),
    ("hi-e1\n(Hi Bob! → Bob speaks A0, Alice speaks A1)",
     make_hi(E0, E1, A0, A1, E1),
     A1, A0),
    ("hi-confused\n(Hi Alice! → Alice speaks A0, Bob speaks A1)",
     make_hi(E0, E1, A0, A1, E0),
     A0, A1),
]

# ── Helpers ────────────────────────────────────────────────────────────────────
def tok_id(word):
    return model.tokenizer.encode(f" {word}", add_special_tokens=False)[0]

def first_of(ids, word):
    tid = tok_id(word)
    return next((i for i,t in enumerate(ids) if t == tid), None)

@torch.no_grad()
def run_baseline(prompt):
    ids = torch.tensor([model.tokenizer.encode(prompt)], device="cuda")
    return model(ids)[0, -1].cpu().float().numpy()

@torch.no_grad()
def run_swap(prompt, layer, pos_tri, pos_sq, delta_np):
    ids = torch.tensor([model.tokenizer.encode(prompt)], device="cuda")
    delta_t = torch.tensor(delta_np, dtype=torch.bfloat16, device="cuda")
    def hook_fn(value, hook):
        value[0, pos_tri] -= delta_t
        value[0, pos_sq]  += delta_t
        return value
    logits = model.run_with_hooks(ids, fwd_hooks=[(f"blocks.{layer}.hook_resid_post", hook_fn)])
    return logits[0, -1].cpu().float().numpy()

def decode_top(logits_np):
    top_id = int(np.argmax(logits_np))
    return model.tokenizer.decode([top_id]).strip()

def get_probs(logits_np, correct, wrong):
    probs = np.exp(logits_np - logits_np.max())
    probs /= probs.sum()
    return float(probs[tok_id(correct)]), float(probs[tok_id(wrong)])

# ── Load probes ────────────────────────────────────────────────────────────────
_suffix = "_basehi" if args.basehi else ""
_cache = (os.path.join(OUTDIR, f"probes_{args.model}_{args.n_probe}_intercept_basehi.joblib")
          if args.basehi else cache_path(OUTDIR, args.model, args.n_probe))
if os.path.exists(_cache):
    print(f"Loading cached probes...")
    probes = load_probes(_cache)
else:
    # Need vocab + training data — train from scratch
    _NAMES = ["Alice","Bob","Charlie","Diana","Emma","James","Olivia","Liam",
              "Sophia","Noah","Ava","Ethan","Isabella","Mason","Mia","William",
              "Charlotte","Benjamin","Amelia","Lucas","Harper","Henry","Evelyn",
              "Alexander","Oliver","Aria","Sebastian","Luna","Jack","Chloe",
              "Owen","Penelope","Aiden","Layla","Nathan","Grace"]
    _COUNTRIES = ["France","Thailand","Germany","Spain","Italy","Brazil","Japan",
                  "Canada","Mexico","Australia","Sweden","Norway","Egypt","India",
                  "China","Argentina","Portugal","Greece","Poland","Turkey",
                  "Denmark","Finland","Austria","Belgium","Netherlands","Chile",
                  "Vietnam","Morocco","Ukraine","Colombia","Romania","Pakistan",
                  "Kenya","Peru","Hungary","Croatia"]
    def single_token(w):
        return len(model.tokenizer.encode(f" {w}", add_special_tokens=False)) == 1
    NAMES     = [n for n in _NAMES     if single_token(n)]
    COUNTRIES = [c for c in _COUNTRIES if single_token(c)]
    rng = np.random.default_rng(42)
    ALL_PAIRS = [(i,j,p,q)
                 for i,j in [(a,b) for a in range(len(NAMES)) for b in range(len(NAMES)) if a!=b]
                 for p,q in [(a,b) for a in range(len(COUNTRIES)) for b in range(len(COUNTRIES)) if a!=b]]
    IDX = rng.permutation(len(ALL_PAIRS))

    def extract_pos(cache, pos):
        return np.stack([cache[f"blocks.{L}.hook_resid_post"][0, pos].cpu().float().numpy()
                         for L in range(NL)])

    print(f"Training probes (n={args.n_probe})...")
    X_probe, y_probe = [], []
    for k, si in enumerate(IDX[:args.n_probe]):
        ni, nj, ci, cj = ALL_PAIRS[si]
        e0, e1 = NAMES[ni], NAMES[nj]
        a0, a1 = COUNTRIES[ci], COUNTRIES[cj]
        try:
            for greeted, lbl_a0 in [(e0,1),(e1,0)]:
                for intro in [(e0,e1),(e1,e0)]:
                    p = make_hi(intro[0], intro[1], a0, a1, greeted)
                    ids = torch.tensor([model.tokenizer.encode(p)], device="cuda")
                    _, c = model.run_with_cache(ids, names_filter=lambda n: "hook_resid_post" in n)
                    t = model.tokenizer.encode(p)
                    tri = intro[0] if greeted == intro[0] else intro[1]
                    sq  = intro[1] if greeted == intro[0] else intro[0]
                    pt = first_of(t, tri); ps = first_of(t, sq)
                    if pt: X_probe.append(extract_pos(c, pt)); y_probe.append(1)
                    if ps: X_probe.append(extract_pos(c, ps)); y_probe.append(0)
                    p0 = first_of(t, a0); p1 = first_of(t, a1)
                    if p0: X_probe.append(extract_pos(c, p0)); y_probe.append(lbl_a0)
                    if p1: X_probe.append(extract_pos(c, p1)); y_probe.append(1-lbl_a0)
        except Exception:
            continue
    X_probe = np.stack(X_probe); y_probe = np.array(y_probe)
    probes = []
    for L in range(NL):
        lr = LogisticRegression(max_iter=2000, C=1.0, solver="lbfgs", fit_intercept=True)
        lr.fit(X_probe[:, L, :], y_probe)
        probes.append(lr)
    save_probes(probes, _cache)

directions = {L: probes[L].coef_[0] / np.linalg.norm(probes[L].coef_[0]) for L in range(NL)}

# ── Run SWAP on fixed example ──────────────────────────────────────────────────
STEER_LAYERS = [5, 8, 10, 12, 13, 15, 20]
ALPHAS       = [0, 5, 10, 15, 20, 30]

# grid[cond][L][alpha] = (top_token_str, p_correct, p_wrong, outcome)
grid = {}
for label, prompt, correct, wrong in CONDS:
    grid[label] = {}
    ids = model.tokenizer.encode(prompt)

    # △ = correct country token position, □ = wrong country token position
    pos_tri = first_of(ids, correct)   # △-tagged country (correct)
    pos_sq  = first_of(ids, wrong)     # □-tagged country (wrong)

    print(f"\n{label.split(chr(10))[0]}: pos_correct={pos_tri}({correct}), pos_wrong={pos_sq}({wrong})")

    for L in STEER_LAYERS:
        grid[label][L] = {}
        dir_L = directions[L]
        for alpha in ALPHAS:
            if alpha == 0:
                logits = run_baseline(prompt)
            else:
                logits = run_swap(prompt, L, pos_tri, pos_sq, alpha * dir_L)
            top = decode_top(logits)
            pc, pw = get_probs(logits, correct, wrong)
            if top == correct:
                outcome = "correct"
            elif top == wrong:
                outcome = "wrong"
            else:
                outcome = "other"
            grid[label][L][alpha] = (top, pc, pw, outcome)
            print(f"  L={L} α={alpha:3d}: top='{top}'  P({correct})={pc:.3f}  P({wrong})={pw:.3f}  [{outcome}]")

# ── Figure ─────────────────────────────────────────────────────────────────────
# 3 conditions stacked vertically.
# Each condition: a grid (rows=alphas, cols=layers) of colored cells with token text.
# Below each grid: the full prompt.

OUTCOME_COLOR = {"correct": "#a8d5a2", "wrong": "#f4a9a8", "other": "#d3d3d3"}

n_rows = len(ALPHAS)
n_cols = len(STEER_LAYERS)

# Each condition panel: grid (cols=layers, rows=alphas) + prompt text row below
# Use GridSpec with explicit height ratios so prompt gets its own space
fig, axes = plt.subplots(len(CONDS), 1, figsize=(16, 4.0 * len(CONDS)))

for ax, (label, prompt, correct, wrong) in zip(axes, CONDS):
    ax.set_xlim(-0.6, n_cols)
    ax.set_ylim(-0.1, n_rows + 0.55)
    ax.set_aspect("equal")
    ax.axis("off")

    # Column headers
    for ci, L in enumerate(STEER_LAYERS):
        ax.text(ci + 0.5, n_rows + 0.35, f"L{L}",
                ha="center", va="bottom", fontsize=8)

    # Row labels
    for ri, alpha in enumerate(ALPHAS):
        ax.text(-0.12, ri + 0.5, f"α={alpha}",
                ha="right", va="center", fontsize=8)

    # Cells
    for ci, L in enumerate(STEER_LAYERS):
        for ri, alpha in enumerate(ALPHAS):
            top, pc, pw, outcome = grid[label][L][alpha]
            color = OUTCOME_COLOR[outcome]
            rect = mpatches.FancyBboxPatch(
                (ci + 0.04, ri + 0.04), 0.92, 0.92,
                boxstyle="round,pad=0.02", linewidth=0.5,
                edgecolor="white", facecolor=color)
            ax.add_patch(rect)
            ax.text(ci + 0.5, ri + 0.63, f'"{top}"',
                    ha="center", va="center", fontsize=5,
                    fontfamily="monospace")
            ax.text(ci + 0.5, ri + 0.27, f"P(✓)={pc:.2f}\nP(✗)={pw:.2f}",
                    ha="center", va="center", fontsize=5, color="#444444")

    cond_short = label.split("\n")[0]
    ax.set_title(cond_short, fontsize=10, pad=6)

# Legend
legend_patches = [
    mpatches.Patch(color=OUTCOME_COLOR["correct"], label=f'correct ({A0}/{A1})'),
    mpatches.Patch(color=OUTCOME_COLOR["wrong"],   label=f'wrong ({A1}/{A0})'),
    mpatches.Patch(color=OUTCOME_COLOR["other"],   label='other token'),
]
fig.legend(handles=legend_patches, loc="upper center", ncol=3,
           fontsize=9, bbox_to_anchor=(0.5, 1.005))

fig.suptitle(
    f"Exact model output under SWAP  ({cfg['name']})\n"
    f"e0={E0}, e1={E1}, a0={A0}(△ in base/hi-confused), a1={A1}  —  "
    "SWAP: push △-country→□ and □-country→△ at one layer.",
    fontsize=9, y=1.03)
out = os.path.join(OUTDIR, f"binding_id_swap_output_exact_{args.model}{_suffix}_extreme.png")
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.close()
print(f"\nSaved {out}")
print("Done.")
