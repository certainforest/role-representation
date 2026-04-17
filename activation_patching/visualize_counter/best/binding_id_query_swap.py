"""
binding_id_query_swap.py — SWAP steering on the query entity token only.

Same structure as binding_id_swap.py but instead of patching the entity
tokens in the introduction lines, we patch the queried entity token in:
  "Question: Where does {e0} live? Answer: {e0} lives in"

The last occurrence of e0 is the one right before the model generates the answer.

If the binding-ID at the query token is what the model uses to look up which
country to output, flipping it here should flip the answer.

SWAP direction:
  - e0=△ (base, hi-confused): subtract d from query pos → push △ toward □ → answer flips
  - e0=□ (hi-e1):             add d to query pos    → push □ toward △ → answer flips

Usage:
  python binding_id_query_swap.py --model llama --gpu 1 --n-probe 100 --n-test 100
"""
import os, sys, argparse
import numpy as np
import torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import Counter
from probe_io import cache_path, load_probes

parser = argparse.ArgumentParser()
parser.add_argument("--model",   choices=["llama","qwen"], default="llama")
parser.add_argument("--gpu",     default="1")
parser.add_argument("--n-probe", type=int, default=100)
parser.add_argument("--basehi",  action="store_true", help="use base+hi probe")
parser.add_argument("--n-test",  type=int, default=100)
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
print(f"Loaded: {NL} layers")

# ── Vocab ──────────────────────────────────────────────────────────────────────
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
print(f"Single-token: {len(NAMES)} names, {len(COUNTRIES)} countries")

rng = np.random.default_rng(42)
ALL_PAIRS = [(i,j,p,q)
             for i,j in [(a,b) for a in range(len(NAMES)) for b in range(len(NAMES)) if a!=b]
             for p,q in [(a,b) for a in range(len(COUNTRIES)) for b in range(len(COUNTRIES)) if a!=b]]
IDX = rng.permutation(len(ALL_PAIRS))

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

# ── Helpers ────────────────────────────────────────────────────────────────────
def question_of(ids, word):
    """Find the occurrence of word in 'Question: Where does {word} live?'
    = second-to-last occurrence (last = Answer prefix, first = introduction)."""
    tid = model.tokenizer.encode(f" {word}", add_special_tokens=False)[0]
    pos = [i for i, t in enumerate(ids) if t == tid]
    return pos[-2] if len(pos) >= 2 else None

def country_tok_id(word):
    return model.tokenizer.encode(f" {word}", add_special_tokens=False)[0]

@torch.no_grad()
def run_baseline(prompt):
    tok = torch.tensor([model.tokenizer.encode(prompt)], device="cuda")
    return model(tok)[0, -1].cpu().float().numpy()

@torch.no_grad()
def run_query_swap(prompt, layer, alpha, pos_query, sign):
    """Flip binding-ID at query position only.
    sign=+1: add alpha*d (push □ toward △)
    sign=-1: subtract alpha*d (push △ toward □)
    """
    tok   = torch.tensor([model.tokenizer.encode(prompt)], device="cuda")
    delta = torch.tensor(sign * alpha * directions[layer], dtype=torch.bfloat16, device="cuda")

    def hook_fn(resid, hook):
        resid[0, pos_query] = resid[0, pos_query] + delta
        return resid

    logits = model.run_with_hooks(
        tok, fwd_hooks=[(f"blocks.{layer}.hook_resid_post", hook_fn)])
    return logits[0, -1].cpu().float().numpy()

def parse_logits(logits_np, a0, a1, correct_key):
    probs  = np.exp(logits_np - logits_np.max())
    probs /= probs.sum()
    top_id = int(np.argmax(logits_np))
    top_str = model.tokenizer.decode([top_id]).strip()
    p_a0 = float(probs[country_tok_id(a0)])
    p_a1 = float(probs[country_tok_id(a1)])
    correct_id = country_tok_id(a0 if correct_key == "a0" else a1)
    wrong_id   = country_tok_id(a1 if correct_key == "a0" else a0)
    if top_id == correct_id:   outcome = "correct"
    elif top_id == wrong_id:   outcome = "wrong"
    else:                      outcome = "other"
    return top_str, p_a0, p_a1, outcome

# ── Load probes ────────────────────────────────────────────────────────────────
_suffix = "_basehi" if args.basehi else ""
_cache = (os.path.join(OUTDIR, f"probes_{args.model}_{args.n_probe}_intercept_basehi.joblib")
          if args.basehi else cache_path(OUTDIR, args.model, args.n_probe))
if os.path.exists(_cache):
    print(f"Loading cached probes (n={args.n_probe})...")
    probes = load_probes(_cache)
else:
    print("No cached probes — run binding_id_swap.py first")
    sys.exit(1)

directions = {L: probes[L].coef_[0] / np.linalg.norm(probes[L].coef_[0])
              for L in range(NL)}

# ── Experiment ─────────────────────────────────────────────────────────────────
STEER_LAYERS = list(range(0, 15, 1))   # every other layer for the heatmap
ALPHAS       = [0,50,100,150,200]

# sign convention:
#   e0=△ (base, hi-confused): subtract → sign=-1
#   e0=□ (hi-e1):             add      → sign=+1
CONDS = [
    # (label, prompt_fn, correct_key, query_sign)
    ("base",     lambda e0,e1,a0,a1: make_base(e0,e1,a0,a1),  "a0", -1),
    ("hi",       lambda e0,e1,a0,a1: make_hi(e0,e1,a0,a1,e1), "a1", +1),
    ("hi-reset", lambda e0,e1,a0,a1: make_hi(e0,e1,a0,a1,e0), "a0", -1),
]

# results[cond][L][alpha] = list of (top_str, p_a0, p_a1, outcome)
results = {
    cname: {L: {a: [] for a in ALPHAS} for L in STEER_LAYERS}
    for cname, *_ in CONDS
}

print(f"\nRunning query-token SWAP (n={args.n_test})...")
for k, si in enumerate(IDX[args.n_probe : args.n_probe + args.n_test]):
    ni, nj, ci, cj = ALL_PAIRS[si]
    e0, e1 = NAMES[ni], NAMES[nj]
    a0, a1 = COUNTRIES[ci], COUNTRIES[cj]

    for cname, pfn, correct_key, sign in CONDS:
        prompt = pfn(e0, e1, a0, a1)
        ids    = model.tokenizer.encode(prompt)

        pos_query = question_of(ids, e0)   # e0 in "Question: Where does {e0} live?"
        if pos_query is None:
            continue

        try:
            baseline = run_baseline(prompt)
        except Exception as e:
            print(f"  SKIP {k} baseline: {type(e).__name__}: {e}")
            continue

        for L in STEER_LAYERS:
            for alpha in ALPHAS:
                try:
                    if alpha == 0:
                        logits = baseline
                    else:
                        logits = run_query_swap(prompt, L, alpha, pos_query, sign)
                    results[cname][L][alpha].append(
                        parse_logits(logits, a0, a1, correct_key))
                except Exception as e:
                    print(f"  SKIP {k} L={L} α={alpha}: {type(e).__name__}: {e}")
                    break  # skip remaining alphas for this layer on error

    if (k+1) % 20 == 0:
        print(f"  {k+1}/{args.n_test}")

# ── Summary table ──────────────────────────────────────────────────────────────
print(f"\nFraction correct (P(correct) > P(wrong)) at each layer/alpha:")
for cname, *_ in CONDS:
    print(f"\n=== {cname} ===")
    print(f"{'Layer':<7} " + "  ".join(f"α={a:<5}" for a in ALPHAS))
    for L in STEER_LAYERS:
        row = ""
        for alpha in ALPHAS:
            entries = results[cname][L][alpha]
            if not entries:
                row += f"  {'?':>7}"
                continue
            correct_key = [c for c, *_ in CONDS if c == cname][0]
            frac = np.mean([e[3] == "correct" for e in entries])
            row += f"  {frac:>7.2f}"
        print(f"L={L:<5} {row}")

# ── Heatmap: frac correct per (layer, alpha) ──────────────────────────────────
CELL_H       = 0.25   # inches per layer row
FIG_W        = 14
FIG_H        = max(3, len(STEER_LAYERS) * CELL_H)
FONT_SIZE    = 10
cmap         = plt.cm.RdBu_r

fig, axes = plt.subplots(1, len(CONDS), figsize=(FIG_W, FIG_H), sharey=True)

for ax, (cname, *_) in zip(axes, CONDS):
    mat = np.full((len(STEER_LAYERS), len(ALPHAS)), np.nan)
    for ri, L in enumerate(STEER_LAYERS):
        for ci, alpha in enumerate(ALPHAS):
            entries = results[cname][L][alpha]
            if entries:
                mat[ri, ci] = np.mean([e[3] == "correct" for e in entries])

    im = ax.imshow(mat, aspect="auto", cmap=cmap, vmin=0, vmax=1, origin="upper")
    ax.set_xticks(range(len(ALPHAS)))
    ax.set_xticklabels([str(a) for a in ALPHAS], fontsize=FONT_SIZE)
    ax.set_yticks(range(len(STEER_LAYERS)))
    ax.set_yticklabels([f"L{L}" for L in STEER_LAYERS], fontsize=FONT_SIZE)
    ax.set_xlabel("α", fontsize=FONT_SIZE)
    ax.set_title(cname, fontsize=FONT_SIZE)
    for ri in range(len(STEER_LAYERS)):
        for ci in range(len(ALPHAS)):
            v = mat[ri, ci]
            if not np.isnan(v):
                ax.text(ci, ri, f"{v:.2f}",
                        ha="center", va="center", fontsize=6.5, color="black")
    plt.colorbar(im, ax=ax, shrink=0.6, label="frac. correct")

axes[0].set_ylabel("Patch layer", fontsize=FONT_SIZE)
plt.tight_layout()
out = os.path.join(OUTDIR, f"binding_id_query_swap_{args.model}{_suffix}.pdf")
plt.savefig(out, dpi=300, bbox_inches="tight")
plt.close()
print(f"\nSaved {out}")
print("Done.")
