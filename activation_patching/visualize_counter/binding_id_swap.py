"""
binding_id_swap.py — SWAP steering: track actual output + P(country1) + P(country2)

SWAP = push △-country toward □ AND □-country toward △ simultaneously.
Expected result: model flips its answer from the correct country to the other one.

Tracks per example × layer × alpha:
  1. model's actual top-1 output token
  2. P(a0) — probability of first country token
  3. P(a1) — probability of second country token

Two conditions:
  hi-e1:       "Hi e1!" → e1(□) says a0, e0(△) says a1. Correct for e0 = a1.
               SWAP: push a1(△)→□ and a0(□)→△. Expect model flips to a0.

  hi-confused: "Hi e0!" → e0(△) says a0, e1(□) says a1. Correct for e0 = a0.
               SWAP: push a0(△)→□ and a1(□)→△. Expect model flips to a1.

Usage:
  python binding_id_swap.py --model llama --gpu 3 --n-probe 100 --n-test 50
"""
import os, sys, argparse
import numpy as np
import torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from probe_io import cache_path, save_probes, load_probes

parser = argparse.ArgumentParser()
parser.add_argument("--model",   choices=["llama","qwen"], default="llama")
parser.add_argument("--gpu",     default="3")
parser.add_argument("--n-probe", type=int, default=100)
parser.add_argument("--n-test",  type=int, default=50)
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
print(f"Loaded: {NL} layers, d_model={model.cfg.d_model}")

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
@torch.no_grad()
def get_ids(prompt):
    return model.tokenizer.encode(prompt, return_tensors="pt").to("cuda")[0].tolist()

def first_of(ids, word):
    tid = model.tokenizer.encode(f" {word}", add_special_tokens=False)[0]
    return next((i for i,t in enumerate(ids) if t == tid), None)

def country_tok_id(word):
    return model.tokenizer.encode(f" {word}", add_special_tokens=False)[0]

def extract_pos(cache, pos):
    return np.stack([
        cache[f"blocks.{L}.hook_resid_post"][0, pos].cpu().float().numpy()
        for L in range(NL)
    ])

@torch.no_grad()
def run_baseline(prompt):
    tok = torch.tensor([model.tokenizer.encode(prompt)], device="cuda")
    logits = model(tok)
    return logits[0, -1].cpu().float().numpy()

@torch.no_grad()
def run_swap(prompt, layer, pos_tri, pos_sq, delta_np):
    """Push △-pos toward □ (subtract) and □-pos toward △ (add) in one pass."""
    tok = torch.tensor([model.tokenizer.encode(prompt)], device="cuda")
    delta_t = torch.tensor(delta_np, dtype=torch.bfloat16, device="cuda")

    def hook_fn(value, hook):
        value[0, pos_tri] = value[0, pos_tri] - delta_t  # △ → □
        value[0, pos_sq]  = value[0, pos_sq]  + delta_t  # □ → △
        return value

    logits = model.run_with_hooks(tok, fwd_hooks=[(f"blocks.{layer}.hook_resid_post", hook_fn)])
    return logits[0, -1].cpu().float().numpy()

def parse_logits(logits_np, a0, a1, correct_key):
    """Return (top_token_str, P(a0), P(a1), outcome) where outcome is 'correct'/'wrong'/'other'."""
    probs = np.exp(logits_np - logits_np.max())
    probs /= probs.sum()
    top_id = int(np.argmax(logits_np))
    top_str = model.tokenizer.decode([top_id]).strip()
    p_a0 = float(probs[country_tok_id(a0)])
    p_a1 = float(probs[country_tok_id(a1)])
    correct_id = country_tok_id(a0 if correct_key == "a0" else a1)
    wrong_id   = country_tok_id(a1 if correct_key == "a0" else a0)
    if top_id == correct_id:
        outcome = "correct"
    elif top_id == wrong_id:
        outcome = "wrong"
    else:
        outcome = "other"
    return top_str, p_a0, p_a1, outcome

@torch.no_grad()
def get_cache(prompt):
    tok = torch.tensor([model.tokenizer.encode(prompt)], device="cuda")
    _, cache = model.run_with_cache(tok, names_filter=lambda n: "hook_resid_post" in n)
    return cache

# ── Train per-layer probes (or load from cache) ───────────────────────────────
_cache = cache_path(OUTDIR, args.model, args.n_probe)
if os.path.exists(_cache):
    print(f"\n[Phase 1] Loading cached probes (n={args.n_probe})...")
    probes = load_probes(_cache)
else:
    print(f"\n[Phase 1] Training probes (n={args.n_probe})...")
    X_probe, y_probe = [], []
    for k, si in enumerate(IDX[:args.n_probe]):
        ni, nj, ci, cj = ALL_PAIRS[si]
        e0, e1 = NAMES[ni], NAMES[nj]
        a0, a1 = COUNTRIES[ci], COUNTRIES[cj]
        try:
            cA = get_cache(make_hi(e0, e1, a0, a1, e0))
            cB = get_cache(make_hi(e0, e1, a0, a1, e1))
            cC = get_cache(make_hi(e1, e0, a0, a1, e0))
            cD = get_cache(make_hi(e1, e0, a0, a1, e1))
            tA = get_ids(make_hi(e0, e1, a0, a1, e0))
            tB = get_ids(make_hi(e0, e1, a0, a1, e1))
            tC = get_ids(make_hi(e1, e0, a0, a1, e0))
            tD = get_ids(make_hi(e1, e0, a0, a1, e1))
            for c, t, tri, sq in [(cA,tA,e0,e1),(cB,tB,e0,e1),(cC,tC,e1,e0),(cD,tD,e1,e0)]:
                p_t = first_of(t, tri); p_s = first_of(t, sq)
                if p_t: X_probe.append(extract_pos(c, p_t)); y_probe.append(1)
                if p_s: X_probe.append(extract_pos(c, p_s)); y_probe.append(0)
            for c, t, lbl in [(cA,tA,1),(cB,tB,0),(cC,tC,0),(cD,tD,1)]:
                p0 = first_of(t, a0); p1 = first_of(t, a1)
                if p0: X_probe.append(extract_pos(c, p0)); y_probe.append(lbl)
                if p1: X_probe.append(extract_pos(c, p1)); y_probe.append(1 - lbl)
        except Exception:
            continue
        if (k+1) % 30 == 0:
            print(f"  {k+1}/{args.n_probe}")

    X_probe = np.stack(X_probe)
    y_probe = np.array(y_probe, dtype=int)
    probes = []
    for L in range(NL):
        lr = LogisticRegression(max_iter=2000, C=1.0, solver="lbfgs", fit_intercept=False)
        lr.fit(X_probe[:, L, :], y_probe)
        probes.append(lr)
    save_probes(probes, _cache)

directions = {}
for L in range(NL):
    w = probes[L].coef_[0]
    directions[L] = w / np.linalg.norm(w)

# ── SWAP experiment ────────────────────────────────────────────────────────────
STEER_LAYERS = [5, 8, 10, 13, 15, 20]
ALPHAS       = [0, 5, 10, 15, 20, 30, 40, 50]

# results[cond][L][alpha] = list of (top_str, p_a0, p_a1)
results = {
    "base":        {L: {a: [] for a in ALPHAS} for L in STEER_LAYERS},
    "hi-e1":       {L: {a: [] for a in ALPHAS} for L in STEER_LAYERS},
    "hi-confused": {L: {a: [] for a in ALPHAS} for L in STEER_LAYERS},
}

print(f"\n[Phase 2] Running SWAP experiments (n={args.n_test})...")
for k, si in enumerate(IDX[args.n_probe : args.n_probe + args.n_test]):
    ni, nj, ci, cj = ALL_PAIRS[si]
    e0, e1 = NAMES[ni], NAMES[nj]
    a0, a1 = COUNTRIES[ci], COUNTRIES[cj]

    prompt_base = make_base(e0, e1, a0, a1)      # base:        correct=a0, △=a0, □=a1
    prompt_e1   = make_hi(e0, e1, a0, a1, e1)   # hi-e1:       correct=a1, △=a1, □=a0
    prompt_con  = make_hi(e0, e1, a0, a1, e0)   # hi-confused: correct=a0, △=a0, □=a1

    try:
        ids_base = get_ids(prompt_base)
        ids_e1   = get_ids(prompt_e1)
        ids_con  = get_ids(prompt_con)
        pos_a0_base = first_of(ids_base, a0)
        pos_a1_base = first_of(ids_base, a1)
        pos_a0_e1   = first_of(ids_e1,  a0)
        pos_a1_e1   = first_of(ids_e1,  a1)
        pos_a0_con  = first_of(ids_con, a0)
        pos_a1_con  = first_of(ids_con, a1)
        if None in (pos_a0_base, pos_a1_base, pos_a0_e1, pos_a1_e1, pos_a0_con, pos_a1_con):
            continue

        for L in STEER_LAYERS:
            dir_L = directions[L]
            for alpha in ALPHAS:
                # base SWAP: push a0(△)→□, push a1(□)→△
                if alpha == 0:
                    logits = run_baseline(prompt_base)
                else:
                    logits = run_swap(prompt_base, L, pos_a0_base, pos_a1_base, alpha * dir_L)
                results["base"][L][alpha].append(parse_logits(logits, a0, a1, "a0"))

                # hi-e1 SWAP: push a1(△)→□, push a0(□)→△
                if alpha == 0:
                    logits = run_baseline(prompt_e1)
                else:
                    logits = run_swap(prompt_e1, L, pos_a1_e1, pos_a0_e1, alpha * dir_L)
                results["hi-e1"][L][alpha].append(parse_logits(logits, a0, a1, "a1"))

                # hi-confused SWAP: push a0(△)→□, push a1(□)→△
                if alpha == 0:
                    logits = run_baseline(prompt_con)
                else:
                    logits = run_swap(prompt_con, L, pos_a0_con, pos_a1_con, alpha * dir_L)
                results["hi-confused"][L][alpha].append(parse_logits(logits, a0, a1, "a0"))

    except Exception:
        continue
    if (k+1) % 10 == 0:
        print(f"  {k+1}/{args.n_test}")

# ── Figure ─────────────────────────────────────────────────────────────────────
# 2 rows (hi-e1, hi-confused) × len(STEER_LAYERS) columns
# Each panel: P(a0) and P(a1) vs alpha (lines), plus bar at bottom for output category
fig, axes = plt.subplots(4, len(STEER_LAYERS), figsize=(22, 16),
                         gridspec_kw={"height_ratios": [2, 2, 2, 1.2]})

CONDS = [
    ("base",        "a0", "a1", "forestgreen", "orange",
     "base  (correct=a0=△, wrong=a1=□)\nSWAP: expect flip → a1"),
    ("hi-e1",       "a1", "a0", "steelblue",   "orange",
     "hi-e1  (correct=a1=△, wrong=a0=□)\nSWAP: expect flip → a0"),
    ("hi-confused", "a0", "a1", "crimson",     "green",
     "hi-confused  (correct=a0=△, wrong=a1=□)\nSWAP: expect flip → a1"),
]

for row, (cond, correct_key, wrong_key, c_correct, c_wrong, row_title) in enumerate(CONDS):
    for col, L in enumerate(STEER_LAYERS):
        ax = axes[row, col]

        mean_p_correct, mean_p_wrong = [], []
        frac_correct, frac_wrong, frac_other = [], [], []

        for alpha in ALPHAS:
            entries = results[cond][L][alpha]
            # entries: list of (top_str, p_a0, p_a1)
            # correct = a1 for hi-e1, a0 for hi-confused
            p_corrects = [e[2] if correct_key == "a1" else e[1] for e in entries]
            p_wrongs   = [e[1] if wrong_key   == "a0" else e[2] for e in entries]
            tops       = [e[0] for e in entries]

            mean_p_correct.append(np.mean(p_corrects))
            mean_p_wrong.append(np.mean(p_wrongs))

            # Fraction where top-1 is correct country, wrong country, or other
            # We need the actual word for the correct/wrong country
            # We stored p_a0, p_a1 but not which country word was a0/a1 per example
            # Instead use top_str comparison — stored as decoded token
            # top_str is decoded from model, country names appear as " France" etc
            # We compare stripped top string against country names
            n = len(tops)
            # Can't directly compare since we don't have per-example a0/a1 here.
            # Use P(correct) > P(wrong) as proxy for "would output correct"
            fc = np.mean([pc > pw for pc, pw in zip(p_corrects, p_wrongs)])
            fw = np.mean([pw > pc for pc, pw in zip(p_corrects, p_wrongs)])
            fo = 1.0 - fc - fw  # ties (rare)
            frac_correct.append(fc)
            frac_wrong.append(fw)
            frac_other.append(fo)

        ax.plot(ALPHAS, mean_p_correct, color=c_correct, lw=2, marker="o", ms=4,
                label=f"P({correct_key}=correct)")
        ax.plot(ALPHAS, mean_p_wrong,   color=c_wrong,   lw=2, marker="s", ms=4,
                linestyle="--", label=f"P({wrong_key}=wrong)")
        ax.axhline(0, color="gray", ls=":", lw=0.5)
        ax.set_ylim(-0.01, 1.0)
        ax.set_xlim(-5, ALPHAS[-1] + 5)
        if col == 0:
            ax.set_ylabel(f"{row_title}\nMean probability", fontsize=8)
            ax.legend(fontsize=7, loc="upper right")
        ax.set_title(f"Layer {L}", fontsize=9)
        ax.set_xlabel("α", fontsize=8)

# Row 3: fraction where P(correct) > P(wrong) vs alpha, per layer
for col, L in enumerate(STEER_LAYERS):
    ax = axes[3, col]
    for cond, correct_key, wrong_key, c_correct, c_wrong, _ in CONDS:
        fracs = []
        for alpha in ALPHAS:
            entries = results[cond][L][alpha]
            p_corrects = [e[2] if correct_key == "a1" else e[1] for e in entries]
            p_wrongs   = [e[1] if wrong_key   == "a0" else e[2] for e in entries]
            fracs.append(np.mean([pc > pw for pc, pw in zip(p_corrects, p_wrongs)]))
        label = "hi-e1" if cond == "hi-e1" else "hi-confused"
        ax.plot(ALPHAS, fracs, color=c_correct, lw=2, marker="o", ms=3, label=label)
    ax.axhline(0.5, color="gray", ls=":", lw=1)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlim(-5, ALPHAS[-1] + 5)
    ax.set_xlabel("α", fontsize=8)
    if col == 0:
        ax.set_ylabel("Frac. P(correct)>P(wrong)", fontsize=8)
        ax.legend(fontsize=7)

fig.suptitle(
    f"SWAP steering: P(a0), P(a1), and answer flip  ({cfg['name']})\n"
    f"SWAP = push △-country → □ AND □-country → △ simultaneously at one layer.\n"
    f"Row 1: base (correct=a0).  Row 2: hi-e1 (correct=a1).  Row 3: hi-confused (correct=a0).  "
    f"Row 4: fraction where model prefers correct answer.  n={args.n_test} examples.",
    fontsize=10)
plt.tight_layout()

out = os.path.join(OUTDIR, "binding_id_swap.png")
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.close()
print(f"\nSaved {out}")

# ── Output figure: what did the model actually produce? ───────────────────────
# 3 rows (conditions) × 2 cols (α=5, α=10)
# Each panel: lines across layers — fraction output=correct/wrong/other
# Dashed = baseline (α=0), solid = steered

PLOT_ALPHAS = [5, 10]

def outcome_fracs(entries):
    n = len(entries)
    if n == 0:
        return 0, 0, 0
    fc = sum(1 for e in entries if e[3] == "correct") / n
    fw = sum(1 for e in entries if e[3] == "wrong")   / n
    fo = sum(1 for e in entries if e[3] == "other")   / n
    return fc, fw, fo

fig2, axes2 = plt.subplots(len(CONDS), len(PLOT_ALPHAS),
                            figsize=(10, 4 * len(CONDS)), sharey=True, sharex=True)

for row, (cond, correct_key, wrong_key, c_correct, c_wrong, row_title) in enumerate(CONDS):
    for col, plot_alpha in enumerate(PLOT_ALPHAS):
        ax = axes2[row, col]

        fc_base, fw_base, fo_base = [], [], []
        fc_strd, fw_strd, fo_strd = [], [], []
        for L in STEER_LAYERS:
            fc, fw, fo = outcome_fracs(results[cond][L][0])
            fc_base.append(fc); fw_base.append(fw); fo_base.append(fo)
            fc, fw, fo = outcome_fracs(results[cond][L][plot_alpha])
            fc_strd.append(fc); fw_strd.append(fw); fo_strd.append(fo)

        x = STEER_LAYERS
        # Baseline (dashed, thin)
        ax.plot(x, fc_base, color="seagreen", ls="--", lw=1, alpha=0.5)
        ax.plot(x, fw_base, color="crimson",  ls="--", lw=1, alpha=0.5)
        ax.plot(x, fo_base, color="gray",     ls="--", lw=1, alpha=0.5)
        # Steered (solid, thick)
        ax.plot(x, fc_strd, color="seagreen", ls="-", lw=2.5, marker="o", ms=5,
                label="output = correct country")
        ax.plot(x, fw_strd, color="crimson",  ls="-", lw=2.5, marker="s", ms=5,
                label="output = wrong country")
        ax.plot(x, fo_strd, color="gray",     ls="-", lw=2.5, marker="^", ms=5,
                label="output = other token")

        ax.set_xticks(STEER_LAYERS)
        ax.set_ylim(-0.05, 1.05)
        ax.axhline(0, color="black", lw=0.5, ls=":")
        ax.set_xlabel("Layer", fontsize=9)
        if col == 0:
            ax.set_ylabel(f"{cond}\nFraction of examples", fontsize=9)
        if row == 0:
            ax.set_title(f"α = {plot_alpha}", fontsize=10)

# Single shared legend
handles, labels = axes2[0, 0].get_legend_handles_labels()
fig2.legend(handles, labels, loc="upper center", ncol=3,
            fontsize=9, bbox_to_anchor=(0.5, 1.02))
fig2.suptitle(
    f"Model actual top-1 output after SWAP  ({cfg['name']})\n"
    "Solid = after SWAP.  Dashed = baseline (α=0).  "
    "Green = correct country, Red = wrong country, Gray = other token.",
    fontsize=10, y=1.06)
plt.tight_layout()
out2 = os.path.join(OUTDIR, "binding_id_swap_outputs.png")
plt.savefig(out2, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved {out2}")

# ── Print summary table ────────────────────────────────────────────────────────
print("\nSummary — P(correct), P(wrong), frac(correct>wrong) across all alphas:")
for cond, correct_key, wrong_key, *_ in CONDS:
    print(f"\n=== {cond} ===")
    print(f"{'Layer':<7} " + "  ".join(f"α={a:<4}" for a in ALPHAS))
    print(f"{'':7} " + "  ".join(f"{'P(c)/P(w)/acc':<14}" for _ in ALPHAS))
    for L in STEER_LAYERS:
        row_str = f"L={L:<5} "
        for alpha in ALPHAS:
            entries = results[cond][L][alpha]
            p_corrects = [e[2] if correct_key == "a1" else e[1] for e in entries]
            p_wrongs   = [e[1] if wrong_key   == "a0" else e[2] for e in entries]
            pc = np.mean(p_corrects)
            pw = np.mean(p_wrongs)
            acc = np.mean([pc_ > pw_ for pc_, pw_ in zip(p_corrects, p_wrongs)])
            row_str += f"{pc:.2f}/{pw:.2f}/{acc:.2f}  "
        print(row_str)
print("\nDone.")
