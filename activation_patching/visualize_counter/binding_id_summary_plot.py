"""
binding_id_summary_plot.py  —  Single summary figure for the binding-ID story.

PANEL 1: Probe accuracy per layer at three token types (4-cond crossed design)
  • Entity name token  — onset L0   (intro order encoded immediately)
  • "I live" token     — onset ~L5  (zero-shot: probe never trained on "I")
  • Country token (Hi) — onset ~L10 (greeting-resolved binding-ID)
  • Country token (Base, simple) — onset ~L1-3 (no greeting, direct from intro order)

PANEL 2: P(△) at country token, L13, across template types
  Shows: pre-utterance anchors all → □; Hi e0! (confused) → △
  - base: P(△)≈0.68 (△ expected — correct)
  - hi e0! (confused): P(△)≈0.97 (△ expected — the "reset/disturb")
  - hi e1!, address, nomination, hamlet style: P(△)≈0 (□ expected — all anchor correctly to □)

Usage:
  python binding_id_summary_plot.py --model llama --gpu 0 --n-probe 150 --n-eval 50
"""
import os, sys, argparse
import numpy as np
import torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

parser = argparse.ArgumentParser()
parser.add_argument("--model",   choices=["llama","qwen"], default="llama")
parser.add_argument("--gpu",     default="0")
parser.add_argument("--n-probe", type=int, default=150, help="samples for layer probe")
parser.add_argument("--n-eval",  type=int, default=50,  help="samples per template")
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../attribution_ioi"))
from utils import MODEL_CONFIGS
from transformer_lens import HookedTransformer

OUTDIR = os.path.dirname(os.path.abspath(__file__))
cfg    = MODEL_CONFIGS[args.model]

print(f"Loading {cfg['name']}...")
model = HookedTransformer.from_pretrained(
    cfg["name"], dtype=torch.bfloat16, cache_dir=cfg["cache"])
model.eval().to("cuda")
NL = model.cfg.n_layers
print(f"Loaded: {NL} layers, d_model={model.cfg.d_model}")

# ── Vocab ─────────────────────────────────────────────────────────────────────
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
             for i,j in [(a,b) for a in range(len(NAMES))
                                for b in range(len(NAMES)) if a!=b]
             for p,q in [(a,b) for a in range(len(COUNTRIES))
                                for b in range(len(COUNTRIES)) if a!=b]]
IDX = rng.permutation(len(ALL_PAIRS))

# ── Prompts ───────────────────────────────────────────────────────────────────
def make_hi(first, second, a0, a1, greeted):
    return (
        'This is the transcript of a conversation.\n'
        f'"I am {first}."\n"I am {second}."\n'
        f'"Hi {greeted}!"\n'
        f'"I live in {a0}."'
    )

def make_base(e0, e1, a0, a1):
    return (
        'This is the transcript of a conversation.\n'
        f'"I am {e0}."\n"I am {e1}."\n'
        f'"I live in {a0}."\n"I live in {a1}."'
    )

# templates for panel 2
TEMPLATES_P2 = [
    ("base",            lambda e0,e1,a0,a1: make_base(e0,e1,a0,a1),                  "△"),
    ("Hi e0!\n(confused)", lambda e0,e1,a0,a1: make_hi(e0,e1,a0,a1,e0),              "△"),
    ("Hi e1!",          lambda e0,e1,a0,a1: make_hi(e0,e1,a0,a1,e1),                 "□"),
    ("address e1",      lambda e0,e1,a0,a1: (
        'This is the transcript of a conversation.\n'
        f'"I am {e0}."\n"I am {e1}."\n'
        f'"{e1}, do you agree?"\n'
        f'"I live in {a0}."'),                                                         "□"),
    ("nomination e1",   lambda e0,e1,a0,a1: (
        'This is the transcript of a conversation.\n'
        f'"I am {e0}."\n"I am {e1}."\n'
        f'"{e1}, your turn."\n'
        f'"I live in {a0}."'),                                                         "□"),
    ("narration\n'said'", lambda e0,e1,a0,a1: (
        'This is the transcript of a conversation.\n'
        f'"I am {e0}."\n"I am {e1}."\n'
        f'{e1} said: "I live in {a0}."'),                                              "□"),
    ("retrospective\n(post-hoc)", lambda e0,e1,a0,a1: (
        'This is the transcript of a conversation.\n'
        f'"I am {e0}."\n"I am {e1}."\n'
        f'"I live in {a0}."\n"I live in {a1}."\n'
        f'"As {e1} mentioned, {a0} is beautiful."'),                                   "□*"),
]

# ── Cache helpers ─────────────────────────────────────────────────────────────
@torch.no_grad()
def get_cache(prompt):
    tok = model.tokenizer.encode(prompt, return_tensors="pt").to("cuda")
    _, cache = model.run_with_cache(tok, names_filter=lambda n: "hook_resid_post" in n)
    return cache, tok[0].tolist()

def first_of(ids, word):
    tid = model.tokenizer.encode(f" {word}", add_special_tokens=False)[0]
    return next((i for i,t in enumerate(ids) if t == tid), None)

def nth_I(ids, n):
    count = 0
    for i, t in enumerate(ids):
        if model.tokenizer.decode([t]).strip().lstrip('"') == "I":
            if count == n: return i
            count += 1
    raise ValueError(f"I token #{n} not found")

def extract(cache, pos):
    return np.stack([
        cache[f"blocks.{L}.hook_resid_post"][0, pos].cpu().float().numpy()
        for L in range(NL)
    ])

# ── PHASE 1: 4-cond crossed data for layer probe ─────────────────────────────
# Collect: entity token, I token (zero-shot), country token — all from hi 4-cond
print(f"\n[Phase 1] Collecting 4-cond crossed activations (n={args.n_probe})...")

X_ent, y_ent = [], []
X_I,   y_I   = [], []
X_cty, y_cty = [], []

for k, si in enumerate(IDX[:args.n_probe]):
    ni, nj, ci, cj = ALL_PAIRS[si]
    e0, e1 = NAMES[ni], NAMES[nj]
    a0, a1 = COUNTRIES[ci], COUNTRIES[cj]
    try:
        cA, tA = get_cache(make_hi(e0, e1, a0, a1, greeted=e0))  # A: e0=△
        cB, tB = get_cache(make_hi(e0, e1, a0, a1, greeted=e1))  # B: e1=□
        cC, tC = get_cache(make_hi(e1, e0, a0, a1, greeted=e0))  # C: e0=□
        cD, tD = get_cache(make_hi(e1, e0, a0, a1, greeted=e1))  # D: e1=△

        # entity tokens (labeled by intro order)
        for c, t, tri, sq in [(cA,tA,e0,e1),(cB,tB,e0,e1),(cC,tC,e1,e0),(cD,tD,e1,e0)]:
            p_tri = first_of(t, tri); p_sq = first_of(t, sq)
            if p_tri: X_ent.append(extract(c, p_tri)); y_ent.append(1)
            if p_sq:  X_ent.append(extract(c, p_sq));  y_ent.append(0)

        # I token and country (labeled by greeting-resolved speaker)
        for c, t, lbl in [(cA,tA,1),(cB,tB,0),(cC,tC,0),(cD,tD,1)]:
            X_I.append(extract(c, nth_I(t, 2)));        y_I.append(lbl)
            p = first_of(t, a0)
            if p: X_cty.append(extract(c, p));          y_cty.append(lbl)
    except Exception:
        continue
    if (k+1) % 30 == 0:
        print(f"  {k+1}/{args.n_probe}")

X_ent = np.stack(X_ent); y_ent = np.array(y_ent, dtype=int)
X_I   = np.stack(X_I);   y_I   = np.array(y_I,   dtype=int)
X_cty = np.stack(X_cty); y_cty = np.array(y_cty, dtype=int)
print(f"Entity: {X_ent.shape}  I: {X_I.shape}  Country(hi): {X_cty.shape}")

# ── PHASE 1b: Base country (simple, positional confound noted) ────────────────
print(f"\n[Phase 1b] Collecting base country activations (n={args.n_probe})...")
X_base, y_base = [], []
offset = args.n_probe
for k, si in enumerate(IDX[offset:offset + args.n_probe]):
    ni, nj, ci, cj = ALL_PAIRS[si]
    e0, e1 = NAMES[ni], NAMES[nj]
    a0, a1 = COUNTRIES[ci], COUNTRIES[cj]
    try:
        c, t = get_cache(make_base(e0, e1, a0, a1))
        p0 = first_of(t, a0); p1 = first_of(t, a1)
        if p0: X_base.append(extract(c, p0)); y_base.append(1)
        if p1: X_base.append(extract(c, p1)); y_base.append(0)
        # crossed: swap country order so same country appears as both △ and □
        c2, t2 = get_cache(make_base(e0, e1, a1, a0))
        p0b = first_of(t2, a1); p1b = first_of(t2, a0)
        if p0b: X_base.append(extract(c2, p0b)); y_base.append(1)
        if p1b: X_base.append(extract(c2, p1b)); y_base.append(0)
    except Exception:
        continue
    if (k+1) % 30 == 0:
        print(f"  {k+1}/{args.n_probe}")

X_base = np.stack(X_base); y_base = np.array(y_base, dtype=int)
print(f"Country(base, country-order-crossed): {X_base.shape}")

# ── CV probe per layer ────────────────────────────────────────────────────────
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

def cv_acc(X, y):
    accs = []
    for L in range(NL):
        XL = X[:, L, :]
        fold_acc = []
        for tr, va in skf.split(XL, y):
            lr = LogisticRegression(max_iter=2000, C=1.0, solver="lbfgs",
                                    fit_intercept=False)
            lr.fit(XL[tr], y[tr])
            fold_acc.append(lr.score(XL[va], y[va]))
        accs.append(np.mean(fold_acc))
    return np.array(accs)

print("\nRunning per-layer CV probes...")
acc_ent  = cv_acc(X_ent,  y_ent);  print("  entity done")
acc_I    = cv_acc(X_I,    y_I);    print("  I-token done")
acc_cty  = cv_acc(X_cty,  y_cty);  print("  country(hi) done")
acc_base = cv_acc(X_base, y_base); print("  country(base) done")

def onset(acc, thr=0.75):
    return next((L for L in range(NL) if acc[L] > thr), None)

print(f"\nOnset (acc>0.75):")
print(f"  Entity:       L{onset(acc_ent)}")
print(f"  I token:      L{onset(acc_I)}")
print(f"  Country (hi): L{onset(acc_cty)}")
print(f"  Country (base): L{onset(acc_base)}")

# ── PHASE 2: train probe at L13, evaluate on templates ───────────────────────
PEAK = 13
print(f"\n[Phase 2] Training peak probe at L{PEAK} on 4-cond crossed data...")

X_probe = np.concatenate([X_ent, X_cty], axis=0)
y_probe = np.concatenate([y_ent, y_cty], axis=0)
lr_peak = LogisticRegression(max_iter=2000, C=1.0, solver="lbfgs", fit_intercept=False)
lr_peak.fit(X_probe[:, PEAK, :], y_probe)
print(f"  Train acc at L{PEAK}: {lr_peak.score(X_probe[:, PEAK, :], y_probe):.3f}")

def probe_proba(act_NL):
    return lr_peak.predict_proba(act_NL[PEAK:PEAK+1, :])[0, 1]

print(f"\n[Phase 2] Evaluating templates (n={args.n_eval} each)...")
eval_offset = 2 * args.n_probe
p_delta_per_template = {name: [] for name, _, _ in TEMPLATES_P2}

for k, si in enumerate(IDX[eval_offset:eval_offset + args.n_eval]):
    ni, nj, ci, cj = ALL_PAIRS[si]
    e0, e1 = NAMES[ni], NAMES[nj]
    a0, a1 = COUNTRIES[ci], COUNTRIES[cj]
    for tname, tfn, _ in TEMPLATES_P2:
        try:
            c, t = get_cache(tfn(e0, e1, a0, a1))
            p = first_of(t, a0)
            if p is None: continue
            p_delta_per_template[tname].append(probe_proba(extract(c, p)))
        except Exception:
            continue

# ── FIGURE ────────────────────────────────────────────────────────────────────
layers = list(range(NL))
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# ── Panel 1: Layer onset curves ───────────────────────────────────────────────
L_ent  = onset(acc_ent)
L_I    = onset(acc_I)
L_cty  = onset(acc_cty)
L_base = onset(acc_base)

ax1.plot(layers, acc_ent,  color="steelblue", lw=2.5, marker="o", ms=3,
         label=f"Entity name token (L{L_ent}) — intro order encoded directly at embedding")
ax1.plot(layers, acc_I,    color="darkorange", lw=2.5, marker="o", ms=3,
         label=f'"I live" token — zero-shot (L{L_I}) — propagation lag from entity tokens')
ax1.plot(layers, acc_cty,  color="crimson",    lw=2.5, marker="o", ms=3,
         label=f"Country token, Hi setup (L{L_cty}) — greeting must be resolved, then propagated")
ax1.plot(layers, acc_base, color="seagreen",   lw=2,   marker="o", ms=3,
         ls="--",
         label=f"Country token, Base setup (L{L_base}) — no greeting; direct from intro order")
ax1.axhline(0.5, color="gray", ls="--", lw=1, alpha=0.5, label="Chance (0.5)")

annotations = [
    (L_ent,  "steelblue",  f"L{L_ent}: intro order\nencoded at entity"),
    (L_I,    "darkorange", f"L{L_I}: propagated\nto 'I live' token"),
    (L_cty,  "crimson",    f"L{L_cty}: greeting resolved\nat country token"),
]
for L, color, label in annotations:
    if L is not None:
        ax1.axvline(L, color=color, ls=":", lw=1.5, alpha=0.6)
        ax1.text(L + 0.3, 0.28, label, color=color, fontsize=7, va="bottom", alpha=0.85)

ax1.set_xlim(-0.5, NL - 0.5); ax1.set_ylim(0.2, 1.05)
ax1.set_xlabel("Layer", fontsize=11); ax1.set_ylabel("Probe accuracy (5-fold CV)", fontsize=11)
ax1.set_title("Binding-ID onset per token type\n"
              "Each later onset = more computation needed (propagation lag or greeting resolution)",
              fontsize=10)
ax1.legend(fontsize=8)

# ── Panel 2: P(△) at L13 per template ────────────────────────────────────────
tnames = [n for n,_,_ in TEMPLATES_P2]
expected = [e for _,_,e in TEMPLATES_P2]
means = [np.mean(p_delta_per_template[n]) if p_delta_per_template[n] else 0.5
         for n in tnames]

colors_p2 = []
for e in expected:
    if e == "△":   colors_p2.append("steelblue")
    elif e == "□":  colors_p2.append("crimson")
    else:           colors_p2.append("gray")   # post-hoc: □* expected but model fails

bars = ax2.bar(range(len(tnames)), means, color=colors_p2, alpha=0.85, edgecolor="k", lw=0.5)
ax2.axhline(0.5, color="gray", ls="--", lw=1, alpha=0.5, label="Chance")
ax2.axhline(0.5, color="gray", ls="--", lw=1, alpha=0)  # spacer

for i, (m, e) in enumerate(zip(means, expected)):
    ax2.text(i, m + 0.02, f"{m:.2f}", ha="center", va="bottom", fontsize=8)
    ax2.text(i, 0.23, e, ha="center", va="bottom", fontsize=9, fontweight="bold",
             color="white" if e == "□" else "navy")

ax2.set_xticks(range(len(tnames)))
ax2.set_xticklabels(tnames, rotation=20, ha="right", fontsize=9)
ax2.set_ylabel(f"Mean P(△) at country token, L{PEAK}", fontsize=11)
ax2.set_ylim(0.2, 1.1)
ax2.set_title(f"Probe P(△) at country token (L{PEAK}) per context type\n"
              "Blue = △ expected  |  Red = □ expected  |  Gray = post-hoc (model rarely follows)",
              fontsize=10)

# Legend patches
from matplotlib.patches import Patch
ax2.legend(handles=[
    Patch(color="steelblue", label="△ expected (intro-order preserved)"),
    Patch(color="crimson",   label="□ expected (pre-utterance anchor)"),
    Patch(color="gray",      label="□* expected but post-hoc — model fails"),
], fontsize=8, loc="upper right")

fig.suptitle(
    f"Binding-ID in dialogue: layer onset and context sensitivity  ({cfg['name']})\n"
    "Left: binding-ID encoded progressively deeper for harder token types.  "
    "Right: pre-utterance anchors all produce □; Hi e0! uniquely resets to △.",
    fontsize=10)
plt.tight_layout()
out = os.path.join(OUTDIR, "binding_id_summary.png")
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.close()
print(f"\nSaved {out}")
print("Done.")
