"""
counter_token_heatmap.py — Token-level binding-ID heatmap (Phase 3 only)

For base / hi-e1 / hi-confused: shows P(△) at every token × every layer,
averaged over N examples. Irrelevant tokens (not entity names, "I" tokens,
or country) are greyed out to avoid OOD artifacts.

Usage:
  python counter_token_heatmap.py --model llama --gpu 2 --n-probe 100 --n-avg 20
"""
import os, sys, argparse
import numpy as np
import torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from probe_io import cache_path, save_probes, load_probes
from matplotlib.gridspec import GridSpec
from sklearn.linear_model import LogisticRegression

parser = argparse.ArgumentParser()
parser.add_argument("--model",   choices=["llama","qwen"], default="llama")
parser.add_argument("--gpu",     default="2")
parser.add_argument("--n-probe", type=int, default=100)
parser.add_argument("--n-avg",   type=int, default=20)
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
        f'"I live in {a0}."\n"I live in {a1}."'
    )

def make_hi(e0, e1, a0, a1, greeted):
    return (
        'This is the transcript of a conversation.\n'
        f'"I am {e0}."\n"I am {e1}."\n'
        f'"Hi {greeted}!"\n'
        f'"I live in {a0}. What about you?"\n'
        f'"I live in {a1}."'
    )

# ── Cache helpers ──────────────────────────────────────────────────────────────
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
    return None

def extract_all(cache, seq_len):
    return np.stack([
        cache[f"blocks.{L}.hook_resid_post"][0, :seq_len].cpu().float().numpy()
        for L in range(NL)
    ])  # (NL, seq_len, d_model)

def tok_label(t):
    s = model.tokenizer.decode([t])
    return s.replace("\n", "↵")[:10]

# ── Train per-layer probes (or load from cache) ───────────────────────────────
def extract_pos(cache, pos):
    return np.stack([
        cache[f"blocks.{L}.hook_resid_post"][0, pos].cpu().float().numpy()
        for L in range(NL)
    ])

_cache = cache_path(OUTDIR, args.model, args.n_probe)
if os.path.exists(_cache):
    print(f"\n[Phase 1] Loading cached probes (n={args.n_probe})...")
    probes = load_probes(_cache)
else:
    print(f"\n[Phase 1] Collecting probe training data (n={args.n_probe})...")
    X_probe, y_probe = [], []
    for k, si in enumerate(IDX[:args.n_probe]):
        ni, nj, ci, cj = ALL_PAIRS[si]
        e0, e1 = NAMES[ni], NAMES[nj]
        a0, a1 = COUNTRIES[ci], COUNTRIES[cj]
        try:
            cA, tA = get_cache(make_hi(e0, e1, a0, a1, greeted=e0))
            cB, tB = get_cache(make_hi(e0, e1, a0, a1, greeted=e1))
            cC, tC = get_cache(make_hi(e1, e0, a0, a1, greeted=e0))
            cD, tD = get_cache(make_hi(e1, e0, a0, a1, greeted=e1))
            for c, t, tri, sq in [(cA,tA,e0,e1),(cB,tB,e0,e1),(cC,tC,e1,e0),(cD,tD,e1,e0)]:
                p_tri = first_of(t, tri); p_sq = first_of(t, sq)
                if p_tri: X_probe.append(extract_pos(c, p_tri)); y_probe.append(1)
                if p_sq:  X_probe.append(extract_pos(c, p_sq));  y_probe.append(0)
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
    print(f"  {X_probe.shape}, balance={y_probe.mean():.2f}")

    print("Training per-layer probes...")
    probes = []
    for L in range(NL):
        lr = LogisticRegression(max_iter=2000, C=1.0, solver="lbfgs", fit_intercept=False)
        lr.fit(X_probe[:, L, :], y_probe)
        probes.append(lr)
    print(f"  L0={probes[0].score(X_probe[:,0,:],y_probe):.3f}  "
          f"L10={probes[10].score(X_probe[:,10,:],y_probe):.3f}  "
          f"L13={probes[13].score(X_probe[:,13,:],y_probe):.3f}")
    save_probes(probes, _cache)

# ── Collect averaged heatmaps ──────────────────────────────────────────────────
SETUPS = [
    ("Base\n(e0=△ speaks a0,  e1=□ speaks a1)",
     lambda e0,e1,a0,a1: make_base(e0,e1,a0,a1), True),
    ("Hi e1!\n(e1=□ speaks a0,  e0=△ speaks a1)",
     lambda e0,e1,a0,a1: make_hi(e0,e1,a0,a1,e1), True),
    ("Hi e0! confused\n(e0=△ speaks a0,  e1=□ speaks a1)",
     lambda e0,e1,a0,a1: make_hi(e0,e1,a0,a1,e0), True),
]

print(f"\n[Phase 3] Averaging heatmaps (n={args.n_avg} per setup)...")
heatmap_data = []

for stitle, pfn, has_a1 in SETUPS:
    mats, ref_labels, rel_pos = [], None, None

    for si in IDX[args.n_probe : args.n_probe + args.n_avg]:
        ni, nj, ci, cj = ALL_PAIRS[si]
        e0, e1 = NAMES[ni], NAMES[nj]
        a0, a1 = COUNTRIES[ci], COUNTRIES[cj]
        try:
            c, ids = get_cache(pfn(e0, e1, a0, a1))
            seq_len = len(ids)
            all_acts = extract_all(c, seq_len)

            mat = np.zeros((NL, seq_len))
            for L in range(NL):
                mat[L] = probes[L].predict_proba(all_acts[L])[:, 1]
            mats.append(mat)

            if ref_labels is None:
                ref_labels = [tok_label(t) for t in ids]
                rel = set(filter(None, [
                    first_of(ids, e0), first_of(ids, e1),
                    nth_I(ids, 0), nth_I(ids, 1), nth_I(ids, 2),
                    first_of(ids, a0),
                ]))
                if has_a1:
                    rel |= set(filter(None, [nth_I(ids, 3), first_of(ids, a1)]))
                rel_pos = rel
        except Exception:
            continue

    avg_mat = np.mean(mats, axis=0)
    heatmap_data.append((stitle, avg_mat, ref_labels, rel_pos))
    print(f"  '{stitle.split(chr(10))[0]}': {len(ref_labels)} tokens, "
          f"{len(rel_pos)} relevant, {len(mats)} examples averaged")

# ── Figure ─────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 9))

cmap = matplotlib.cm.RdBu_r.copy()
cmap.set_bad(color="lightgrey")

for ax, (stitle, avg_mat, tok_strs, rel_pos) in zip(axes, heatmap_data):
    seq_len  = len(tok_strs)
    display  = avg_mat.T.copy()   # (seq_len, NL)
    for i in range(seq_len):
        if i not in rel_pos:
            display[i, :] = np.nan

    im = ax.imshow(display, aspect="auto", cmap=cmap, vmin=0, vmax=1, origin="upper")
    ax.set_xticks([0, 5, 10, 15, 20, 25, 30])
    ax.set_xlabel("Layer", fontsize=10)
    ax.set_yticks(range(seq_len))
    ax.set_yticklabels(tok_strs, fontsize=7, family="monospace")
    for tick, i in zip(ax.get_yticklabels(), range(seq_len)):
        tick.set_fontweight("bold"  if i in rel_pos else "normal")
        tick.set_color("black"      if i in rel_pos else "#bbbbbb")
    ax.set_title(stitle, fontsize=10)
    plt.colorbar(im, ax=ax, label="P(△)", shrink=0.6)

fig.suptitle(
    f"Binding-ID token-level attribution  ({cfg['name']})\n"
    f"P(△) at every token × every layer, averaged over {args.n_avg} examples.  "
    "Grey = masked (OOD — not entity/I/country tokens).  "
    "Bold rows = in-distribution positions.",
    fontsize=10)
plt.tight_layout()

out = os.path.join(OUTDIR, "binding_id_token_heatmap.png")
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.close()
print(f"\nSaved {out}")
print("Done.")
