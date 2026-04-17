"""
probe_two_alpha_steer_rewrite.py — Probe-direction SWAP, all layers, layer-diff injection.

Matches the mean-intervention logic from binding_id_mean_intervention.py:
  - Hook resid_pre at ALL layers (not just a few hand-picked ones)
  - Layer-diff injection: add alpha*(d[l] - d[l-1]) at each layer so the
    CUMULATIVE effect at layer l equals alpha*d[l]
  - slot-tri gets -alpha*d[l], slot-sq gets +alpha*d[l] (swap)

Usage:
  python probe_two_alpha_steer_rewrite.py --model llama --gpu 1 --n-eval 100
  python probe_two_alpha_steer_rewrite.py --model llama --gpu 1 --alpha-e 2 --alpha-c 3 --n-eval 100
"""
import os, sys, argparse
import numpy as np
import torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from probe_io import cache_path, load_probes

parser = argparse.ArgumentParser()
parser.add_argument("--model",   choices=["llama","qwen"], default="llama")
parser.add_argument("--gpu",     default="1")
parser.add_argument("--n-probe", type=int, default=100)
parser.add_argument("--n-eval",  type=int, default=100)
parser.add_argument("--alpha-e",   type=float, default=2.0, help="alpha for entity tokens")
parser.add_argument("--alpha-c",   type=float, default=3.0, help="alpha for country tokens")
parser.add_argument("--max-layer", type=int,   default=None,
                    help="only inject up to this layer (inclusive); default=all layers")
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../attribution_ioi"))
from utils import MODEL_CONFIGS
from transformer_lens import HookedTransformer

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROBE_DIR  = os.path.join(SCRIPT_DIR, "..")   # probes live in visualize_counter/
OUTDIR     = SCRIPT_DIR                        # plots go in best/
cfg    = MODEL_CONFIGS[args.model]

print(f"Loading {cfg['name']}...")
model = HookedTransformer.from_pretrained(cfg["name"], dtype=torch.bfloat16, cache_dir=cfg["cache"])
model.eval().to("cuda")
NL = model.cfg.n_layers
print(f"Loaded: {NL} layers")
print(f"alpha_e={args.alpha_e}, alpha_c={args.alpha_c}")

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
def first_of(ids, word):
    tid = model.tokenizer.encode(f" {word}", add_special_tokens=False)[0]
    return next((i for i,t in enumerate(ids) if t == tid), None)

def tok_id(word):
    return model.tokenizer.encode(f" {word}", add_special_tokens=False)[0]

# ── Load probes ────────────────────────────────────────────────────────────────
_cache = cache_path(PROBE_DIR, args.model, args.n_probe)
if os.path.exists(_cache):
    print("Loading cached probes...")
    probes = load_probes(_cache)
else:
    print("No cached probes found — train first")
    sys.exit(1)

# Normalized probe directions: [NL, d_model]
directions = np.stack([probes[L].coef_[0] / np.linalg.norm(probes[L].coef_[0])
                       for L in range(NL)])   # float64, shape [NL, d_model]

# ── Layer-diff injection (mirrors mean_intervention logic) ─────────────────────
# We want cumulative effect at layer l = alpha * d[l].
# At resid_pre[l] we inject alpha * (d[l] - d[l-1]), with d[-1] = 0.
# Result: sum_{k<=l} inject[k] = alpha * d[l].  ✓
def make_layer_diff(dirs, alpha, max_layer=None):
    """dirs: [NL, D] float64 -> diff_tensor: [NL, D] float32
    Layers > max_layer get 0 injection (cumulative stays at alpha*d[max_layer])."""
    if max_layer is None:
        max_layer = len(dirs) - 1
    scaled = alpha * dirs                        # [NL, D]
    diff   = np.zeros_like(scaled)
    diff[0] = scaled[0]
    diff[1:max_layer+1] = scaled[1:max_layer+1] - scaled[:max_layer]
    # layers beyond max_layer stay 0 — cumulative frozen at alpha*d[max_layer]
    return torch.tensor(diff, dtype=torch.float32)  # [NL, D]

@torch.no_grad()
def run_swap(prompt, patch_ent, patch_cty,
             pos_tri_ent, pos_sq_ent, pos_tri_cty, pos_sq_cty,
             diff_E, diff_A):
    """
    Hook resid_pre at every layer.
    slot-tri gets -diff[l], slot-sq gets +diff[l].
    Cumulative effect at layer l: tri shifted by -alpha*d[l], sq by +alpha*d[l].
    """
    tok = torch.tensor([model.tokenizer.encode(prompt)], device="cuda")
    hooks = []

    for l in range(NL):
        def make_hook(layer, _dE, _dA, _p_tri_e, _p_sq_e, _p_tri_c, _p_sq_c):
            def hook_fn(resid, hook):
                if _dE is not None:
                    d = _dE[layer].to(resid.device)
                    resid[0, _p_tri_e] -= d
                    resid[0, _p_sq_e]  += d
                if _dA is not None:
                    d = _dA[layer].to(resid.device)
                    resid[0, _p_tri_c] -= d
                    resid[0, _p_sq_c]  += d
                return resid
            return hook_fn

        e_arg = diff_E if patch_ent else None
        a_arg = diff_A if patch_cty else None
        hooks.append((
            f"blocks.{l}.hook_resid_pre",
            make_hook(l, e_arg, a_arg,
                      pos_tri_ent, pos_sq_ent, pos_tri_cty, pos_sq_cty)
        ))

    logits = model.run_with_hooks(tok, fwd_hooks=hooks)
    return logits[0, -1].cpu().float().numpy()

def answer_correct(logits_np, correct, wrong):
    return float(logits_np[tok_id(correct)] > logits_np[tok_id(wrong)])

# ── Precompute layer-diff tensors ──────────────────────────────────────────────
max_layer = args.max_layer  # None = all layers
diff_E = make_layer_diff(directions, args.alpha_e, max_layer)  # [NL, D]
diff_A = make_layer_diff(directions, args.alpha_c, max_layer)  # [NL, D]
inj_range = f"0-{max_layer if max_layer is not None else NL-1}"

# ── Conditions ─────────────────────────────────────────────────────────────────
INTERV_SETS = [
    # (label, patch_ent, patch_cty)
    ("control",      False, False),
    ("entity-only",  True,  False),
    ("country-only", False, True),
    ("both",         True,  True),
]

CONDS = [
    # (label, prompt_fn, tri-entity, sq-entity, correct_key)
    ("base",        lambda e0,e1,a0,a1: make_base(e0,e1,a0,a1),  "e0","e1","a0"),
    ("hi-e1",       lambda e0,e1,a0,a1: make_hi(e0,e1,a0,a1,e1), "e1","e0","a1"),
    ("hi-confused", lambda e0,e1,a0,a1: make_hi(e0,e1,a0,a1,e0), "e0","e1","a0"),
]

results = {
    iname: {cname: [] for cname, *_ in CONDS}
    for iname, *_ in INTERV_SETS
}

print(f"\nRunning (n={args.n_eval})  [layers {inj_range} of {NL-1}, layer-diff injection]...")
eval_offset = args.n_probe

for k, si in enumerate(IDX[eval_offset : eval_offset + args.n_eval]):
    ni, nj, ci, cj = ALL_PAIRS[si]
    e0, e1 = NAMES[ni], NAMES[nj]
    a0, a1 = COUNTRIES[ci], COUNTRIES[cj]

    for cname, pfn, tri_e, sq_e, correct_key in CONDS:
        prompt  = pfn(e0, e1, a0, a1)
        ids     = model.tokenizer.encode(prompt)
        tri_ent = e0 if tri_e == "e0" else e1
        sq_ent  = e0 if sq_e  == "e0" else e1
        correct = a0 if correct_key == "a0" else a1
        wrong   = a1 if correct_key == "a0" else a0

        p_tri_ent = first_of(ids, tri_ent)
        p_sq_ent  = first_of(ids, sq_ent)
        p_tri_cty = first_of(ids, a0)
        p_sq_cty  = first_of(ids, a1)
        if any(p is None for p in [p_tri_ent, p_sq_ent, p_tri_cty, p_sq_cty]):
            continue

        try:
            baseline_logits = None
            for iname, pe, pc in INTERV_SETS:
                if not pe and not pc:
                    if baseline_logits is None:
                        tok = torch.tensor([model.tokenizer.encode(prompt)], device="cuda")
                        with torch.no_grad():
                            baseline_logits = model(tok)[0, -1].cpu().float().numpy()
                    logits = baseline_logits
                else:
                    logits = run_swap(prompt, pe, pc,
                                      p_tri_ent, p_sq_ent, p_tri_cty, p_sq_cty,
                                      diff_E, diff_A)
                results[iname][cname].append(answer_correct(logits, correct, wrong))
        except Exception:
            continue

    if (k+1) % 20 == 0:
        print(f"  {k+1}/{args.n_eval}")

# ── Summary table ──────────────────────────────────────────────────────────────
print(f"\nResults  (alpha_e={args.alpha_e}, alpha_c={args.alpha_c}, layers {inj_range})")
print()
print(f"  {'':14s}", end="")
for cname, *_ in CONDS:
    print(f"  {cname:>15s}", end="")
print()
print(f"  {'-'*14}", end="")
for _ in CONDS:
    print(f"  {'':>15s}", end="")
print()
for iname, *_ in INTERV_SETS:
    print(f"  {iname:14s}", end="")
    for cname, *_ in CONDS:
        vals = results[iname][cname]
        acc  = np.mean(vals) if vals else float("nan")
        n    = len(vals)
        print(f"  {acc:>13.1%} ({n:3d})", end="")
    print()

# ── Bar chart ──────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, len(CONDS), figsize=(13, 4), sharey=True)
colors = {"control": "steelblue", "entity-only": "tomato",
          "country-only": "orange", "both": "seagreen"}

for ax, (cname, *_) in zip(axes, CONDS):
    inames = [i for i, *_ in INTERV_SETS]
    vals   = [np.mean(results[i][cname]) if results[i][cname] else float("nan")
              for i in inames]
    bars = ax.bar(inames, vals, color=[colors[i] for i in inames], alpha=0.85)
    for bar, v in zip(bars, vals):
        if not np.isnan(v):
            ax.text(bar.get_x() + bar.get_width()/2, v + 0.02,
                    f"{v:.0%}", ha="center", va="bottom", fontsize=9)
    ax.axhline(0.5, color="black", lw=0.8, ls="--", alpha=0.4)
    ax.set_ylim(0, 1.15)
    ax.set_title(cname, fontsize=11)
    ax.set_xticklabels(inames, rotation=20, ha="right", fontsize=8)
    if ax == axes[0]:
        ax.set_ylabel("Accuracy", fontsize=10)

fig.suptitle(
    f"Probe-direction SWAP  ({cfg['name']})  — layers {inj_range}, layer-diff\n"
    f"alpha_e={args.alpha_e}   alpha_c={args.alpha_c}\n"
    "entity/country-only should flip (→0%), both should restore (→control)",
    fontsize=10)
plt.tight_layout()
out = os.path.join(OUTDIR, f"probe_two_alpha_steer_rewrite_{args.model}.png")
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.close()
print(f"\nSaved {out}")
print("Done.")
