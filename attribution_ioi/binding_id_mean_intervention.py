"""
Replication of Feng & Steinhardt (2024) Table 1.

Matches the paper's actual implementation:
  - extract_k_means: delta = mean(slot-1 in default - slot-0 in shifted)
    uses context-only prompts (no query), entity_width=2, country_width=1
  - test_mean_interventions: purely ADDITIVE injection, no context freeze
    slot-0 += delta[layer], slot-1 -= delta[layer]
  - layer-diff injection so cumulative effect = delta[l] at layer l
  - train/eval split to avoid overlap

Usage:
  python binding_id_mean_intervention.py --model llama --gpu 0
  python binding_id_mean_intervention.py --model llama --gpu 0 --n-samples 500
"""
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model",     choices=["llama","llamabase", "qwen"], default="llama")
parser.add_argument("--gpu",       default="0")
parser.add_argument("--n-samples", type=int, default=500)
parser.add_argument("--n-eval",    type=int, default=100)
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from transformer_lens import HookedTransformer
from utils import MODEL_CONFIGS

cfg = MODEL_CONFIGS[args.model]

OUTDIR = (f"/mnt/ssd/aryawu/role-representation/attribution"
          f"/circuit_results_{args.model}/_mean_intervention")
os.makedirs(OUTDIR, exist_ok=True)

_NAMES_RAW = [
    "Alice", "Bob", "Charlie", "Diana", "Emma", "James",
    "Olivia", "Liam", "Sophia", "Noah", "Ava", "Ethan",
    "Isabella", "Mason", "Mia", "William", "Charlotte", "Benjamin",
    "Amelia", "Lucas", "Harper", "Henry", "Evelyn", "Alexander",
    "Oliver", "Aria", "Sebastian", "Luna", "Jack", "Chloe",
    "Owen", "Penelope", "Aiden", "Layla", "Nathan", "Grace",
]
_COUNTRIES_RAW = [
    "France", "Thailand", "Germany", "Spain", "Italy", "Brazil",
    "Japan", "Canada", "Mexico", "Australia", "Sweden", "Norway",
    "Egypt", "India", "China", "Argentina", "Portugal", "Greece",
    "Poland", "Turkey", "Denmark", "Finland", "Austria", "Belgium",
    "Netherlands", "Chile", "Vietnam", "Morocco", "Ukraine", "Colombia",
    "Romania", "Pakistan", "Kenya", "Peru", "Hungary", "Croatia",
]

# ── Load model ────────────────────────────────────────────────────────────────
print(f"\nLoading {cfg['name']}...")
model = HookedTransformer.from_pretrained(
    cfg["name"], dtype=torch.bfloat16, cache_dir=cfg["cache"])
model.eval().to("cuda")
NL = model.cfg.n_layers
D  = model.cfg.d_model
print(f"Loaded: {NL}L d={D}")

# ── Filter to single-token words ──────────────────────────────────────────────
def is_single_token(word):
    return len(model.tokenizer.encode(f" {word}", add_special_tokens=False)) == 1

NAMES     = [n for n in _NAMES_RAW     if is_single_token(n)]
COUNTRIES = [c for c in _COUNTRIES_RAW if is_single_token(c)]
print(f"Single-token names: {len(NAMES)}  countries: {len(COUNTRIES)}")

ALL_NAME_PAIRS    = [(i,j) for i in range(len(NAMES))
                           for j in range(len(NAMES))     if i != j]
ALL_COUNTRY_PAIRS = [(i,j) for i in range(len(COUNTRIES))
                            for j in range(len(COUNTRIES)) if i != j]

MAX_SAMPLES = min(len(ALL_NAME_PAIRS), len(ALL_COUNTRY_PAIRS))

# Train/eval split — strictly disjoint
rng_split = np.random.default_rng(0)
all_idx   = rng_split.permutation(MAX_SAMPLES)
N_TRAIN   = min(args.n_samples, int(0.8 * MAX_SAMPLES))
TRAIN_IDX = all_idx[:N_TRAIN]
EVAL_IDX  = all_idx[N_TRAIN:]
N_SAMPLES = N_TRAIN
print(f"Train: {len(TRAIN_IDX)}  Eval pool: {len(EVAL_IDX)}")


# ══════════════════════════════════════════════════════════════════════════════
# PROMPT TEMPLATES
# ══════════════════════════════════════════════════════════════════════════════
def make_context_only(e0, e1, a0, a1):
    """No query — used for extract_k_means only."""
    return (
        'This is the transcript of a conversation.\n'
        f'"I am {e0}."\n"I am {e1}."\n'
        f'"I live in {a0}."\n"I live in {a1}."'
    )

def make_prompt_query_e0(e0, e1, a0, a1):
    """Query slot-0 entity → correct answer is a0."""
    return (
        'This is the transcript of a conversation.\n'
        f'"I am {e0}."\n"I am {e1}."\n'
        f'"I live in {a0}."\n"I live in {a1}."\n'
        f'Question: Where does {e0} live? Answer:'
    ), a0

def make_prompt_query_e1(e0, e1, a0, a1):
    """Query slot-1 entity → correct answer is a1."""
    return (
        'This is the transcript of a conversation.\n'
        f'"I am {e0}."\n"I am {e1}."\n'
        f'"I live in {a0}."\n"I live in {a1}."\n'
        f'Question: Where does {e1} live? Answer:'
    ), a1

def make_prompt_query_a0(e0, e1, a0, a1):
    """Query slot-0 country → correct answer is e0."""
    return (
        'This is the transcript of a conversation.\n'
        f'"I am {e0}."\n"I am {e1}."\n'
        f'"I live in {a0}."\n"I live in {a1}."\n'
        f'Question: Who lives in {a0}? Answer:'
    ), e0

def make_prompt_query_a1(e0, e1, a0, a1):
    """Query slot-1 country → correct answer is e1."""
    return (
        'This is the transcript of a conversation.\n'
        f'"I am {e0}."\n"I am {e1}."\n'
        f'"I live in {a0}."\n"I live in {a1}."\n'
        f'Question: Who lives in {a1}? Answer:'
    ), e1


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def get_token_pos(tok_ids, target):
    tid = model.tokenizer.encode(f" {target}", add_special_tokens=False)[0]
    for i, t in enumerate(tok_ids):
        if t == tid:
            return i
    raise ValueError(f"{target!r} not found")

def top1(logits, ans_id, foil_id):
    return int(logits[0, -1, ans_id].item() > logits[0, -1, foil_id].item())


# ══════════════════════════════════════════════════════════════════════════════
# extract_k_means
# entity_width=2: name token + following token, averaged
# country_width=1: country token only
# ══════════════════════════════════════════════════════════════════════════════
def extract_k_means(n_samples, label=""):
    print(f"\nextract_k_means [{label}]  n={n_samples}")

    rng         = np.random.default_rng(42)
    name_idx    = TRAIN_IDX[rng.permutation(len(TRAIN_IDX))[:n_samples]]
    country_idx = TRAIN_IDX[rng.permutation(len(TRAIN_IDX))[:n_samples]]

    dE_acc, sE_acc = [], []
    dA_acc, sA_acc = [], []

    for i in range(n_samples):
        e0 = NAMES[ALL_NAME_PAIRS[name_idx[i]][0]]
        e1 = NAMES[ALL_NAME_PAIRS[name_idx[i]][1]]
        a0 = COUNTRIES[ALL_COUNTRY_PAIRS[country_idx[i]][0]]
        a1 = COUNTRIES[ALL_COUNTRY_PAIRS[country_idx[i]][1]]

        prompt_default = make_context_only(e0, e1, a0, a1)
        prompt_shifted = make_context_only(e1, e0, a1, a0)

        tok_d = model.to_tokens(prompt_default, prepend_bos=True)[0].tolist()
        tok_s = model.to_tokens(prompt_shifted, prepend_bos=True)[0].tolist()

        try:
            pos_e1_d = get_token_pos(tok_d, e1)
            pos_a1_d = get_token_pos(tok_d, a1)
            pos_e1_s = get_token_pos(tok_s, e1)
            pos_a1_s = get_token_pos(tok_s, a1)
        except ValueError:
            continue

        inp_d = model.to_tokens(prompt_default, prepend_bos=True)
        inp_s = model.to_tokens(prompt_shifted, prepend_bos=True)

        with torch.no_grad():
            _, cache_d = model.run_with_cache(
                inp_d, names_filter=lambda n: "resid_pre" in n)
            _, cache_s = model.run_with_cache(
                inp_s, names_filter=lambda n: "resid_pre" in n)

        # entity width=2: name token + following token, averaged -> [NL, d]
        e1_default = torch.stack([
            cache_d[f"blocks.{l}.hook_resid_pre"][0, pos_e1_d:pos_e1_d+2].float().cpu().mean(0)
            for l in range(NL)])
        e1_shifted = torch.stack([
            cache_s[f"blocks.{l}.hook_resid_pre"][0, pos_e1_s:pos_e1_s+2].float().cpu().mean(0)
            for l in range(NL)])

        # country width=1: country token only -> [NL, d]
        a1_default = torch.stack([
            cache_d[f"blocks.{l}.hook_resid_pre"][0, pos_a1_d:pos_a1_d+2].float().cpu().mean(0)
            for l in range(NL)])
        a1_shifted = torch.stack([
            cache_s[f"blocks.{l}.hook_resid_pre"][0, pos_a1_s:pos_a1_s+2].float().cpu().mean(0)
            for l in range(NL)])

        dE_acc.append(e1_default.numpy())
        sE_acc.append(e1_shifted.numpy())
        dA_acc.append(a1_default.numpy())
        sA_acc.append(a1_shifted.numpy())

        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{n_samples}  ({len(dE_acc)} valid)")

    delta_E = np.mean(np.array(dE_acc) - np.array(sE_acc), axis=0)  # [NL, d]
    delta_A = np.mean(np.array(dA_acc) - np.array(sA_acc), axis=0)

    print(f"  Valid: {len(dE_acc)}/{n_samples}"
          f"  |∆E|={np.linalg.norm(delta_E):.4f}"
          f"  |∆A|={np.linalg.norm(delta_A):.4f}")

    return {
        "names":     torch.tensor(delta_E, dtype=torch.float32),  # [NL, d]
        "countries": torch.tensor(delta_A, dtype=torch.float32),
    }


# ══════════════════════════════════════════════════════════════════════════════
# run_intervened_model — purely additive, layer-diff injection
# ══════════════════════════════════════════════════════════════════════════════
def layer_diff(delta):
    """Convert absolute [NL, d] delta to layer-to-layer differences.
    Cumulative effect at layer l = delta[l]."""
    diff = torch.zeros_like(delta)
    diff[0] = delta[0]
    diff[1:] = delta[1:] - delta[:-1]
    return diff

def run_intervened_model(tok, pos_e0, pos_e1, pos_a0, pos_a1, delta_E, delta_A):
    """
    entity width=2: inject at pos and pos+1
    country width=1: inject at pos only
    """
    dE_diff = layer_diff(delta_E) if delta_E is not None else None
    dA_diff = layer_diff(delta_A) if delta_A is not None else None

    all_hooks = []
    for layer in range(NL):
        def make_hook(l, _dE, _dA, _pe0, _pe1, _pa0, _pa1):
            def hook_fn(resid, hook):
                if _dE is not None:
                    d = _dE[l].to(resid.device)
                    resid[0, _pe0,   :] += d
                    resid[0, _pe0+1, :] += d
                    resid[0, _pe1,   :] -= d
                    resid[0, _pe1+1, :] -= d
                if _dA is not None:
                    d = _dA[l].to(resid.device)
                    resid[0, _pa0, :] += d
                    resid[0, _pa0+1, :] += d
                    resid[0, _pa1, :] -= d
                    resid[0, _pa1+1, :] -= d
                return resid
            return hook_fn
        all_hooks.append((
            f"blocks.{layer}.hook_resid_pre",
            make_hook(layer, dE_diff, dA_diff, pos_e0, pos_e1, pos_a0, pos_a1)
        ))
    with torch.no_grad():
        logits = model.run_with_hooks(tok, fwd_hooks=all_hooks, return_type="logits")
    return logits


# ══════════════════════════════════════════════════════════════════════════════
# test_mean_interventions
# ══════════════════════════════════════════════════════════════════════════════
def test_mean_interventions(means, n_eval, prompt_fn, label=""):
    print(f"\ntest_mean_interventions [{label}]  n={n_eval}")

    dE = means["names"]      # [NL, d]
    dA = means["countries"]  # [NL, d]

    rE = torch.randn_like(dE)
    rA = torch.randn_like(dA)
    for l in range(NL):
        rE[l] = rE[l] / rE[l].norm() * dE[l].norm()
        rA[l] = rA[l] / rA[l].norm() * dA[l].norm()

    conditions = {
        "control":   (None, None),
        "attribute": (None, dA),
        "entity":    (dE,   None),
        "both":      (dE,   dA),
        "random":    (rE,   rA),
    }
    results = {c: [] for c in conditions}

    rng         = np.random.default_rng(999)
    n_eval      = min(n_eval, len(EVAL_IDX))
    perm        = rng.permutation(len(EVAL_IDX))
    name_idx    = EVAL_IDX[perm[:n_eval]]
    country_idx = EVAL_IDX[perm[:n_eval]]

    for i in range(n_eval):
        e0 = NAMES[ALL_NAME_PAIRS[name_idx[i]][0]]
        e1 = NAMES[ALL_NAME_PAIRS[name_idx[i]][1]]
        a0 = COUNTRIES[ALL_COUNTRY_PAIRS[country_idx[i]][0]]
        a1 = COUNTRIES[ALL_COUNTRY_PAIRS[country_idx[i]][1]]

        prompt, ans = prompt_fn(e0, e1, a0, a1)
        tok     = model.to_tokens(prompt, prepend_bos=True)
        tok_ids = tok[0].tolist()

        try:
            pos_E0 = get_token_pos(tok_ids, e0)
            pos_E1 = get_token_pos(tok_ids, e1)
            pos_A0 = get_token_pos(tok_ids, a0)
            pos_A1 = get_token_pos(tok_ids, a1)
        except ValueError:
            continue

        ans_id = model.tokenizer.encode(f" {ans}", add_special_tokens=False)[0]
        if ans == a0:   foil = a1
        elif ans == a1: foil = a0
        elif ans == e0: foil = e1
        else:           foil = e0
        foil_id = model.tokenizer.encode(f" {foil}", add_special_tokens=False)[0]

        if i == 0:
            tok_strs = [model.tokenizer.decode(t) for t in tok_ids]
            print(f"\n  [debug i=0] prompt: {prompt!r}")
            print(f"  pos_E0={pos_E0} → {tok_strs[pos_E0]!r}")
            print(f"  pos_E1={pos_E1} → {tok_strs[pos_E1]!r}")
            print(f"  pos_A0={pos_A0} → {tok_strs[pos_A0]!r}")
            print(f"  pos_A1={pos_A1} → {tok_strs[pos_A1]!r}")
            print(f"  pos_E0+1 → {tok_strs[pos_E0+1]!r}")
            print(f"  pos_E1+1 → {tok_strs[pos_E1+1]!r}")
            print(f"  pos_A0+1 → {tok_strs[pos_A0+1]!r}")
            print(f"  pos_A1+1 → {tok_strs[pos_A1+1]!r}")
            print(f"  ans={ans!r}  foil={foil!r}")
            with torch.no_grad():
                logits_clean = model(tok)
            gap_clean = logits_clean[0, -1, ans_id].item() - logits_clean[0, -1, foil_id].item()
            logits_attr = run_intervened_model(tok, pos_E0, pos_E1, pos_A0, pos_A1, None, dA)
            gap_attr  = logits_attr[0, -1, ans_id].item()  - logits_attr[0, -1, foil_id].item()
            logits_ent  = run_intervened_model(tok, pos_E0, pos_E1, pos_A0, pos_A1, dE, None)
            gap_ent   = logits_ent[0, -1, ans_id].item()   - logits_ent[0, -1, foil_id].item()
            logits_both = run_intervened_model(tok, pos_E0, pos_E1, pos_A0, pos_A1, dE, dA)
            gap_both  = logits_both[0, -1, ans_id].item()  - logits_both[0, -1, foil_id].item()
            print(f"  logit gap (ans-foil): clean={gap_clean:.3f}  attr={gap_attr:.3f}"
                  f"  entity={gap_ent:.3f}  both={gap_both:.3f}")

        for cond, (_dE, _dA) in conditions.items():
            if _dE is None and _dA is None:
                with torch.no_grad():
                    logits = model(tok)
            else:
                logits = run_intervened_model(
                    tok, pos_E0, pos_E1, pos_A0, pos_A1, _dE, _dA)
            results[cond].append(top1(logits, ans_id, foil_id))

        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{n_eval}")

    accs = {c: np.mean(v) for c, v in results.items() if v}
    print(f"\n  {'Condition':12s}  {'Acc':>7s}  {'N':>5s}")
    print(f"  {'-'*26}")
    for c in ["control", "attribute", "entity", "both", "random"]:
        if c in accs:
            print(f"  {c:12s}  {accs[c]:7.1%}  {len(results[c]):5d}")
    return accs


# ══════════════════════════════════════════════════════════════════════════════
# PLOT
# ══════════════════════════════════════════════════════════════════════════════
def plot_bars(accs, fname):
    conds = ["control", "attribute", "entity", "both", "random"]
    vals  = [accs.get(c, float('nan')) for c in conds]
    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(conds, vals, alpha=0.85)
    for bar, v in zip(bars, vals):
        if not np.isnan(v):
            ax.text(bar.get_x() + bar.get_width()/2, v + 0.02,
                    f"{v:.0%}", ha='center', va='bottom', fontsize=9)
    ax.axhline(0.5, color='black', lw=0.8, linestyle='--', alpha=0.5, label='chance')
    ax.set_ylabel("Accuracy"); ax.set_ylim(0, 1.15)
    ax.set_title(f"Mean Intervention [{args.model}]  n_samples={N_SAMPLES}\n"
                 f"Expected: control≈99%, attr≈0%, entity≈0%, both≈97%, random≈100%")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved {os.path.basename(fname)}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*65}")
print(f"Replicating Table 1  [{args.model}]")
print(f"n_samples={N_SAMPLES}  n_eval={args.n_eval}")
print(f"{'='*65}")

means = extract_k_means(N_SAMPLES, "standard")

print("\nPer-layer delta norms:")
for l in range(0, NL, 4):
    print(f"  L{l:02d}  |∆E|={means['names'][l].norm():.3f}"
          f"  |∆A|={means['countries'][l].norm():.3f}")

accs_e0 = test_mean_interventions(means, args.n_eval, make_prompt_query_e0, "query-e0")
accs_e1 = test_mean_interventions(means, args.n_eval, make_prompt_query_e1, "query-e1")
accs_a0 = test_mean_interventions(means, args.n_eval, make_prompt_query_a0, "query-a0")
accs_a1 = test_mean_interventions(means, args.n_eval, make_prompt_query_a1, "query-a1")

plot_bars(accs_e0, f"{OUTDIR}/table1_query_e0.png")
plot_bars(accs_e1, f"{OUTDIR}/table1_query_e1.png")
plot_bars(accs_a0, f"{OUTDIR}/table1_query_a0.png")
plot_bars(accs_a1, f"{OUTDIR}/table1_query_a1.png")

print(f"\n{'='*65}")
print(f"TABLE 1 REPLICATION  [{args.model}]")
print(f"{'='*65}")
hdr = f"  {'':16s}" + "".join(
    f"  {c:>9s}" for c in ["control", "attribute", "entity", "both", "random"])
print(hdr); print("  " + "-"*64)
for lbl, accs in [("querying E0", accs_e0), ("querying E1", accs_e1),
                  ("querying A0", accs_a0), ("querying A1", accs_a1)]:
    row = "".join(f"  {accs.get(c, float('nan')):9.1%}"
                  for c in ["control", "attribute", "entity", "both", "random"])
    print(f"  {lbl:16s}{row}")

np.save(f"{OUTDIR}/delta_E.npy", means["names"].numpy())
np.save(f"{OUTDIR}/delta_A.npy", means["countries"].numpy())
print(f"\nAll outputs -> {OUTDIR}/")