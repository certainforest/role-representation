"""
Compute binding-only head voting summary across all experiment groups.

For each model, computes 4 groups:
  1. Swap+Null forward
  2. Swap+Null reverse
  3. hi_swap+hi_null forward
  4. hi_swap+hi_null reverse

For each group: top-K heads per experiment → vote → binding-only vs control-only vs shared.
Then computes intersections: forward∩reverse for each condition, and across all 4 groups.

Output: results/{model}/head_voting_summary.txt
"""
import os
import json
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--K", type=int, default=5)
parser.add_argument("--models", nargs="+", default=["qwen", "llama"])
parser.add_argument("--base-dir", default="/mnt/ssd/aryawu/role-representation/attribution_ioi/results")
args = parser.parse_args()

GROUPS = [
    ("Swap+Null",       ["Swap",          "Null"         ]),
    ("Swap+Null rev",   ["reverse_Swap",  "reverse_Null" ]),
    ("hi forward",      ["hi_swap",       "hi_null"      ]),
    ("hi reverse",      ["reverse_hi_swap","reverse_hi_null"]),
]


def top_heads(flat, K, n_heads):
    flat = np.array(flat)
    heads = set()
    for idx in np.argsort(flat)[-K:][::-1]:
        heads.add((int(idx // n_heads), int(idx % n_heads)))
    return heads


def vote(experiments, K, n_layers, n_heads):
    binding_vote = {}
    control_vote = {}
    for e in experiments:
        for lh in top_heads(e["per_head_patch_flat"], K, n_heads):
            if e["kind"] == "BINDING":
                binding_vote[lh] = binding_vote.get(lh, 0) + 1
            else:
                control_vote[lh] = control_vote.get(lh, 0) + 1
    binding_heads = set(binding_vote)
    control_heads = set(control_vote)
    binding_only = binding_heads - control_heads
    control_only = control_heads - binding_heads
    shared       = binding_heads & control_heads
    return binding_only, control_only, shared, binding_vote, control_vote


for model in args.models:
    json_path = os.path.join(args.base_dir, model, "F_raw_top_heads.json")
    with open(json_path) as f:
        raw = json.load(f)

    n_layers = raw["n_layers"]
    n_heads  = raw["n_heads"]
    all_exps = raw["experiments"]

    out_path = os.path.join(args.base_dir, model, "head_voting_summary.txt")
    lines = []

    def p(s=""):
        lines.append(s)
        print(s)

    p(f"{'='*70}")
    p(f"MODEL: {model}  K={args.K}  n_layers={n_layers}  n_heads={n_heads}")
    p(f"{'='*70}")

    group_binding_only = {}

    for group_name, types in GROUPS:
        exps = [e for e in all_exps if e["type"] in types]
        if not exps:
            p(f"\n[{group_name}] NO DATA")
            continue

        binding_only, control_only, shared, bv, cv = vote(exps, args.K, n_layers, n_heads)
        group_binding_only[group_name] = binding_only

        p(f"\n── {group_name} ({len(exps)} exps, types={types}) ──")
        p(f"  Binding-only : {sorted(binding_only)}")
        p(f"  Control-only : {sorted(control_only)}")
        p(f"  Shared       : {sorted(shared)}")

    # ── Pairwise intersections ──────────────────────────────────────────────
    p(f"\n{'─'*70}")
    p("INTERSECTIONS")
    p(f"{'─'*70}")

    # Forward ∩ Reverse for each condition
    for fwd, rev in [("Swap+Null", "Swap+Null rev"), ("hi forward", "hi reverse")]:
        if fwd in group_binding_only and rev in group_binding_only:
            inter = group_binding_only[fwd] & group_binding_only[rev]
            p(f"\n  {fwd} ∩ {rev} (binding-only):")
            p(f"    {sorted(inter)}")

    # Across all 4 groups
    all_sets = [s for s in group_binding_only.values() if s]
    if len(all_sets) == 4:
        inter_all = all_sets[0]
        for s in all_sets[1:]:
            inter_all = inter_all & s
        p(f"\n  ALL 4 groups ∩ (binding-only):")
        p(f"    {sorted(inter_all)}")

    # Across forward groups only
    fwd_sets = [group_binding_only.get(g) for g in ["Swap+Null", "hi forward"] if g in group_binding_only]
    if len(fwd_sets) == 2:
        inter_fwd = fwd_sets[0] & fwd_sets[1]
        p(f"\n  Forward only (Swap+Null ∩ hi) binding-only:")
        p(f"    {sorted(inter_fwd)}")

    # Across reverse groups only
    rev_sets = [group_binding_only.get(g) for g in ["Swap+Null rev", "hi reverse"] if g in group_binding_only]
    if len(rev_sets) == 2:
        inter_rev = rev_sets[0] & rev_sets[1]
        p(f"\n  Reverse only (Swap+Null ∩ hi) binding-only:")
        p(f"    {sorted(inter_rev)}")

    p()
    with open(out_path, "w") as f:
        f.write("\n".join(lines))
    print(f"\nSaved → {out_path}")
