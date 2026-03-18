"""
Post-hoc head voting from a saved F_raw_top_heads.json.

Loads the JSON written incrementally by run_ioi_style.py and produces
voting plots without needing to re-run the model.

Supports filtering by --type and --kind so you can compare e.g. Swap-only
vs. all types, or BINDING-only vs. CONTROL-only.

Output saved next to the JSON file (or --outdir).

Usage:
  python run_ioi_posthoc.py results/qwen/F_raw_top_heads.json
  python run_ioi_posthoc.py results/qwen/F_raw_top_heads.json --K 5
  python run_ioi_posthoc.py results/qwen/F_raw_top_heads.json --type Swap
  python run_ioi_posthoc.py results/qwen/F_raw_top_heads.json --kind BINDING --type Null
"""
import os
import json
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("json", help="Path to F_raw_top_heads.json")
parser.add_argument("--K",      type=int, default=10,
                    help="Top-K heads per experiment (default: 10)")
parser.add_argument("--type",   nargs='+', default=None,
                    help="Filter by exact type(s), e.g. --type Swap Null  or  --type reverse_Swap reverse_Null")
parser.add_argument("--kind",   default=None, choices=["BINDING", "CONTROL", None],
                    help="Filter by kind (BINDING or CONTROL)")
parser.add_argument("--outdir", default=None,
                    help="Output directory (default: same folder as JSON)")
args = parser.parse_args()

with open(args.json) as f:
    raw = json.load(f)

# ── Filter ────────────────────────────────────────────────────────────────────
experiments = raw["experiments"]
if args.type:
    experiments = [e for e in experiments if e["type"] in args.type]
if args.kind:
    experiments = [e for e in experiments if e["kind"] == args.kind]

if not experiments:
    raise ValueError(f"No experiments match type={args.type!r} kind={args.kind!r}")

n_layers = raw["n_layers"]
n_heads  = raw["n_heads"]
n_exps   = len(experiments)
K        = args.K

outdir = args.outdir or os.path.dirname(os.path.abspath(args.json))
os.makedirs(outdir, exist_ok=True)

filter_tag = ""
if args.type: filter_tag += "_" + "+".join(args.type)
if args.kind: filter_tag += f"_{args.kind}"

print(f"Post-hoc voting: model={raw['model']}  K={K}  "
      f"n_exps={n_exps}  filter={filter_tag or 'none'}")
for e in experiments:
    print(f"  {e['key']:25s} [{e['kind']:7s}] type={e['type']}")


def top_heads(flat, K):
    flat = np.array(flat)
    top = []
    for idx in np.argsort(flat)[-K:][::-1]:
        top.append((int(idx // n_heads), int(idx % n_heads)))
    return top


# ── D. Head voting: all filtered experiments ──────────────────────────────────
head_vote = {}
for e in experiments:
    for lh in top_heads(e["per_head_patch_flat"], K):
        head_vote[lh] = head_vote.get(lh, 0) + 1

vote_map = np.zeros((n_layers, n_heads))
for (l, h), cnt in head_vote.items():
    vote_map[l, h] = cnt

fig, ax = plt.subplots(figsize=(n_heads * 0.45, n_layers * 0.3))
im = ax.imshow(vote_map, aspect='auto', cmap='YlOrRd', origin='lower', vmin=0, vmax=n_exps)
plt.colorbar(im, ax=ax, label=f"# experiments in top-{K} (out of {n_exps})")
ax.set_xlabel("Head"); ax.set_ylabel("Layer")
ax.set_title(f"Head Voting: {n_exps} experiments (top-{K}){filter_tag}")
ax.set_xticks(range(n_heads)); ax.set_yticks(range(n_layers))
for (l, h), cnt in head_vote.items():
    if cnt >= max(2, n_exps // 2):
        ax.text(h, l, str(cnt), ha='center', va='center', fontsize=7,
                color='white' if cnt >= n_exps * 0.75 else 'black', fontweight='bold')
plt.tight_layout()
out = os.path.join(outdir, f"D_head_voting_all_K{K}{filter_tag}.png")
plt.savefig(out, dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved {out}")

# ── E. BINDING vs CONTROL head voting ─────────────────────────────────────────
binding_exps = [e for e in experiments if e["kind"] == "BINDING"]
control_exps = [e for e in experiments if e["kind"] == "CONTROL"]

if binding_exps and control_exps:
    binding_set, control_set = set(), set()
    fig, axes = plt.subplots(1, 2, figsize=(n_heads * 0.9, n_layers * 0.3))
    for ax, kind_exps, kind_label, kind_set in zip(
        axes,
        [binding_exps, control_exps],
        [f"BINDING ({len(binding_exps)} exps)", f"CONTROL ({len(control_exps)} exps)"],
        [binding_set, control_set],
    ):
        vote_m = np.zeros((n_layers, n_heads))
        for e in kind_exps:
            for l, h in top_heads(e["per_head_patch_flat"], K):
                vote_m[l, h] += 1
                kind_set.add((l, h))
        im = ax.imshow(vote_m, aspect='auto', cmap='YlOrRd', origin='lower',
                       vmin=0, vmax=len(kind_exps))
        plt.colorbar(im, ax=ax, label=f"# exps in top-{K}")
        ax.set_xlabel("Head"); ax.set_ylabel("Layer"); ax.set_title(kind_label)
        ax.set_xticks(range(n_heads)); ax.set_yticks(range(n_layers))
        for l in range(n_layers):
            for h in range(n_heads):
                if vote_m[l, h] >= len(kind_exps) * 0.5:
                    ax.text(h, l, str(int(vote_m[l, h])), ha='center', va='center',
                            fontsize=7, color='white', fontweight='bold')
    fig.suptitle(f"Head Voting: BINDING vs CONTROL (top-{K}){filter_tag}", fontsize=11)
    plt.tight_layout()
    out = os.path.join(outdir, f"E_head_voting_binding_vs_control_K{K}{filter_tag}.png")
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved {out}")

    binding_only = binding_set - control_set
    control_only = control_set - binding_set
    print(f"\nBinding-only (top-{K}): {sorted(binding_only)}")
    print(f"Control-only (top-{K}): {sorted(control_only)}")
    print(f"Shared:                 {sorted(binding_set & control_set)}")
else:
    print("  Skipping E (need both BINDING and CONTROL experiments)")
