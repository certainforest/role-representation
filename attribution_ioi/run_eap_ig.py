"""
EAP-IG (Edge Attribution Patching with Integrated Gradients) for the IOI-style
speaker binding task.

Uses the same experiment definitions as run_ioi_style.py (from utils.py).
Finds the minimal subgraph of edges responsible for the binding-ID resolution:
  SOURCE (canonical slot assignment → correct answer)
  BASE   (swapped slot assignment   → wrong answer)

Runs BINDING experiments only by default (since these have a clean signal axis).
Use --include-control to also run CONTROL experiments.

Setups:
  --setup base : SWAP_EXPERIMENTS    (standard 2-speaker conversation)
  --setup hi   : HI_SWAP_EXPERIMENTS (Hi-greeting conversation)

Output: results/{model}/eap_ig_{setup}/

Usage:
  python run_eap_ig.py --model llama --gpu 1 --setup base
  python run_eap_ig.py --model qwen  --gpu 2 --setup base
  python run_eap_ig.py --model llama --gpu 1 --setup hi
"""
import os, sys, re, argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model",  choices=["llama", "qwen"], default="llama")
parser.add_argument("--gpu",    default="0")
parser.add_argument("--setup",  choices=["base", "hi"],    default="base")
parser.add_argument("--topn",   type=int, default=200,
                    help="Number of top edges to keep in the circuit")
parser.add_argument("--ig-steps", type=int, default=10,
                    help="Integrated gradient steps")
parser.add_argument("--bf16", action="store_true",
                    help="Use bfloat16 instead of float32 (saves memory, slightly less accurate)")
parser.add_argument("--include-control", action="store_true",
                    help="Also include CONTROL experiments in the pool")
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

import torch
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from transformer_lens import HookedTransformer
from eap.graph import Graph
from eap.attribute import attribute
from eap.evaluate import evaluate_graph, evaluate_baseline

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import SWAP_EXPERIMENTS, HI_SWAP_EXPERIMENTS, MODEL_CONFIGS

# ── Select experiment pool ────────────────────────────────────────────────────
_pool = SWAP_EXPERIMENTS if args.setup == "base" else HI_SWAP_EXPERIMENTS
if args.include_control:
    EXPERIMENTS = _pool
else:
    EXPERIMENTS = [e for e in _pool if e["kind"] == "BINDING"]

print(f"Setup={args.setup}  include_control={args.include_control}")
print(f"Running {len(EXPERIMENTS)} experiments:")
for e in EXPERIMENTS:
    print(f"  [{e['kind']:7s}] {e['key']}  "
          f"src={e['source_answer']!r} base={e['base_answer']!r}")

OUTDIR = os.path.join(
    "/mnt/ssd/aryawu/role-representation/attribution_ioi/results",
    args.model, f"eap_ig_{args.setup}"
)
os.makedirs(OUTDIR, exist_ok=True)

# ── Load model ────────────────────────────────────────────────────────────────
cfg = MODEL_CONFIGS[args.model]
print(f"\nLoading {cfg['name']}...")
model = HookedTransformer.from_pretrained(
    cfg["name"], dtype=torch.bfloat16 if args.bf16 else torch.float32,
    cache_dir=cfg["cache"],
)
# Required EAP-IG config flags
model.cfg.use_attn_result                = True
model.cfg.use_split_qkv_input           = True
model.cfg.use_hook_mlp_in               = True
model.cfg.ungroup_grouped_query_attention = True  # GQA (Llama-3.1-8B, Qwen3-8B)
model.eval().to("cuda")
NL, NH = model.cfg.n_layers, model.cfg.n_heads
print(f"Loaded: {NL}L {NH}H")

# ── Build dataset: validate predictions + collect token IDs ───────────────────
print("\nValidating model predictions on experiments...")
raw_data = []
for e in EXPERIMENTS:
    src_id  = model.tokenizer.encode(e["source_answer"], add_special_tokens=False)[0]
    base_id = model.tokenizer.encode(e["base_answer"],   add_special_tokens=False)[0]

    src_tok  = model.to_tokens(e["source_prompt"], prepend_bos=True)
    base_tok = model.to_tokens(e["base_prompt"],   prepend_bos=True)
    with torch.no_grad():
        src_logits  = model(src_tok)
        base_logits = model(base_tok)

    src_pred  = model.tokenizer.decode(src_logits[0, -1].argmax().item())
    base_pred = model.tokenizer.decode(base_logits[0, -1].argmax().item())
    src_ok  = src_logits[0, -1].argmax().item()  == src_id
    base_ok = base_logits[0, -1].argmax().item() == base_id
    print(f"  {e['key']:25s} [{e['kind']:7s}]  "
          f"src={src_pred!r} {'✓' if src_ok else '✗'}  "
          f"base={base_pred!r} {'✓' if base_ok else '✗'}")
    if not src_ok:
        print(f"    WARNING: source prediction wrong — skipping {e['key']}")
        continue
    if not base_ok:
        print(f"    WARNING: base prediction wrong — skipping {e['key']}")
        continue

    # EAP-IG: clean=SOURCE (correct binding), corrupted=BASE (swapped binding)
    raw_data.append((
        e["source_prompt"],   # clean
        e["base_prompt"],     # corrupted
        {"target": src_id, "wrong": base_id},
    ))

assert raw_data, "No valid experiments — cannot run EAP-IG"
print(f"\n{len(raw_data)} experiments will be used for attribution.")

# ── DataLoader ────────────────────────────────────────────────────────────────
from torch.utils.data import DataLoader, Dataset

class BindingDataset(Dataset):
    def __init__(self, data): self.data = data
    def __len__(self): return len(self.data)
    def __getitem__(self, idx): return self.data[idx]

def collate_fn(batch):
    return (
        [b[0] for b in batch],
        [b[1] for b in batch],
        {"target": [b[2]["target"] for b in batch],
         "wrong":  [b[2]["wrong"]  for b in batch]},
    )

dataloader = DataLoader(BindingDataset(raw_data), batch_size=1,
                        shuffle=False, collate_fn=collate_fn)

# ── Metric: negative logit-diff (EAP-IG minimizes, so this finds edges that
#    maximise logit(src_answer) - logit(base_answer)) ─────────────────────────
def metric(logits, clean_logits, input_lengths, label):
    last  = input_lengths - 1
    bidx  = torch.arange(logits.shape[0], device=logits.device)
    tgt   = torch.tensor(label["target"], device=logits.device)
    wrong = torch.tensor(label["wrong"],  device=logits.device)
    ld = logits[bidx, last, tgt] - logits[bidx, last, wrong]
    return -ld.mean()

# ── Build graph and run EAP-IG ────────────────────────────────────────────────
print(f"\nBuilding computation graph...")
graph = Graph.from_model(model)
print(f"Graph: {len(graph.nodes)} nodes, {len(graph.edges)} edges")

print(f"\nRunning EAP-IG (ig_steps={args.ig_steps}, {len(raw_data)} exps)...")
attribute(
    model=model, graph=graph, dataloader=dataloader, metric=metric,
    method="EAP-IG-inputs", ig_steps=args.ig_steps,
)
print("Attribution complete.")

# ── Extract and rank edges ────────────────────────────────────────────────────
edge_scores = [
    (name, float(edge.score), abs(float(edge.score)))
    for name, edge in graph.edges.items()
    if edge.score is not None
]
edge_scores.sort(key=lambda x: -x[2])

print(f"\nTop 30 edges by |score|:")
for name, score, abs_score in edge_scores[:30]:
    print(f"  {name:50s}  {score:+.4f}")

with open(f"{OUTDIR}/edge_scores.txt", "w") as f:
    f.write(f"EAP-IG Edge Scores  model={args.model}  setup={args.setup}  "
            f"topn={args.topn}  ig_steps={args.ig_steps}  n_exps={len(raw_data)}\n")
    f.write(f"{'Edge':50s}  {'Score':>10s}  {'|Score|':>10s}\n")
    f.write("-" * 75 + "\n")
    for name, score, abs_score in edge_scores:
        f.write(f"{name:50s}  {score:>10.4f}  {abs_score:>10.4f}\n")
print(f"  Saved edge_scores.txt ({len(edge_scores)} edges)")

# ── Apply top-N pruning + evaluate faithfulness ───────────────────────────────
print(f"\nApplying top-{args.topn} pruning...")
graph.apply_topn(args.topn, absolute=True, prune=False)
print(f"Active edges: {int(graph.in_graph.sum().item())}")

print("\nEvaluating faithfulness...")
baseline_score = evaluate_baseline(model=model, dataloader=dataloader, metrics=metric)
circuit_score  = evaluate_graph(model=model, graph=graph,
                                 dataloader=dataloader, metrics=metric)
b = float(baseline_score.mean()) if hasattr(baseline_score, 'mean') else float(baseline_score)
c = float(circuit_score.mean())  if hasattr(circuit_score,  'mean') else float(circuit_score)
faithfulness = c / b if b != 0 else float("nan")
print(f"Baseline: {b:.4f}  Circuit: {c:.4f}  Faithfulness: {faithfulness:.1%}")

# ── Parse edges into per-head / per-MLP score matrices ───────────────────────
def parse_node(s):
    s = s.split("<")[0].strip()
    if s in ("embed", "input", "logits"): return (s, None, None)
    m = re.match(r"a(\d+)\.h(\d+)$", s)
    if m: return ("attn", int(m.group(1)), int(m.group(2)))
    m = re.match(r"m(\d+)$", s)
    if m: return ("mlp", int(m.group(1)), None)
    return None

attn_scores = np.zeros((NL, NH))
mlp_scores  = np.zeros(NL)

for name, score, abs_score in edge_scores:
    parts = name.split("->")
    if len(parts) != 2: continue
    for node in [parse_node(p.strip()) for p in parts]:
        if node is None: continue
        if node[0] == "attn" and node[1] < NL and node[2] < NH:
            attn_scores[node[1], node[2]] = max(attn_scores[node[1], node[2]], abs_score)
        elif node[0] == "mlp" and node[1] < NL:
            mlp_scores[node[1]] = max(mlp_scores[node[1]], abs_score)

# ── Plot 1: attention head heatmap ─────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(NH * 0.45, NL * 0.28 + 0.8))
vmax = max(attn_scores.max(), 0.01)
im = ax.imshow(attn_scores, aspect="auto", cmap="YlOrRd", origin="lower",
               vmin=0, vmax=vmax)
plt.colorbar(im, ax=ax, label="Max |EAP-IG score|")
ax.set_xlabel("Head"); ax.set_ylabel("Layer")
ax.set_title(f"EAP-IG: Attention Head Importance\n"
             f"model={args.model}  setup={args.setup}  "
             f"topn={args.topn}  {len(raw_data)} exps  faithfulness={faithfulness:.0%}")
ax.set_xticks(range(NH)); ax.set_yticks(range(NL))
# Annotate top 10 heads
flat = attn_scores.flatten()
for idx in np.argsort(flat)[-10:]:
    l, h = divmod(idx, NH)
    ax.add_patch(plt.Rectangle((h-0.5, l-0.5), 1, 1,
                                fill=False, edgecolor="blue", lw=1.5))
    ax.text(h, l, f"{flat[idx]:.2f}", ha="center", va="center",
            fontsize=5, color="white" if flat[idx] > vmax * 0.7 else "black")
plt.tight_layout()
plt.savefig(f"{OUTDIR}/attn_head_scores.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved attn_head_scores.png")

# ── Plot 2: MLP layer scores ───────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(max(8, NL * 0.4), 3.5))
ax.bar(range(NL), mlp_scores, color="steelblue")
ax.set_xticks(range(NL))
ax.set_xticklabels([f"L{l}" for l in range(NL)], rotation=45, fontsize=7)
ax.set_ylabel("Max |EAP-IG score|")
ax.set_title(f"EAP-IG: MLP Layer Importance  [{args.model}  {args.setup}]")
ax.grid(True, alpha=0.3, axis="y")
plt.tight_layout()
plt.savefig(f"{OUTDIR}/mlp_scores.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved mlp_scores.png")

# ── Plot 3: highlight our target binding heads on the heatmap ─────────────────
TARGET_HEADS = {
    "qwen":  {"base": [(21,18),(21,19),(23,25),(23,26)],
               "hi":   [(21,18),(22,10),(23,30),(26,25)]},
    "llama": {"base": [(13,18),(14,7),(15,8),(15,11),(19,20)],
               "hi":   [(13,13),(14,4),(14,6),(14,7),(14,22),(15,11)]},
}
targets = TARGET_HEADS[args.model][args.setup]

fig, ax = plt.subplots(figsize=(NH * 0.45, NL * 0.28 + 0.8))
im = ax.imshow(attn_scores, aspect="auto", cmap="YlOrRd", origin="lower",
               vmin=0, vmax=vmax)
plt.colorbar(im, ax=ax, label="Max |EAP-IG score|")
ax.set_xlabel("Head"); ax.set_ylabel("Layer")
ax.set_title(f"EAP-IG + Target Binding Heads (green=binding-head, blue=top-10 EAP)\n"
             f"model={args.model}  setup={args.setup}  faithfulness={faithfulness:.0%}")
ax.set_xticks(range(NH)); ax.set_yticks(range(NL))
# Blue: top-10 EAP-IG
for idx in np.argsort(flat)[-10:]:
    l, h = divmod(idx, NH)
    ax.add_patch(plt.Rectangle((h-0.5, l-0.5), 1, 1,
                                fill=False, edgecolor="blue", lw=1.5))
    ax.text(h, l, f"{flat[idx]:.2f}", ha="center", va="center",
            fontsize=5, color="white" if flat[idx] > vmax * 0.7 else "black")
# Green: our identified target binding heads
for l, h in targets:
    ax.add_patch(plt.Rectangle((h-0.5, l-0.5), 1, 1,
                                fill=False, edgecolor="lime", lw=2.5))
plt.tight_layout()
plt.savefig(f"{OUTDIR}/attn_head_scores_with_targets.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved attn_head_scores_with_targets.png")

# ── Summary ────────────────────────────────────────────────────────────────────
print(f"\n{'='*65}")
print(f"EAP-IG SUMMARY  model={args.model}  setup={args.setup}  topn={args.topn}")
print(f"{'='*65}")
print(f"Faithfulness: {faithfulness:.1%}")
print(f"\nTop 10 attention heads by max |score|:")
for idx in np.argsort(flat)[-10:][::-1]:
    l, h = divmod(idx, NH)
    tag = " ← TARGET" if (l, h) in targets else ""
    print(f"  L{l:2d}H{h:2d}: {flat[idx]:.4f}{tag}")

print(f"\nEAP-IG score of each target binding head:")
for l, h in targets:
    print(f"  L{l:2d}H{h:2d}: {attn_scores[l, h]:.4f}  "
          f"(rank {int(np.sum(flat > attn_scores[l,h])) + 1} / {NL*NH})")

print(f"\nTop 5 MLP layers:")
for idx in np.argsort(mlp_scores)[-5:][::-1]:
    print(f"  L{idx:2d}: {mlp_scores[idx]:.4f}")

print(f"\nAll outputs → {OUTDIR}/")
