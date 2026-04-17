"""
circuit_decomposition_golden.py

Full IOI-style circuit decomposition for ex1_1_1 on Llama-3.1-8B-Instruct,
followed by component patching (Q/K/V/Z/Pattern) for every head in the
discovered circuit.

Discovery method (backwards from logits):
  Phase 1  — z-patch each (layer, head) in isolation on BASE → finds direct
             logit writers (output heads).
  Phase 2+ — for each output head, IOI path-patch all senders → its Q/K/V →
             finds what feeds each head; recurse until signal dies.

After discovery:
  Component patch — patch Q / K / V / Pattern / Z individually for each
                    circuit head from SOURCE onto BASE → reveals which input
                    carries the binding signal at each node.

Usage:
  python circuit_decomposition_golden.py --gpu 0
"""
import os, argparse

parser = argparse.ArgumentParser()
parser.add_argument("--gpu", default="0")
parser.add_argument("--top-k",   type=int, default=8,
                    help="Senders per phase to carry forward (default: 8)")
parser.add_argument("--thresh1", type=float, default=0.15,
                    help="Phase-2 threshold (default: 0.15)")
parser.add_argument("--thresh",  type=float, default=0.10,
                    help="Phase-3+ threshold (default: 0.10)")
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm
from transformer_lens import HookedTransformer, utils

# ── Model ─────────────────────────────────────────────────────────────────────
MODEL_NAME  = "meta-llama/Llama-3.1-8B-Instruct"
MODEL_CACHE = "/mnt/ssd/aryawu/.cache/huggingface/hub"
OUTDIR      = ("/mnt/ssd/aryawu/role-representation/attribution"
               "/circuit_results/golden_llama")
os.makedirs(OUTDIR, exist_ok=True)

print(f"Loading {MODEL_NAME}...")
model = HookedTransformer.from_pretrained(MODEL_NAME, dtype=torch.bfloat16,
                                          cache_dir=MODEL_CACHE)
model.eval().to("cuda")

NL  = model.cfg.n_layers
NH  = model.cfg.n_heads
NKV = model.cfg.n_key_value_heads
GQA = NH // NKV
ALL_Z = [utils.get_act_name("z", l) for l in range(NL)]
print(f"{NL}L {NH}H  GQA={GQA}x  d={model.cfg.d_model}")


# ══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT ex1_1_1
# SOURCE: Alice first → Alice lives in France
# BASE:   Bob first  → Alice lives in Thailand
# ══════════════════════════════════════════════════════════════════════════════
SRC_PROMPT = (
    'This is the transcript of a conversation.\n'
    '"I am Alice."\n"I am Bob."\n'
    '"I live in France."\n"I live in Thailand."\n'
    'Question: Where does Alice live? Answer:'
)
BASE_PROMPT = (
    'This is the transcript of a conversation.\n'
    '"I am Bob."\n"I am Alice."\n'
    '"I live in France."\n"I live in Thailand."\n'
    'Question: Where does Alice live? Answer:'
)
SRC_ANS  = " France"
BASE_ANS = " Thailand"

src_id  = model.tokenizer.encode(SRC_ANS,  add_special_tokens=False)[0]
base_id = model.tokenizer.encode(BASE_ANS, add_special_tokens=False)[0]
src_tok  = model.to_tokens(SRC_PROMPT,  prepend_bos=True)
base_tok = model.to_tokens(BASE_PROMPT, prepend_bos=True)
assert src_tok.shape == base_tok.shape

with torch.no_grad():
    r_src  = model(src_tok)
    r_base = model(base_tok)
assert r_src[0, -1].argmax().item()  == src_id,  "SOURCE prediction wrong"
assert r_base[0, -1].argmax().item() == base_id, "BASE prediction wrong"

ld_src  = (r_src[0, -1, src_id]  - r_src[0, -1, base_id]).item()
ld_base = (r_base[0, -1, src_id] - r_base[0, -1, base_id]).item()
print(f"\nSOURCE LD={ld_src:.3f}  BASE LD={ld_base:.3f}")

def metric(logits):
    ld = logits[0, -1, src_id] - logits[0, -1, base_id]
    return float((ld - ld_base) / (ld_src - ld_base))

with torch.no_grad():
    _, src_cache  = model.run_with_cache(src_tok)
    _, base_cache = model.run_with_cache(base_tok)

tok_labels = [repr(model.tokenizer.decode(t)) for t in src_tok[0].tolist()]


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def patch_recv_qkv(act, hook, recv_h, recv_input, patched_cache):
    if recv_input in ("k", "v"):
        kv_h = recv_h // GQA
        act[:, :, kv_h] = patched_cache[hook.name][:, :, kv_h]
    else:
        act[:, :, recv_h] = patched_cache[hook.name][:, :, recv_h]
    return act


def plot_heatmap(data, title, fname, vmin=None, vmax=None,
                 highlight_heads=None, xlabel="Head", ylabel="Layer"):
    rows, cols = data.shape
    fig, ax = plt.subplots(figsize=(cols * 0.38, rows * 0.28 + 0.5))
    vmax_ = vmax if vmax is not None else max(float(np.abs(data).max()), 0.05)
    vmin_ = vmin if vmin is not None else -vmax_
    im = ax.imshow(data, aspect="auto", cmap="RdBu_r", origin="lower",
                   vmin=vmin_, vmax=vmax_)
    plt.colorbar(im, ax=ax, label="Metric (1=SOURCE, 0=BASE)")
    if highlight_heads:
        for hl, hh in highlight_heads:
            if hl < rows and hh < cols:
                ax.add_patch(plt.Rectangle(
                    (hh - 0.5, hl - 0.5), 1, 1, fill=False, edgecolor="lime", lw=2))
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=9)
    ax.set_xticks(range(0, cols, 4)); ax.set_yticks(range(0, rows, 2))
    plt.tight_layout()
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 1 — direct logit writers via z-patching
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("PHASE 1: direct logit writers (z-patch)")
print("="*60)

phase1 = np.zeros((NL, NH))
for layer in tqdm(range(NL), desc="Phase1"):
    for head in range(NH):
        hooks = [
            (name,
             lambda z, hook, _l=layer, _h=head: (
                 z.__setitem__(Ellipsis, base_cache[hook.name]) or
                 z.__setitem__((slice(None), slice(None), _h),
                               src_cache[hook.name][:, :, _h])
                 if hook.layer() == _l else
                 z.__setitem__(Ellipsis, base_cache[hook.name])
             ) or z)
            for name in ALL_Z
        ]
        with torch.no_grad():
            logits = model.run_with_hooks(base_tok, fwd_hooks=hooks, return_type="logits")
        phase1[layer, head] = metric(logits)

plot_heatmap(phase1, title="Phase 1: Direct Logit Writers (ex1_1_1)",
             fname=f"{OUTDIR}/phase1.png", vmin=0, vmax=1)

flat1 = phase1.flatten()
output_heads = set()
print("\n  Top 10 direct logit writers:")
for idx in np.argsort(flat1)[-10:][::-1]:
    lx, hx = divmod(idx, NH)
    output_heads.add((lx, hx))
    print(f"    L{lx:2d}H{hx:2d}: {flat1[idx]:.4f}")
# keep top-8 for recursion
output_heads = {lh for lh, _ in
                sorted(((divmod(i, NH), flat1[i]) for i in np.argsort(flat1)[-args.top_k:][::-1]),
                       key=lambda x: -x[1])}
print(f"\n  Output heads (top-{args.top_k}): {sorted(output_heads)}")


# ══════════════════════════════════════════════════════════════════════════════
# IOI PATH PATCHING CORE
# ══════════════════════════════════════════════════════════════════════════════
def path_patch_senders_to_receiver(recv_layer, recv_head, recv_input):
    """
    IOI path patching (Appendix B):
      Step A: freeze ALL z to BASE, patch sender (sl,sh) to SOURCE,
              cache receiver Q/K/V.
      Step B: run BASE normally, only patch receiver Q/K/V to Step A value,
              measure metric.
    Returns float array (recv_layer, NH).
    """
    recv_hook = utils.get_act_name(recv_input, recv_layer)
    results   = np.zeros((recv_layer, NH))

    for sl in tqdm(range(recv_layer),
                   desc=f"→ L{recv_layer}H{recv_head}.{recv_input.upper()}"):
        for sh in range(NH):
            def hook_A(z, hook, _sl=sl, _sh=sh):
                z[...] = base_cache[hook.name]
                if hook.layer() == _sl:
                    z[:, :, _sh, :] = src_cache[hook.name][:, :, _sh, :]
                return z

            with torch.no_grad(), model.hooks(fwd_hooks=[(n, hook_A) for n in ALL_Z]):
                _, patched = model.run_with_cache(
                    base_tok, names_filter=lambda n: n == recv_hook)

            recv_val = patched[recv_hook].clone()

            def hook_B(act, hook, _rv=recv_val, _rh=recv_head, _ri=recv_input):
                return patch_recv_qkv(act, hook, _rh, _ri,
                                      {hook.name: _rv})

            with torch.no_grad():
                logits = model.run_with_hooks(base_tok,
                                              fwd_hooks=[(recv_hook, hook_B)],
                                              return_type="logits")
            results[sl, sh] = metric(logits)

    return results


# ══════════════════════════════════════════════════════════════════════════════
# PHASES 2 → N — backwards recursive search
# ══════════════════════════════════════════════════════════════════════════════
# all_edges: list of (phase, recv_layer, recv_head, recv_input, results_array)
#   so we can report any specific edge score later.
all_phases   = {"Phase1": set(output_heads)}
all_edges    = []              # (phase_name, rl, rh, ri, np.array)
circuit_heads = set(output_heads)  # every head that appears in any phase

current_receivers = set(output_heads)
visited_receivers = set(output_heads)
phase_num         = 2
threshold         = args.thresh1

while True:
    phase_name = f"Phase{phase_num}"
    print(f"\n{'='*60}")
    print(f"{phase_name}: receivers = {sorted(current_receivers)}")
    print(f"{'='*60}")

    sender_scores = {}  # (sl, sh) → max score across all (receiver, input) pairs

    for recv_layer, recv_head in sorted(current_receivers):
        if recv_layer == 0:
            continue
        for recv_input in ["q", "k", "v"]:
            print(f"\n  → L{recv_layer}H{recv_head} {recv_input.upper()}")
            r = path_patch_senders_to_receiver(recv_layer, recv_head, recv_input)
            all_edges.append((phase_name, recv_layer, recv_head, recv_input, r))

            flat = r.flatten()
            top5 = np.argsort(flat)[-5:][::-1]
            print("  Top senders: " + ", ".join(
                f"L{sl}H{sh}:{flat[i]:.3f}"
                for i in top5
                for sl, sh in [divmod(i, NH)]
            ))

            plot_heatmap(
                flat.reshape(recv_layer, NH),
                title=(f"{phase_name}: senders → {recv_input.upper()} "
                       f"of L{recv_layer}H{recv_head}"),
                fname=f"{OUTDIR}/{phase_name}_L{recv_layer}H{recv_head}_{recv_input}.png",
                vmin=0, vmax=1,
                highlight_heads=sorted(circuit_heads),
            )

            for i, v in enumerate(flat):
                sl, sh = divmod(i, NH)
                sender_scores[(sl, sh)] = max(sender_scores.get((sl, sh), 0.0), float(v))

    ranked        = sorted(sender_scores.items(), key=lambda x: -x[1])
    next_receivers = {lh for lh, v in ranked[:args.top_k] if v >= threshold}
    all_phases[phase_name] = next_receivers
    threshold = args.thresh

    print(f"\n  {phase_name} top-{args.top_k} senders (thresh={threshold}):")
    for lh, v in ranked[:args.top_k]:
        tag = " ✓" if v >= threshold else ""
        print(f"    L{lh[0]:2d}H{lh[1]:2d}: {v:.4f}{tag}")

    if not next_receivers:
        print(f"\n  Signal died — circuit complete.")
        break

    new_receivers = next_receivers - visited_receivers
    if not new_receivers:
        print(f"\n  All next receivers already visited — stopping.")
        break

    if phase_num >= 10:
        print(f"\n  Phase limit reached — stopping.")
        break

    circuit_heads   |= new_receivers
    visited_receivers |= new_receivers
    current_receivers  = new_receivers
    phase_num += 1


# ══════════════════════════════════════════════════════════════════════════════
# COMPONENT PATCHING for every circuit head
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("COMPONENT PATCHING for discovered circuit heads")
print("="*60)

COMPS = ["q", "k", "v", "pattern", "z"]

def patch_component(comp, layer, head):
    hook_name = utils.get_act_name(comp, layer)
    def hook_fn(act, hook):
        if comp == "pattern":
            act[:, head] = src_cache[hook.name][:, head]
        elif comp in ("k", "v"):
            act[:, :, head // GQA] = src_cache[hook.name][:, :, head // GQA]
        else:
            act[:, :, head] = src_cache[hook.name][:, :, head]
        return act
    with torch.no_grad():
        logits = model.run_with_hooks(base_tok, fwd_hooks=[(hook_name, hook_fn)],
                                      return_type="logits")
    return metric(logits)

circuit_sorted = sorted(circuit_heads)
comp_scores = {}
for l, h in tqdm(circuit_sorted, desc="Component patch"):
    comp_scores[(l, h)] = {c: patch_component(c, l, h) for c in COMPS}

print(f"\n  {'Head':10s}  " + "  ".join(f"{c:7s}" for c in COMPS))
for l, h in circuit_sorted:
    vals = "  ".join(f"{comp_scores[(l,h)][c]:7.4f}" for c in COMPS)
    print(f"  L{l:2d}H{h:2d}:    {vals}")


# ══════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print("CIRCUIT SUMMARY  —  ex1_1_1  Llama-3.1-8B-Instruct")
print(f"{'='*60}")

for phase_name, heads in all_phases.items():
    print(f"  {phase_name}: {sorted(heads)}")

print(f"\n  Phase-1 scores for circuit heads:")
for l, h in circuit_sorted:
    print(f"    L{l:2d}H{h:2d}: {phase1[l,h]:.4f}")

print(f"\n  Component patching summary:")
print(f"  {'Head':10s}  " + "  ".join(f"{c:7s}" for c in COMPS))
for l, h in circuit_sorted:
    vals = "  ".join(f"{comp_scores[(l,h)][c]:7.4f}" for c in COMPS)
    print(f"  L{l:2d}H{h:2d}:    {vals}")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE: component patching grid for all circuit heads
# ══════════════════════════════════════════════════════════════════════════════
n = len(circuit_sorted)
ncols = min(n, 4)
nrows = (n + ncols - 1) // ncols
fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 3))
axes = np.array(axes).reshape(-1)

for i, (l, h) in enumerate(circuit_sorted):
    ax = axes[i]
    vals = [comp_scores[(l, h)][c] for c in COMPS]
    ax.bar(COMPS, vals, color=["steelblue" if v >= 0 else "firebrick" for v in vals])
    ax.axhline(0, color="black", lw=0.5)
    ax.set_ylim(-0.15, 1.1)
    ax.set_title(f"L{l}H{h}  (Phase1={phase1[l,h]:.3f})", fontsize=9)
    ax.set_ylabel("Metric"); ax.grid(True, alpha=0.3, axis="y")

for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)

fig.suptitle("Component patching for circuit heads — ex1_1_1\n"
             "(which input carries the binding signal at each node?)", fontsize=11)
plt.tight_layout()
plt.savefig(f"{OUTDIR}/component_patch_grid.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"\nSaved component_patch_grid.png → {OUTDIR}/")
print("\nDone.")
