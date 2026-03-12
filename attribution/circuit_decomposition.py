"""
IOI-style incremental circuit decomposition for speaker binding.
Correct backwards order: logits → output heads → middle → early → root.

Path patching follows IOI paper (appendix B) exactly:
  Step A: run on BASE, freeze ALL z to BASE except sender → patch to SOURCE.
          Cache receiver Q/K/V under this intervention.
  Step B: run on BASE normally, ONLY patch receiver Q/K/V to Step A value.
          No z-freezing in Step B — receiver and all downstream layers run freely.

Supports Llama-3.1-8B and Qwen3-8B.
"""
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model", choices=["llama", "qwen"], default="llama",
                    help="Which model to run: llama (Llama-3.1-8B) or qwen (Qwen3-8B)")
parser.add_argument("--gpu", default="3", help="CUDA_VISIBLE_DEVICES (default: 3)")
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm
from transformer_lens import HookedTransformer, utils

MODEL_CONFIGS = {
    "llama": {
        "name":  "meta-llama/Llama-3.1-8B-Instruct",
        "cache": "/mnt/ssd/aryawu/.cache/huggingface/hub",
    },
    "qwen": {
        "name":  "Qwen/Qwen3-8B",
        "cache": "/mnt/ssd/aryawu/.cache/huggingface/hub",
    },
}

MODEL_KEY = args.model

# ══════════════════════════════════════════════════════════════
# EXPERIMENT CONFIG
# ══════════════════════════════════════════════════════════════
# Known binding-ID heads per model (from 8-experiment voting + inspect_heads).
# Run run_speaker_circuit.py --model qwen first to discover Qwen's binding heads.
BINDING_HEADS_BY_MODEL = {
    "llama": [(13, 18), (15, 8)],   # from Llama-3.1-8B-Instruct voting
    "qwen":  [(23, 26)],            # L23H26: top-1 all 4 binding, absent all 4 controls (Δ=0.415)
                                    # L21H18 dropped: Δ=0.108, entity-swap-specific only
}
BINDING_HEADS = BINDING_HEADS_BY_MODEL[MODEL_KEY]

BINDING_EXPS_4 = [
    {
        "key": "ex1_1_1", "label": "Ex1 Entity/Entity",
        "source_prompt": (
            'This is the transcript of a conversation.\n'
            '"I am Alice."\n"I am Bob."\n'
            '"I live in France."\n"I live in Thailand."\n'
            'Question: Where does Alice live? Answer:'
        ),
        "source_answer": " France",
        "base_prompt": (
            'This is the transcript of a conversation.\n'
            '"I am Bob."\n"I am Alice."\n'
            '"I live in France."\n"I live in Thailand."\n'
            'Question: Where does Alice live? Answer:'
        ),
        "base_answer": " Thailand",
    },
    {
        "key": "ex1_1_2", "label": "Ex1 Attr/Attr",
        "source_prompt": (
            'This is the transcript of a conversation.\n'
            '"I am Alice."\n"I am Bob."\n'
            '"I live in France."\n"I live in Thailand."\n'
            'Question: Who lives in France? Answer:'
        ),
        "source_answer": " Alice",
        "base_prompt": (
            'This is the transcript of a conversation.\n'
            '"I am Alice."\n"I am Bob."\n'
            '"I live in Thailand."\n"I live in France."\n'
            'Question: Who lives in France? Answer:'
        ),
        "base_answer": " Bob",
    },
    {
        "key": "ex2_1_1", "label": "Ex2 Entity/Entity",
        "source_prompt": (
            'This is the transcript of a conversation.\n'
            '"I am Alice."\n"I am Bob."\n'
            '"I like basketball."\n"I like soccer."\n'
            'Question: What sport does Alice like? Answer: Alice likes'
        ),
        "source_answer": " basketball",
        "base_prompt": (
            'This is the transcript of a conversation.\n'
            '"I am Bob."\n"I am Alice."\n'
            '"I like basketball."\n"I like soccer."\n'
            'Question: What sport does Alice like? Answer: Alice likes'
        ),
        "base_answer": " soccer",
    },
    {
        "key": "ex2_1_2", "label": "Ex2 Attr/Attr",
        "source_prompt": (
            'This is the transcript of a conversation.\n'
            '"I am Alice."\n"I am Bob."\n'
            '"I like basketball."\n"I like soccer."\n'
            'Question: Who likes basketball? Answer:'
        ),
        "source_answer": " Alice",
        "base_prompt": (
            'This is the transcript of a conversation.\n'
            '"I am Alice."\n"I am Bob."\n'
            '"I like soccer."\n"I like basketball."\n'
            'Question: Who likes basketball? Answer:'
        ),
        "base_answer": " Bob",
    },
]

EXPERIMENTS = [BINDING_EXPS_4[0], BINDING_EXPS_4[2]]   # ex1_1_1, ex2_1_1 for phases 1-4

# ══════════════════════════════════════════════════════════════
# LOAD MODEL
# ══════════════════════════════════════════════════════════════
cfg = MODEL_CONFIGS[MODEL_KEY]
OUTDIR = (f"/mnt/ssd/aryawu/role-representation/attribution"
          f"/circuit_results/_decomp_{MODEL_KEY}")
os.makedirs(OUTDIR, exist_ok=True)

print(f"\nLoading {cfg['name']}...")
model = HookedTransformer.from_pretrained(
    cfg["name"], dtype=torch.bfloat16, cache_dir=cfg["cache"])
model.eval().to("cuda")

NL  = model.cfg.n_layers
NH  = model.cfg.n_heads
NKV = model.cfg.n_key_value_heads
GQA = NH // NKV
ALL_Z = [utils.get_act_name("z", l) for l in range(NL)]
print(f"{NL}L {NH}H {NKV}KV d={model.cfg.d_model}  GQA={GQA}x")


# ══════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════
def setup(exp):
    src_id  = model.tokenizer.encode(exp["source_answer"], add_special_tokens=False)[0]
    base_id = model.tokenizer.encode(exp["base_answer"],   add_special_tokens=False)[0]
    src_tok  = model.to_tokens(exp["source_prompt"], prepend_bos=True)
    base_tok = model.to_tokens(exp["base_prompt"],   prepend_bos=True)
    assert src_tok.shape == base_tok.shape, "Prompt length mismatch"

    with torch.no_grad():
        r_src  = model(src_tok)
        r_base = model(base_tok)
    assert r_src[0, -1].argmax().item()  == src_id, \
        f"SOURCE wrong: got {model.tokenizer.decode(r_src[0,-1].argmax().item())!r}"
    assert r_base[0, -1].argmax().item() == base_id, \
        f"BASE wrong: got {model.tokenizer.decode(r_base[0,-1].argmax().item())!r}"

    ld_src  = (r_src[0, -1, src_id]  - r_src[0, -1, base_id]).item()
    ld_base = (r_base[0, -1, src_id] - r_base[0, -1, base_id]).item()

    def metric(logits):
        ld = logits[0, -1, src_id] - logits[0, -1, base_id]
        return (ld - ld_base) / (ld_src - ld_base)

    with torch.no_grad():
        _, src_cache  = model.run_with_cache(src_tok)
        _, base_cache = model.run_with_cache(base_tok)

    return dict(src_id=src_id, base_id=base_id,
                src_tok=src_tok, base_tok=base_tok,
                src_cache=src_cache, base_cache=base_cache,
                metric=metric,
                tok_labels=[repr(model.tokenizer.decode(t))
                            for t in src_tok[0].tolist()])


def patch_recv_qkv(act, hook, recv_h, recv_input, patched_cache):
    """GQA-aware: patch receiver head's Q or K or V from patched_cache."""
    if recv_input in ("k", "v"):
        kv_h = recv_h // GQA
        act[:, :, kv_h] = patched_cache[hook.name][:, :, kv_h]
    else:
        act[:, :, recv_h] = patched_cache[hook.name][:, :, recv_h]
    return act


def plot_heatmap(data, xlabel, ylabel, title, fname,
                 cmap="RdBu_r", vmin=None, vmax=None,
                 highlight_heads=None, xtick_step=4, ytick_step=2):
    rows, cols = data.shape
    fig, ax = plt.subplots(figsize=(cols * 0.38, rows * 0.28 + 0.5))
    vmax_ = vmax if vmax is not None else max(float(np.abs(data).max()), 0.05)
    vmin_ = vmin if vmin is not None else -vmax_
    im = ax.imshow(data, aspect="auto", cmap=cmap, origin="lower", vmin=vmin_, vmax=vmax_)
    plt.colorbar(im, ax=ax, label="Metric (1=SOURCE, 0=BASE)")
    if highlight_heads:
        for hl, hh in highlight_heads:
            if hl < rows and hh < cols:
                ax.add_patch(plt.Rectangle(
                    (hh - 0.5, hl - 0.5), 1, 1, fill=False, edgecolor="lime", lw=2))
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.set_title(title, fontsize=9)
    ax.set_xticks(range(0, cols, xtick_step))
    ax.set_yticks(range(0, rows, ytick_step))
    plt.tight_layout()
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()


# ══════════════════════════════════════════════════════════════
# PHASE 0: Q/K/V/Pattern/Z patching for known binding heads
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("PHASE 0: Q/K/V/Pattern/Z patching — binding heads")
print("=" * 60)

def patch_component(comp, layer, head, d):
    hook_name = utils.get_act_name(comp, layer)
    def hook_fn(act, hook):
        if comp == "pattern":
            act[:, head] = d["src_cache"][hook.name][:, head]
        elif comp in ("k", "v"):
            kv_h = head // GQA
            act[:, :, kv_h] = d["src_cache"][hook.name][:, :, kv_h]
        else:
            act[:, :, head] = d["src_cache"][hook.name][:, :, head]
        return act
    with torch.no_grad():
        logits = model.run_with_hooks(
            d["base_tok"], fwd_hooks=[(hook_name, hook_fn)], return_type="logits")
    return d["metric"](logits).item()

components = ["q", "k", "v", "pattern", "z"]
phase0 = {(l, h): {e["key"]: {} for e in BINDING_EXPS_4} for l, h in BINDING_HEADS}

for exp in BINDING_EXPS_4:
    print(f"\n  {exp['key']} ({exp['label']})")
    d = setup(exp)
    for l, h in BINDING_HEADS:
        for comp in components:
            val = patch_component(comp, l, h, d)
            phase0[(l, h)][exp["key"]][comp] = val
            print(f"    L{l}H{h} {comp:7s}: {val:.4f}")

fig, axes = plt.subplots(1, len(BINDING_HEADS), figsize=(8 * len(BINDING_HEADS), 5))
axes = [axes] if len(BINDING_HEADS) == 1 else axes
colors = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3"]
for ax, (l, h) in zip(axes, BINDING_HEADS):
    x = np.arange(len(components))
    w = 0.18
    for i, exp in enumerate(BINDING_EXPS_4):
        vals = [phase0[(l, h)][exp["key"]][c] for c in components]
        ax.bar(x + (i - 1.5) * w, vals, w, label=exp["label"],
               color=colors[i], alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels([c.upper() for c in components])
    ax.set_ylim(-0.15, 1.05); ax.axhline(0, color="black", lw=0.5)
    ax.set_ylabel("Patch metric"); ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_title(f"L{l}H{h} — which component carries the binding signal?")
plt.suptitle(f"Phase 0: Q/K/V/Pattern/Z patching  [{MODEL_KEY}]", fontsize=12)
plt.tight_layout()
plt.savefig(f"{OUTDIR}/phase0_qkv.png", dpi=150, bbox_inches="tight")
plt.close()
print("\n  Saved phase0_qkv.png")


# ══════════════════════════════════════════════════════════════
# PHASE 1: Direct logit writers
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("PHASE 1: Direct logit writers")
print("=" * 60)

def run_phase1(d, desc=""):
    """
    For each head (l, h): freeze ALL head z to BASE, patch (l, h) to SOURCE.
    Measures which heads have a direct causal path to the output logits.
    """
    results = torch.zeros(NL, NH)
    for layer in tqdm(range(NL), desc=f"Phase1 {desc}"):
        for head in range(NH):
            hooks = [
                (name,
                 lambda z, hook, _l=layer, _h=head: (
                     z.__setitem__(Ellipsis, d["base_cache"][hook.name]) or
                     z.__setitem__((slice(None), slice(None), _h),
                                   d["src_cache"][hook.name][:, :, _h])
                     if hook.layer() == _l else
                     z.__setitem__(Ellipsis, d["base_cache"][hook.name])
                 ) or z)
                for name in ALL_Z
            ]
            with torch.no_grad():
                logits = model.run_with_hooks(
                    d["base_tok"], fwd_hooks=hooks, return_type="logits")
            results[layer, head] = d["metric"](logits).item()
    return results

phase1_results = {}
for exp in EXPERIMENTS:
    print(f"\n  {exp['key']}...")
    d = setup(exp)
    r = run_phase1(d, exp["key"])
    phase1_results[exp["key"]] = (r, d)

    plot_heatmap(
        r.numpy(), xlabel="Head", ylabel="Layer",
        title=f"Phase 1: Direct Logit Writers [{MODEL_KEY}] — {exp['label']}\n(green = binding heads)",
        fname=f"{OUTDIR}/phase1_{exp['key']}.png",
        vmin=0, vmax=1, highlight_heads=BINDING_HEADS,
    )

    flat = r.numpy().flatten()
    print(f"  Top 10 direct logit writers:")
    for idx in np.argsort(flat)[-10:][::-1]:
        lx, hx = divmod(idx, NH)
        tag = " ← BINDING HEAD" if (lx, hx) in BINDING_HEADS else ""
        print(f"    L{lx}H{hx}: {flat[idx]:.4f}{tag}")

output_heads = set()
for exp in EXPERIMENTS:
    r, _ = phase1_results[exp["key"]]
    flat = r.numpy().flatten()
    for idx in np.argsort(flat)[-8:][::-1]:
        lx, hx = divmod(idx, NH)
        output_heads.add((lx, hx))
output_heads = sorted(output_heads)
print(f"\n  Output heads (union top-8): {output_heads}")


# ══════════════════════════════════════════════════════════════
# PATH PATCHING CORE — correct IOI implementation
# ══════════════════════════════════════════════════════════════
def path_patch_senders_to_receiver(recv_layer, recv_head, recv_input, d, desc=""):
    """
    For each sender (sl < recv_layer, sh):

      Step A [IOI Step 2]:
        Run on BASE tokens.
        Freeze ALL head z to BASE except sender (sl, sh) → patch to SOURCE.
        Cache receiver's Q/K/V under this isolated intervention.

      Step B [IOI Step 3]:
        Run on BASE tokens.
        ONLY patch receiver Q/K/V to Step A cache value.
        No z-freezing — receiver and all downstream layers run freely.
        Measure metric.

    This exactly matches get_path_patch_head_to_heads() from the IOI notebook.
    """
    recv_input_hook = utils.get_act_name(recv_input, recv_layer)
    results = torch.zeros(recv_layer, NH)

    for sl in tqdm(range(recv_layer),
                   desc=f"Path→L{recv_layer}H{recv_head}.{recv_input} {desc}"):
        for sh in range(NH):

            # ── Step A: freeze all z to BASE, patch sender (sl,sh) to SOURCE ──
            def hook_step_A(z, hook, _sl=sl, _sh=sh):
                z[...] = d["base_cache"][hook.name][...]          # freeze to BASE
                if hook.layer() == _sl:
                    z[:, :, _sh, :] = d["src_cache"][hook.name][:, :, _sh, :]  # patch sender
                return z

            hooks_A = [(name, hook_step_A) for name in ALL_Z]
            with torch.no_grad(), model.hooks(fwd_hooks=hooks_A):
                _, patched_cache = model.run_with_cache(
                    d["base_tok"],
                    names_filter=lambda n: n == recv_input_hook,
                )

            # ── Step B: run BASE normally, only patch receiver Q/K/V ──
            # NO z-freezing — exactly matching IOI Step 3
            def hook_step_B(act, hook,
                            _rh=recv_head, _ri=recv_input, _pc=patched_cache):
                return patch_recv_qkv(act, hook, _rh, _ri, _pc)

            with torch.no_grad():
                logits = model.run_with_hooks(
                    d["base_tok"],
                    fwd_hooks=[(recv_input_hook, hook_step_B)],
                    return_type="logits",
                )
            results[sl, sh] = d["metric"](logits).item()

    return results


def run_phase(phase_name, receiver_heads, d, highlight_heads=None,
              metric_threshold=0.10, top_k=8):
    """
    Run one full phase: scan all senders → every (receiver, Q/K/V).
    Track max score per sender. Return top senders as next receivers.
    """
    print(f"\n{'='*60}")
    print(f"{phase_name}: {len(receiver_heads)} receivers = {sorted(receiver_heads)}")
    print(f"{'='*60}")

    sender_scores = {}   # (sl, sh) → max score across all receivers and inputs

    for recv_layer, recv_head in sorted(receiver_heads):
        if recv_layer == 0:
            continue
        for recv_input in ["q", "k", "v"]:
            print(f"\n  → L{recv_layer}H{recv_head} {recv_input.upper()}")

            r = path_patch_senders_to_receiver(
                recv_layer, recv_head, recv_input, d, desc=phase_name)

            flat = r.numpy().flatten()
            top5 = np.argsort(flat)[-5:][::-1]
            print("    Top senders: " + ", ".join(
                f"L{sl}H{sh}:{flat[i]:.3f}"
                for i in top5
                for sl, sh in [divmod(i, NH)]
            ))

            plot_heatmap(
                flat.reshape(recv_layer, NH),
                xlabel="Sender Head", ylabel="Sender Layer",
                title=(f"Senders → {recv_input.upper()} of "
                       f"L{recv_layer}H{recv_head}  [{MODEL_KEY}]\n{phase_name}"),
                fname=f"{OUTDIR}/{phase_name}_L{recv_layer}H{recv_head}_{recv_input}.png",
                highlight_heads=highlight_heads,
            )

            for i, v in enumerate(flat):
                sl, sh = divmod(i, NH)
                sender_scores[(sl, sh)] = max(sender_scores.get((sl, sh), 0.0), float(v))

    ranked = sorted(sender_scores.items(), key=lambda x: -x[1])
    next_receivers = {lh for lh, v in ranked[:top_k] if v >= metric_threshold}

    print(f"\n  {phase_name} → next receivers (top-{top_k}, thresh={metric_threshold}):")
    for lh, v in ranked[:top_k]:
        tag = " ✓" if v >= metric_threshold else ""
        print(f"    L{lh[0]}H{lh[1]}: {v:.4f}{tag}")

    return next_receivers


# ══════════════════════════════════════════════════════════════
# RUN PHASES 2 → N  (loop until signal dies)
# ══════════════════════════════════════════════════════════════
d_primary = setup(EXPERIMENTS[0])   # ex1_1_1

# Phase 2 uses a higher threshold (output heads are strong writers)
# subsequent phases use a lower threshold as signal attenuates
all_phases = {"Phase1 (output)": set(output_heads)}
current_receivers = set(output_heads)
visited_receivers = set(output_heads)   # all heads ever used as receivers
phase_num = 2
threshold = 0.15

while True:
    phase_name = f"Phase{phase_num}"
    next_receivers = run_phase(
        phase_name, receiver_heads=current_receivers,
        d=d_primary, highlight_heads=BINDING_HEADS,
        metric_threshold=threshold, top_k=8,
    )
    all_phases[phase_name] = next_receivers
    threshold = 0.10

    if not next_receivers:
        print(f"\n  Signal died at {phase_name} — circuit complete.")
        break

    # Remove any heads already processed as receivers (cycle detection)
    new_receivers = next_receivers - visited_receivers
    if not new_receivers:
        print(f"\n  All next receivers already visited — cycle detected, stopping.")
        break

    if phase_num >= 10:
        print(f"\n  Reached phase limit (10) — stopping.")
        break

    visited_receivers |= new_receivers
    current_receivers = new_receivers   # only recurse into genuinely new heads
    phase_num += 1

# ══════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print(f"CIRCUIT SUMMARY  [{MODEL_KEY}]")
print(f"{'='*60}")
for phase_name, heads in all_phases.items():
    print(f"  {phase_name}: {sorted(heads)}")

print(f"\n  Where do known binding heads appear?")
for bl, bh in BINDING_HEADS:
    stages = [name for name, heads in all_phases.items() if (bl, bh) in heads]
    print(f"    L{bl}H{bh}: {stages if stages else 'not found in any phase'}")

print(f"\n  All outputs → {OUTDIR}/")
