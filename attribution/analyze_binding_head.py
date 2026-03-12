"""
Analyze what feeds INTO the binding-ID head (L13H18 for Llama, L23H26 for Qwen).

For the binding head, we want to know:
  1. Which component carries the binding signal? (Q / K / V / Pattern / Z)
  2. Which earlier heads wrote the binding-relevant information into the
     positions that Q and K read from?

Approach:
  Phase 0: Patch Q/K/V/Pattern/Z of the binding head individually.
           Tells us which component carries the signal.

  Phase 1: Path patch every sender (L0H* ... L(bind-1)H*) -> binding_head.Q
           For each sender: freeze all z to BASE, patch sender to SOURCE,
           measure how much this changes the binding head's Q input.
           → finds what wrote binding-relevant info into the Q-read position.

  Phase 2: Same but for binding_head.K
           → finds what wrote binding-relevant info into the K-read positions.

Only runs ex1_1_1 (SOURCE: Alice first → France; BASE: Bob first → Thailand).

Usage:
  python analyze_binding_head.py --model llama --gpu 0
  python analyze_binding_head.py --model qwen  --gpu 1
"""
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model", choices=["llama", "qwen"], default="llama")
parser.add_argument("--gpu", default="0")
parser.add_argument("--skip-patching", action="store_true",
                    help="Skip path patching (phases 1&2), only run attn pattern + phase 0")
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
        "name":         "meta-llama/Llama-3.1-8B-Instruct",
        "cache":        "/mnt/ssd/aryawu/.cache/huggingface/hub",
        "binding_head": (13, 18),
    },
    "qwen": {
        "name":         "Qwen/Qwen3-8B",
        "cache":        "/mnt/ssd/brent/hf_home/hub",
        "binding_head": (23, 26),
    },
}

cfg          = MODEL_CONFIGS[args.model]
BIND_LAYER   = cfg["binding_head"][0]
BIND_HEAD    = cfg["binding_head"][1]
OUTDIR       = (f"/mnt/ssd/aryawu/role-representation/attribution"
                f"/circuit_results_{args.model}/_binding_head")
os.makedirs(OUTDIR, exist_ok=True)

EXP = {
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
}

# ── Load model ────────────────────────────────────────────────────────────────
print(f"Loading {cfg['name']}...")
model = HookedTransformer.from_pretrained(
    cfg["name"], dtype=torch.bfloat16, cache_dir=cfg["cache"])
model.eval().to("cuda")

NL  = model.cfg.n_layers
NH  = model.cfg.n_heads
NKV = model.cfg.n_key_value_heads
GQA = NH // NKV
print(f"Loaded: {NL}L {NH}H {NKV}KV  GQA={GQA}x  binding head: L{BIND_LAYER}H{BIND_HEAD}")

ALL_Z = [utils.get_act_name("z", l) for l in range(NL)]

# ── Setup: tokenize, run forward passes, build metric ────────────────────────
src_id  = model.tokenizer.encode(EXP["source_answer"], add_special_tokens=False)[0]
base_id = model.tokenizer.encode(EXP["base_answer"],   add_special_tokens=False)[0]

src_tok  = model.to_tokens(EXP["source_prompt"], prepend_bos=True)
base_tok = model.to_tokens(EXP["base_prompt"],   prepend_bos=True)
assert src_tok.shape == base_tok.shape

with torch.no_grad():
    r_src  = model(src_tok)
    r_base = model(base_tok)

assert r_src[0, -1].argmax().item()  == src_id,  "SOURCE wrong prediction"
assert r_base[0, -1].argmax().item() == base_id, "BASE wrong prediction"

ld_src  = (r_src[0, -1, src_id]  - r_src[0, -1, base_id]).item()
ld_base = (r_base[0, -1, src_id] - r_base[0, -1, base_id]).item()
print(f"SOURCE LD={ld_src:.3f}  BASE LD={ld_base:.3f}")

def metric(logits):
    ld = logits[0, -1, src_id] - logits[0, -1, base_id]
    return (ld - ld_base) / (ld_src - ld_base)

with torch.no_grad():
    _, src_cache  = model.run_with_cache(src_tok)
    _, base_cache = model.run_with_cache(base_tok)

tok_strs = [model.tokenizer.decode(t) for t in src_tok[0].tolist()]
seq_len  = len(tok_strs)
diff_pos = [i for i, (a, b) in enumerate(
    zip(src_tok[0].tolist(), base_tok[0].tolist())) if a != b]
print(f"seq_len={seq_len}  diff_pos={diff_pos} {[tok_strs[p] for p in diff_pos]}")


# ═══════════════════════════════════════════════════════════════════════════════
# ATTENTION PATTERN: SOURCE / BASE / DIFF for the binding head
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*65}")
print(f"ATTENTION PATTERN: L{BIND_LAYER}H{BIND_HEAD}")
print(f"{'='*65}")

src_attn  = src_cache["pattern",  BIND_LAYER][0, BIND_HEAD].float().cpu().numpy()
base_attn = base_cache["pattern", BIND_LAYER][0, BIND_HEAD].float().cpu().numpy()
diff_attn = src_attn - base_attn

fig, axes = plt.subplots(1, 3, figsize=(seq_len * 0.38 * 3, seq_len * 0.38 + 0.8))
for ax, data, title, cmap in zip(
    axes,
    [src_attn, base_attn, diff_attn],
    ["SOURCE", "BASE", "SOURCE − BASE"],
    ["Blues", "Blues", "RdBu_r"],
):
    vmax = max(abs(data).max(), 0.01)
    im = ax.imshow(data, cmap=cmap, aspect="auto",
                   vmin=-vmax if "−" in title else 0, vmax=vmax)
    plt.colorbar(im, ax=ax, shrink=0.6)
    ax.set_xticks(range(seq_len))
    ax.set_xticklabels(tok_strs, rotation=90, fontsize=6)
    ax.set_yticks(range(seq_len))
    ax.set_yticklabels(tok_strs, fontsize=6)
    ax.set_xlabel("Key (attended to)")
    ax.set_ylabel("Query (attending from)")
    for pos in diff_pos:
        ax.axvline(x=pos - 0.5, color="orange", alpha=0.6, lw=1.5)
        ax.axvline(x=pos + 0.5, color="orange", alpha=0.6, lw=1.5)
        ax.axhline(y=pos - 0.5, color="orange", alpha=0.3, lw=1.0)
        ax.axhline(y=pos + 0.5, color="orange", alpha=0.3, lw=1.0)
    ax.set_title(f"L{BIND_LAYER}H{BIND_HEAD} | {title}", fontsize=9)

fig.suptitle(f"Attention Pattern: L{BIND_LAYER}H{BIND_HEAD} [{args.model}]  ex1_1_1\n"
             f"orange = differing positions", fontsize=10)
plt.tight_layout()
plt.savefig(f"{OUTDIR}/attn_pattern.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved attn_pattern.png")

# Print top attended-to positions for the last few query rows
print(f"\n  Top attended-to positions (last 8 query rows):")
for qpos in range(seq_len - 8, seq_len):
    row = src_attn[qpos]
    top3 = np.argsort(row)[-3:][::-1]
    top3_str = ", ".join(f"{repr(tok_strs[k])}({row[k]:.2f})" for k in top3)
    print(f"    q[{qpos}]={repr(tok_strs[qpos]):15s}  attends to: {top3_str}")


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 0: Which component of the binding head carries the signal?
# Patch Q / K / V / Pattern / Z individually from SOURCE into BASE run.
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*65}")
print(f"PHASE 0: Q/K/V/Pattern/Z patching for L{BIND_LAYER}H{BIND_HEAD}")
print(f"{'='*65}")

def patch_single_component(comp):
    hook_name = utils.get_act_name(comp, BIND_LAYER)
    def hook_fn(act, hook):
        if comp == "pattern":
            act[:, BIND_HEAD] = src_cache[hook.name][:, BIND_HEAD]
        elif comp in ("k", "v"):
            kv_h = BIND_HEAD // GQA
            act[:, :, kv_h] = src_cache[hook.name][:, :, kv_h]
        else:  # q or z
            act[:, :, BIND_HEAD] = src_cache[hook.name][:, :, BIND_HEAD]
        return act
    with torch.no_grad():
        logits = model.run_with_hooks(base_tok, fwd_hooks=[(hook_name, hook_fn)],
                                      return_type="logits")
    return metric(logits).item()

components = ["q", "k", "v", "pattern", "z"]
phase0 = {c: patch_single_component(c) for c in components}
for c, v in phase0.items():
    print(f"  {c:8s}: {v:.4f}")

fig, ax = plt.subplots(figsize=(6, 3.5))
colors = ['steelblue' if v >= 0 else 'firebrick' for v in phase0.values()]
ax.bar(components, list(phase0.values()), color=colors)
ax.axhline(0, color='black', lw=0.5)
ax.axhline(1, color='green', lw=0.8, linestyle='--', alpha=0.6, label='SOURCE')
ax.set_ylabel("Patch metric (1=SOURCE, 0=BASE)")
ax.set_title(f"Phase 0: Which component carries the binding signal?\n"
             f"L{BIND_LAYER}H{BIND_HEAD} [{args.model}]  ex1_1_1")
ax.legend(); ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig(f"{OUTDIR}/phase0_components.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved phase0_components.png")


# ═══════════════════════════════════════════════════════════════════════════════
# RESIDUAL STREAM DECOMPOSITION in Q-space (Llama) or V-space (Qwen)
#
# For each earlier component (embed, pos_embed, head outputs, MLP outputs):
#   1. Compute (src_output - base_output) at the relevant position
#   2. Project through W_Q (if Q-dominant) or W_V (if V-dominant)
#   3. Dot with the binding direction in Q/V space
#
# Binding direction = unit vector along (src_Q - base_Q) or (src_V - base_V)
# of the binding head itself, at the relevant position.
# This answers: "who wrote the signal that L{B}H{H} reads?"
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*65}")
print(f"RESIDUAL DECOMPOSITION in dominant component space")
print(f"{'='*65}")

# Determine dominant component from Phase 0
if phase0["q"] >= phase0["v"]:
    DOMINANT = "q"
    W_proj   = model.W_Q[BIND_LAYER, BIND_HEAD].float()          # [d_model, d_head]
    proj_pos = seq_len - 1                                        # last token
    dir_src  = src_cache["q",  BIND_LAYER][0, proj_pos, BIND_HEAD, :].float()
    dir_base = base_cache["q", BIND_LAYER][0, proj_pos, BIND_HEAD, :].float()
else:
    DOMINANT = "v"
    kv_head  = BIND_HEAD // GQA
    W_proj   = model.W_V[BIND_LAYER, kv_head].float()            # [d_model, d_head]
    # V position: the token most attended-to from the last token in source
    attn_last = src_cache["pattern", BIND_LAYER][0, BIND_HEAD, -1, :]
    proj_pos  = int(attn_last.cpu().argmax().item())
    dir_src  = src_cache["v",  BIND_LAYER][0, proj_pos, kv_head, :].float()
    dir_base = base_cache["v", BIND_LAYER][0, proj_pos, kv_head, :].float()

bind_dir = (dir_src - dir_base).cpu()
bind_dir = bind_dir / (bind_dir.norm() + 1e-8)

print(f"  Dominant: {DOMINANT.upper()}-circuit")
print(f"  Projection position: {proj_pos} = {repr(tok_strs[proj_pos])}")
print(f"  Binding direction norm (pre-normalise): {(dir_src - dir_base).norm().item():.4f}")

W_proj = W_proj.cpu()

def proj_score(out_src, out_base):
    """Project (src - base) component output through W_proj, dot with bind_dir."""
    delta = (out_src - out_base).cpu().float()
    return (delta @ W_proj).dot(bind_dir).item()

# --- Embedding and positional embedding ---
embed_score = proj_score(src_cache["hook_embed"][0, proj_pos, :],
                          base_cache["hook_embed"][0, proj_pos, :])
# pos_embed only exists for models with additive positional embeddings (not RoPE)
if "hook_pos_embed" in src_cache.cache_dict:
    pos_emb_score = proj_score(src_cache["hook_pos_embed"][0, proj_pos, :],
                                base_cache["hook_pos_embed"][0, proj_pos, :])
else:
    pos_emb_score = None  # RoPE model — positional info is in attention, not residual stream

# --- Heads and MLPs for each layer ---
head_scores = np.zeros((BIND_LAYER, NH))
mlp_scores  = np.zeros(BIND_LAYER)

for l in range(BIND_LAYER):
    # head output = z @ W_O  (W_O: [n_heads, d_head, d_model])
    z_src  = src_cache["z",  l][0, proj_pos, :, :].float().cpu()   # [n_heads, d_head]
    z_base = base_cache["z", l][0, proj_pos, :, :].float().cpu()
    W_O    = model.W_O[l].float().cpu()                           # [n_heads, d_head, d_model]
    for h in range(NH):
        out_src  = z_src[h]  @ W_O[h]   # [d_model]
        out_base = z_base[h] @ W_O[h]
        head_scores[l, h] = proj_score(out_src, out_base)
    mlp_scores[l] = proj_score(
        src_cache["mlp_out",  l][0, proj_pos, :],
        base_cache["mlp_out", l][0, proj_pos, :])

# --- Plot 1: head score heatmap ---
vmax = max(abs(head_scores).max(), 0.01)
fig, ax = plt.subplots(figsize=(NH * 0.45, BIND_LAYER * 0.32 + 0.8))
im = ax.imshow(head_scores, aspect="auto", cmap="RdBu_r", origin="lower",
               vmin=-vmax, vmax=vmax)
plt.colorbar(im, ax=ax, label=f"Contribution to binding dir in {DOMINANT.upper()}-space")
ax.set_xlabel("Head"); ax.set_ylabel("Layer")
ax.set_title(f"Residual decomp → {DOMINANT.upper()} of L{BIND_LAYER}H{BIND_HEAD}\n"
             f"[{args.model}]  pos={proj_pos} ({repr(tok_strs[proj_pos])})")
ax.set_xticks(range(NH)); ax.set_yticks(range(BIND_LAYER))
# Highlight top 5
flat = head_scores.flatten()
for idx in np.argsort(np.abs(flat))[-5:]:
    sl, sh = divmod(idx, NH)
    ax.add_patch(plt.Rectangle((sh-0.5, sl-0.5), 1, 1,
                                fill=False, edgecolor="lime", lw=2))
plt.tight_layout()
plt.savefig(f"{OUTDIR}/resid_decomp_heads.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved resid_decomp_heads.png")

# --- Plot 2: MLP scores bar chart + embed ---
fig, ax = plt.subplots(figsize=(max(8, BIND_LAYER * 0.5), 3.5))
x = np.arange(BIND_LAYER)
colors = ["steelblue" if v >= 0 else "firebrick" for v in mlp_scores]
ax.bar(x, mlp_scores, color=colors, label="MLP")
ax.axhline(0, color="black", lw=0.5)
# Mark embed/pos_embed
ax.axhline(embed_score, color="green", lw=1.5, linestyle="--",
           label=f"embed ({embed_score:.3f})")
if pos_emb_score is not None:
    ax.axhline(pos_emb_score, color="purple", lw=1.5, linestyle=":",
               label=f"pos_embed ({pos_emb_score:.3f})")
ax.set_xticks(x); ax.set_xticklabels([f"L{l}" for l in x], fontsize=7)
ax.set_ylabel(f"Contribution to binding dir in {DOMINANT.upper()}-space")
ax.set_title(f"MLP contributions → {DOMINANT.upper()} of L{BIND_LAYER}H{BIND_HEAD}\n"
             f"[{args.model}]  pos={proj_pos} ({repr(tok_strs[proj_pos])})")
ax.legend(fontsize=8); ax.grid(True, alpha=0.3, axis="y")
plt.tight_layout()
plt.savefig(f"{OUTDIR}/resid_decomp_mlps.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved resid_decomp_mlps.png")

# --- Print top contributors ---
print(f"\n  Top 10 heads (by |score|):")
for idx in np.argsort(np.abs(flat))[-10:][::-1]:
    sl, sh = divmod(idx, NH)
    print(f"    L{sl}H{sh}: {flat[idx]:+.4f}")
print(f"\n  MLP scores by layer:")
for l in range(BIND_LAYER):
    print(f"    L{l}: {mlp_scores[l]:+.4f}")
print(f"\n  Embedding:  {embed_score:+.4f}")
if pos_emb_score is not None:
    print(f"  Pos embed:  {pos_emb_score:+.4f}")
else:
    print(f"  Pos embed:  N/A (RoPE model)")


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 1 & 2: Path patching — senders (L0..L(bind-1)) -> binding_head.Q and .K
#
# For each sender (sl, sh):
#   Step A: run BASE, freeze all z to BASE, patch sender (sl,sh) z to SOURCE.
#           Cache binding head's Q (or K) input under this intervention.
#   Step B: run BASE normally, ONLY patch binding head's Q (or K) to Step A value.
#           Measure metric.
#
# This isolates: "how much does sender (sl,sh) contribute to binding_head.Q/K?"
# ═══════════════════════════════════════════════════════════════════════════════

def path_patch_senders_to_binding_input(recv_input):
    """
    recv_input: 'q' or 'k'
    Returns matrix [BIND_LAYER, NH] of patch metric values.
    """
    recv_hook_name = utils.get_act_name(recv_input, BIND_LAYER)
    results = torch.zeros(BIND_LAYER, NH)

    for sl in tqdm(range(BIND_LAYER),
                   desc=f"Path patch senders → L{BIND_LAYER}H{BIND_HEAD}.{recv_input.upper()}"):
        for sh in range(NH):

            # Step A: freeze all z to BASE, patch sender (sl, sh) to SOURCE
            def hook_step_A(z, hook, _sl=sl, _sh=sh):
                z[...] = base_cache[hook.name][...]
                if hook.layer() == _sl:
                    z[:, :, _sh, :] = src_cache[hook.name][:, :, _sh, :]
                return z

            hooks_A = [(name, hook_step_A) for name in ALL_Z]
            with torch.no_grad(), model.hooks(fwd_hooks=hooks_A):
                _, patched_cache = model.run_with_cache(
                    base_tok, names_filter=lambda n: n == recv_hook_name)

            # Step B: run BASE normally, only patch binding head's Q or K
            def hook_step_B(act, hook, _ri=recv_input, _pc=patched_cache):
                if _ri in ("k", "v"):
                    kv_h = BIND_HEAD // GQA
                    act[:, :, kv_h] = _pc[hook.name][:, :, kv_h]
                else:
                    act[:, :, BIND_HEAD] = _pc[hook.name][:, :, BIND_HEAD]
                return act

            with torch.no_grad():
                logits = model.run_with_hooks(
                    base_tok,
                    fwd_hooks=[(recv_hook_name, hook_step_B)],
                    return_type="logits")
            results[sl, sh] = metric(logits).item()

    return results


if args.skip_patching:
    print("\nSkipping path patching (--skip-patching)")
    exit(0)

print(f"\n{'='*65}")
print(f"PHASE 1: Path patch senders → L{BIND_LAYER}H{BIND_HEAD}.Q")
print(f"{'='*65}")
q_results = path_patch_senders_to_binding_input("q")

print(f"\n{'='*65}")
print(f"PHASE 2: Path patch senders → L{BIND_LAYER}H{BIND_HEAD}.K")
print(f"{'='*65}")
k_results = path_patch_senders_to_binding_input("k")


# ── Plot Q and K sender heatmaps ──────────────────────────────────────────────
def plot_sender_heatmap(data, recv_input, fname):
    rows, cols = data.shape
    vmax = max(float(data.abs().max()), 0.05)
    fig, ax = plt.subplots(figsize=(cols * 0.45, rows * 0.32 + 0.8))
    im = ax.imshow(data.numpy(), aspect='auto', cmap='RdBu_r', origin='lower',
                   vmin=0, vmax=vmax)
    plt.colorbar(im, ax=ax, label="Metric (1=SOURCE, 0=BASE)")
    ax.set_xlabel("Sender Head"); ax.set_ylabel("Sender Layer")
    ax.set_title(f"Senders → L{BIND_LAYER}H{BIND_HEAD}.{recv_input.upper()}\n"
                 f"[{args.model}]  ex1_1_1  (which earlier head writes binding info?)")
    ax.set_xticks(range(cols))
    ax.set_yticks(range(rows))
    # Annotate top cells
    flat = data.numpy().flatten()
    for idx in np.argsort(flat)[-5:][::-1]:
        sl, sh = divmod(idx, cols)
        ax.add_patch(plt.Rectangle((sh-0.5, sl-0.5), 1, 1,
                                    fill=False, edgecolor='lime', lw=2))
    plt.tight_layout()
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close()

plot_sender_heatmap(q_results, "q",
                    f"{OUTDIR}/phase1_senders_to_Q.png")
print(f"  Saved phase1_senders_to_Q.png")

plot_sender_heatmap(k_results, "k",
                    f"{OUTDIR}/phase2_senders_to_K.png")
print(f"  Saved phase2_senders_to_K.png")


# ── Combined Q + K side by side ───────────────────────────────────────────────
vmax = max(float(q_results.abs().max()), float(k_results.abs().max()), 0.05)
fig, axes = plt.subplots(1, 2, figsize=(NH * 0.45 * 2, BIND_LAYER * 0.32 + 1))
for ax, data, recv_input in zip(axes, [q_results, k_results], ["Q", "K"]):
    im = ax.imshow(data.numpy(), aspect='auto', cmap='RdBu_r', origin='lower',
                   vmin=0, vmax=vmax)
    plt.colorbar(im, ax=ax, label="Metric")
    ax.set_xlabel("Sender Head"); ax.set_ylabel("Sender Layer")
    ax.set_title(f"Senders → {recv_input} of L{BIND_LAYER}H{BIND_HEAD}")
    ax.set_xticks(range(NH)); ax.set_yticks(range(BIND_LAYER))
    flat = data.numpy().flatten()
    for idx in np.argsort(flat)[-5:][::-1]:
        sl, sh = divmod(idx, NH)
        ax.add_patch(plt.Rectangle((sh-0.5, sl-0.5), 1, 1,
                                    fill=False, edgecolor='lime', lw=2))
fig.suptitle(f"Path Patching: What feeds into L{BIND_LAYER}H{BIND_HEAD}? [{args.model}]  ex1_1_1",
             fontsize=11)
plt.tight_layout()
plt.savefig(f"{OUTDIR}/phase12_senders_QK.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved phase12_senders_QK.png")


# ── Print top senders ─────────────────────────────────────────────────────────
print(f"\n{'='*65}")
print(f"SUMMARY: Top senders into L{BIND_LAYER}H{BIND_HEAD} [{args.model}]")
print(f"{'='*65}")
for label, data in [("Q", q_results), ("K", k_results)]:
    flat = data.numpy().flatten()
    print(f"\n  Top 10 senders → {label}:")
    for idx in np.argsort(flat)[-10:][::-1]:
        sl, sh = divmod(idx, NH)
        print(f"    L{sl}H{sh}: {flat[idx]:.4f}")

print(f"\nAll plots → {OUTDIR}/")
