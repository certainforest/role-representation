"""
Trace when and how binding information first appears at a target token position.

Usage:
  python trace_query_alice.py --model llama --gpu 0 --target query_alice
  python trace_query_alice.py --model llama --gpu 0 --target intro_alice
  python trace_query_alice.py --model qwen  --gpu 0 --target query_alice
"""
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model", choices=["llama", "qwen"], default="llama")
parser.add_argument("--gpu", default="0")
parser.add_argument("--target", choices=["query_alice", "intro_alice"],
                    default="query_alice",
                    help="Which token position to decompose")
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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

OUTDIR = (f"/mnt/ssd/aryawu/role-representation/attribution"
          f"/circuit_results_{args.model}/_trace_{args.target}")
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
print(f"Loading {MODEL_CONFIGS[args.model]['name']}...")
model = HookedTransformer.from_pretrained(
    MODEL_CONFIGS[args.model]["name"],
    dtype=torch.bfloat16,
    cache_dir=MODEL_CONFIGS[args.model]["cache"])
model.eval().to("cuda")

NL = model.cfg.n_layers
NH = model.cfg.n_heads
print(f"Loaded: {NL}L {NH}H")

# ── Tokenize + forward passes ─────────────────────────────────────────────────
src_tok  = model.to_tokens(EXP["source_prompt"], prepend_bos=True)
base_tok = model.to_tokens(EXP["base_prompt"],   prepend_bos=True)
assert src_tok.shape == base_tok.shape

tok_strs = [model.tokenizer.decode(t) for t in src_tok[0].tolist()]
seq_len  = len(tok_strs)
diff_pos = [i for i, (a, b) in enumerate(
    zip(src_tok[0].tolist(), base_tok[0].tolist())) if a != b]

# Query Alice: the 'Alice' token in "Where does Alice live?"
# Find it: last occurrence of Alice before the end
alice_tok_id = model.tokenizer.encode(" Alice", add_special_tokens=False)[0]
alice_positions = [i for i, t in enumerate(src_tok[0].tolist()) if t == alice_tok_id]
# intro Alice = first occurrence, query Alice = second occurrence
intro_alice_pos = alice_positions[0]
query_alice_pos = alice_positions[1]
print(f"seq_len={seq_len}")
print(f"diff_pos={diff_pos} = {[repr(tok_strs[p]) for p in diff_pos]}")
print(f"intro Alice = pos {intro_alice_pos} ({repr(tok_strs[intro_alice_pos])})")
print(f"query Alice = pos {query_alice_pos} ({repr(tok_strs[query_alice_pos])})")

with torch.no_grad():
    _, src_cache  = model.run_with_cache(src_tok)
    _, base_cache = model.run_with_cache(base_tok)

TARGET_POS = query_alice_pos if args.target == "query_alice" else intro_alice_pos
print(f"Target: {args.target} = pos {TARGET_POS} ({repr(tok_strs[TARGET_POS])})")


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Per-component activation diff at TARGET_POS
#    score = ||src_output - base_output|| at TARGET_POS for each component
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\nDecomposing residual stream at pos {TARGET_POS} = {repr(tok_strs[TARGET_POS])}")

head_diff = np.zeros((NL, NH))   # L2 norm of (src - base) head output
mlp_diff  = np.zeros(NL)

for l in range(NL):
    z_src  = src_cache["z",  l][0, TARGET_POS, :, :].float().cpu()  # [NH, d_head]
    z_base = base_cache["z", l][0, TARGET_POS, :, :].float().cpu()
    W_O    = model.W_O[l].float().cpu()                              # [NH, d_head, d_model]
    for h in range(NH):
        out_src  = z_src[h]  @ W_O[h]   # [d_model]
        out_base = z_base[h] @ W_O[h]
        head_diff[l, h] = (out_src - out_base).norm().item()

    mlp_src  = src_cache["mlp_out",  l][0, TARGET_POS, :].float().cpu()
    mlp_base = base_cache["mlp_out", l][0, TARGET_POS, :].float().cpu()
    mlp_diff[l] = (mlp_src - mlp_base).norm().item()

# embedding diff
embed_diff = (src_cache["hook_embed"][0, TARGET_POS, :].float().cpu() -
              base_cache["hook_embed"][0, TARGET_POS, :].float().cpu()).norm().item()

# ── Plot: head diff heatmap ───────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(NH * 0.45, NL * 0.28 + 0.8))
im = ax.imshow(head_diff, aspect="auto", cmap="Reds", origin="lower")
plt.colorbar(im, ax=ax, label="||src_output − base_output||")
ax.set_xlabel("Head"); ax.set_ylabel("Layer")
ax.set_title(f"Head output diff at query Alice (pos {TARGET_POS})\n"
             f"[{args.model}]  — when does binding info first appear?")
ax.set_xticks(range(NH)); ax.set_yticks(range(NL))
# Highlight top 5
flat = head_diff.flatten()
for idx in np.argsort(flat)[-5:]:
    sl, sh = divmod(idx, NH)
    ax.add_patch(plt.Rectangle((sh-0.5, sl-0.5), 1, 1,
                                fill=False, edgecolor="blue", lw=2))
plt.tight_layout()
plt.savefig(f"{OUTDIR}/head_diff_at_query_alice.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved head_diff_at_query_alice.png")

# ── Plot: MLP diff + accumulated total ───────────────────────────────────────
fig, axes = plt.subplots(2, 1, figsize=(max(10, NL * 0.5), 6), sharex=True)

# MLP per layer
ax = axes[0]
colors = ["steelblue"] * NL
ax.bar(range(NL), mlp_diff, color=colors)
ax.axhline(embed_diff, color="green", lw=1.5, linestyle="--",
           label=f"embed diff ({embed_diff:.3f})")
ax.set_ylabel("||src − base||")
ax.set_title(f"MLP output diff at query Alice (pos {TARGET_POS})  [{args.model}]")
ax.legend(fontsize=8); ax.grid(True, alpha=0.3, axis="y")

# Accumulated residual stream diff (total signal buildup)
acc_diff = np.zeros(NL + 1)
acc_diff[0] = embed_diff
for l in range(NL):
    # contribution of this layer = max head diff + mlp diff (rough accumulation)
    acc_diff[l + 1] = acc_diff[l] + head_diff[l].max() + mlp_diff[l]

ax2 = axes[1]
ax2.plot(range(NL + 1), acc_diff, marker="o", markersize=3, color="firebrick")
ax2.set_xlabel("Layer"); ax2.set_ylabel("Accumulated ||src − base||")
ax2.set_title("Accumulated binding signal at query Alice (rough)")
ax2.set_xticks(range(0, NL + 1, 2)); ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f"{OUTDIR}/mlp_diff_at_query_alice.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved mlp_diff_at_query_alice.png")


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Attention patterns of top-writing heads at query Alice
#    For the top-5 heads by diff norm: where did they attend from?
#    Hypothesis: they attended from intro Alice (pos 11) or line-ending tokens
# ═══════════════════════════════════════════════════════════════════════════════
top5_idx = np.argsort(flat)[-5:][::-1]
top5_heads = [(int(l), int(h)) for l, h in [divmod(i, NH) for i in top5_idx]]

print(f"\nTop 5 heads writing to query Alice:")
for (l, h), idx in zip(top5_heads, top5_idx):
    print(f"  L{l}H{h}: diff={flat[idx]:.4f}")
    src_row  = src_cache["pattern",  l][0, h, TARGET_POS, :].float().cpu().numpy()
    base_row = base_cache["pattern", l][0, h, TARGET_POS, :].float().cpu().numpy()
    top3_src  = np.argsort(src_row)[-3:][::-1]
    top3_base = np.argsort(base_row)[-3:][::-1]
    src_str  = ", ".join(f"{repr(tok_strs[k])}({src_row[k]:.2f})"  for k in top3_src)
    base_str = ", ".join(f"{repr(tok_strs[k])}({base_row[k]:.2f})" for k in top3_base)
    print(f"    SOURCE attends to: {src_str}")
    print(f"    BASE   attends to: {base_str}")

# Plot attention patterns of top-5 heads (query row = TARGET_POS)
fig, axes = plt.subplots(len(top5_heads), 3,
                          figsize=(seq_len * 0.35 * 3, len(top5_heads) * seq_len * 0.18 + 0.5))
if len(top5_heads) == 1:
    axes = axes[np.newaxis, :]

for row, (l, h) in enumerate(top5_heads):
    src_attn  = src_cache["pattern",  l][0, h].float().cpu().numpy()
    base_attn = base_cache["pattern", l][0, h].float().cpu().numpy()
    diff_attn = src_attn - base_attn

    for col, (data, title, cmap) in enumerate([
        (src_attn,  "SOURCE",        "Blues"),
        (base_attn, "BASE",          "Blues"),
        (diff_attn, "SOURCE − BASE", "RdBu_r"),
    ]):
        ax = axes[row, col]
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
        # Mark target pos and diff positions
        ax.axhline(y=TARGET_POS - 0.5, color="lime",   alpha=0.7, lw=1.5)
        ax.axhline(y=TARGET_POS + 0.5, color="lime",   alpha=0.7, lw=1.5)
        for pos in diff_pos:
            ax.axvline(x=pos - 0.5, color="orange", alpha=0.5, lw=1.2)
            ax.axvline(x=pos + 0.5, color="orange", alpha=0.5, lw=1.2)
        ax.set_title(f"L{l}H{h} diff={flat[l*NH+h]:.3f} | {title}", fontsize=8)

fig.suptitle(f"Attention patterns: top-5 heads writing to query Alice (pos {TARGET_POS})\n"
             f"[{args.model}]  lime=query Alice row, orange=differing positions", fontsize=9)
plt.tight_layout()
plt.savefig(f"{OUTDIR}/top_heads_attn_patterns.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved top_heads_attn_patterns.png")


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Summary: first layer where binding diff exceeds threshold
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*65}")
print(f"SUMMARY  [{args.model}]  — binding signal at query Alice (pos {TARGET_POS})")
print(f"{'='*65}")

threshold = np.percentile(head_diff, 95)  # top 5% of head diffs
print(f"\nHead diff threshold (95th pct): {threshold:.4f}")
print(f"First layer with a head above threshold:")
for l in range(NL):
    if head_diff[l].max() >= threshold:
        h = head_diff[l].argmax()
        print(f"  L{l}H{h}: {head_diff[l, h]:.4f}")
        break

print(f"\nTop 10 heads by diff at query Alice:")
for idx in np.argsort(flat)[-10:][::-1]:
    sl, sh = divmod(idx, NH)
    print(f"  L{sl}H{sh}: {flat[idx]:.4f}")

print(f"\nTop 5 MLP layers by diff at query Alice:")
for idx in np.argsort(mlp_diff)[-5:][::-1]:
    print(f"  L{idx}: {mlp_diff[idx]:.4f}")

print(f"\nAll plots → {OUTDIR}/")
