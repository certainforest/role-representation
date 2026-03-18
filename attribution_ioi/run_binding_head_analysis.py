"""
Targeted binding-head analysis: attention patterns + output semantics + upstream attribution.

Validates the 4-step binding hypothesis (EXPERIMENT_BASE_ANALYSIS.md, EXPERIMENT_Hi_ANALYSIS.md):
  Step 0: binding-ID assigned per line, early layers
  Step 1: query entity/attribute read
  Step 2: binding-ID retrieved for query (layers 5-13 Llama, 5-21 Qwen)
  Step 3: answer looked up via binding-ID   ← e.g. L13H18 Llama
  Step 4: answer copied to final token      ← e.g. L15H8 Llama

For each identified binding-specific head, produces:
  1_attn.png     — SOURCE / BASE / SOURCE-BASE attention → what does the head READ?
  2_output.png   — head output semantics → what does the head WRITE?
                     row 1: logit-diff contribution per position (SOURCE vs BASE)
                     row 2: output norm per position (SOURCE vs BASE)
                     row 3: top-10 tokens promoted at last position
  3_upstream.png — residual decomp into head's dominant input (Q or V) → WHO writes into it?

Usage:
  python run_binding_head_analysis.py --model llama --gpu 0 --setup base
  python run_binding_head_analysis.py --model llama --gpu 0 --setup hi
  python run_binding_head_analysis.py --model qwen  --gpu 1 --setup base
  python run_binding_head_analysis.py --model qwen  --gpu 1 --setup hi
"""
import os, sys, argparse
import torch
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from transformer_lens import HookedTransformer
from transformer_lens import utils as tl_utils

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import SWAP_EXPERIMENTS, HI_SWAP_EXPERIMENTS

# ── Target heads: forward ∩ reverse binding-only, K=5 ────────────────────────
TARGET_HEADS = {
    "qwen": {
        "base": [(21,18), (21,19), (23,25), (23,26)],
        "hi":   [(21,18), (22,10), (23,30), (26,25)],
    },
    "llama": {
        "base": [(13,18), (14,7), (15,8), (15,11), (19,20)],
        "hi":   [(13,13), (14,4), (14,6), (14,7), (14,22), (15,11)],
    },
}

MODEL_CONFIGS = {
    "llama": {"name": "meta-llama/Llama-3.1-8B-Instruct",
              "cache": "/mnt/ssd/aryawu/.cache/huggingface/hub"},
    "qwen":  {"name": "Qwen/Qwen3-8B",
              "cache": "/mnt/ssd/aryawu/.cache/huggingface/hub"},
}

BASE_RESULTS = "/mnt/ssd/aryawu/role-representation/attribution_ioi/results"

parser = argparse.ArgumentParser()
parser.add_argument("--model",  choices=["llama", "qwen"], default="llama")
parser.add_argument("--gpu",    default="0")
parser.add_argument("--setup",  choices=["base", "hi"],    default="base",
                    help="base: standard Swap; hi: Hi-greeting Swap")
parser.add_argument("--exp",    choices=["name", "country"], default="name",
                    help="name: 'Where does Alice live?' (→ France); "
                         "country: 'Who lives in France?' (→ Alice)")
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

# ── Select primary experiment ─────────────────────────────────────────────────
# BINDING experiments:
#   name:    Q_name_bind    / Q_name_hi_bind    "Where does Alice live?" → France
#   country: Q_country_bind / Q_country_hi_bind "Who lives in France?"   → Alice
_key_map = {
    ("base", "name"):    ("Q_name_bind",       SWAP_EXPERIMENTS),
    ("base", "country"): ("Q_country_bind",    SWAP_EXPERIMENTS),
    ("hi",   "name"):    ("Q_name_hi_bind",    HI_SWAP_EXPERIMENTS),
    ("hi",   "country"): ("Q_country_hi_bind", HI_SWAP_EXPERIMENTS),
}
_exp_key, _pool = _key_map[(args.setup, args.exp)]
exp = next(e for e in _pool if e["key"] == _exp_key and e["kind"] == "BINDING")

print(f"Setup: {args.setup}  experiment: {exp['key']} [{exp['kind']}]")
print(f"SOURCE answer: {exp['source_answer']!r}  BASE answer: {exp['base_answer']!r}")

# ── Load model ────────────────────────────────────────────────────────────────
cfg = MODEL_CONFIGS[args.model]
print(f"\nLoading {cfg['name']}...")
model = HookedTransformer.from_pretrained(cfg["name"], dtype=torch.bfloat16,
                                          cache_dir=cfg["cache"])
model.eval().to("cuda")

NL      = model.cfg.n_layers
NH      = model.cfg.n_heads
NKV     = model.cfg.n_key_value_heads
GQA     = NH // NKV
d_model = model.cfg.d_model
d_head  = model.cfg.d_head
print(f"Loaded: {NL}L {NH}H NKV={NKV} GQA={GQA}x  d_model={d_model} d_head={d_head}")

# ── Tokenize + forward passes ─────────────────────────────────────────────────
src_tok  = model.to_tokens(exp["source_prompt"], prepend_bos=True)
base_tok = model.to_tokens(exp["base_prompt"],   prepend_bos=True)
assert src_tok.shape == base_tok.shape, \
    f"Token length mismatch: {src_tok.shape} vs {base_tok.shape}"
seq_len  = src_tok.shape[1]

src_ids  = src_tok[0].tolist()
base_ids = base_tok[0].tolist()
tok_strs = [model.tokenizer.decode(t) for t in src_ids]
diff_pos = [i for i, (a, b) in enumerate(zip(src_ids, base_ids)) if a != b]
print(f"\nseq_len={seq_len}  diff_pos={diff_pos}")
print(f"  diff tokens: {[(i, repr(tok_strs[i])) for i in diff_pos]}")

# Identify key semantic positions for printing
# (name positions = first diff_pos entries in base setup; query entity near end)
print(f"\nAll token positions:")
for i, t in enumerate(tok_strs):
    marker = " ← DIFF" if i in diff_pos else ""
    print(f"  [{i:2d}] {repr(t)}{marker}")

src_id  = model.tokenizer.encode(exp["source_answer"], add_special_tokens=False)[0]
base_id = model.tokenizer.encode(exp["base_answer"],   add_special_tokens=False)[0]
print(f"\nsrc_id={src_id} ({exp['source_answer']!r})  base_id={base_id} ({exp['base_answer']!r})")

with torch.no_grad():
    src_logits_raw  = model(src_tok)
    base_logits_raw = model(base_tok)

src_pred  = model.tokenizer.decode(src_logits_raw[0, -1].argmax().item())
base_pred = model.tokenizer.decode(base_logits_raw[0, -1].argmax().item())
print(f"\nSOURCE pred: {src_pred!r}  (expected {exp['source_answer']!r}) "
      f"{'✓' if src_pred.strip() == exp['source_answer'].strip() else '✗ WRONG'}")
print(f"BASE   pred: {base_pred!r}  (expected {exp['base_answer']!r}) "
      f"{'✓' if base_pred.strip() == exp['base_answer'].strip() else '✗ WRONG'}")
assert src_logits_raw[0, -1].argmax().item() == src_id,  f"SOURCE wrong: {src_pred!r}"
assert base_logits_raw[0, -1].argmax().item() == base_id, f"BASE wrong: {base_pred!r}"

ld_src  = (src_logits_raw[0, -1, src_id]  - src_logits_raw[0, -1, base_id]).item()
ld_base = (base_logits_raw[0, -1, src_id] - base_logits_raw[0, -1, base_id]).item()
print(f"\nLogit diff: SOURCE={ld_src:.3f}  BASE={ld_base:.3f}  gap={ld_src - ld_base:.3f}")
print("(gap > 0 means there is a signal to explain)")

def metric(logits):
    ld = logits[0, -1, src_id] - logits[0, -1, base_id]
    denom = ld_src - ld_base
    if abs(denom) < 0.01:
        return ld / (abs(ld_src) + 1e-6)
    return (ld - ld_base) / denom

# Cache activations for both prompts
print("\nCaching activations for SOURCE and BASE runs...")
with torch.no_grad():
    _, src_cache  = model.run_with_cache(src_tok)
    _, base_cache = model.run_with_cache(base_tok)
print("Done.")

# Logit-diff direction in d_model space (W_U[:,src] - W_U[:,base])
# Used for: "how much does a d_model vector promote src_answer over base_answer?"
# Note: approximation — ignores layer norm before W_U, but valid for relative comparisons.
W_U    = model.W_U.float().detach().cpu()   # [d_model, vocab]
ld_dir = (W_U[:, src_id] - W_U[:, base_id])  # [d_model]  — the "correct answer direction"
print(f"\nLogit-diff direction norm: {ld_dir.norm().item():.4f}")
print(f"(Any head output dotted with this shows how much it promotes {exp['source_answer']!r})")

# ─────────────────────────────────────────────────────────────────────────────
# ANALYSIS LOOP
# ─────────────────────────────────────────────────────────────────────────────
target_heads = TARGET_HEADS[args.model][args.setup]
out_base = os.path.join(BASE_RESULTS, args.model, f"binding_head_analysis_{args.setup}_{args.exp}")
print(f"\nAnalyzing {len(target_heads)} heads: {[f'L{l}H{h}' for l,h in target_heads]}")
print(f"Output dir: {out_base}/\n")

for LAYER, HEAD in target_heads:
    print(f"\n{'='*65}")
    print(f"L{LAYER}H{HEAD}  [{args.model}  setup={args.setup}]")
    print(f"{'='*65}")

    outdir = os.path.join(out_base, f"L{LAYER}H{HEAD}")
    os.makedirs(outdir, exist_ok=True)

    # ── 1. Attention patterns ─────────────────────────────────────────────────
    # cache["pattern", l] shape: [batch, n_heads, seq, seq]
    # attn[q, k] = attention weight FROM query position q TO key position k
    src_attn  = src_cache["pattern",  LAYER][0, HEAD].float().cpu().numpy()   # [seq, seq]
    base_attn = base_cache["pattern", LAYER][0, HEAD].float().cpu().numpy()
    diff_attn = src_attn - base_attn   # positive = SOURCE attends more than BASE

    # Key diagnostic: what does the LAST token (answer position) attend to?
    # Hypothesis: lookup/copy heads should attend to name or country tokens from query pos.
    print(f"\n[Attn] Last token (pos {seq_len-1}={repr(tok_strs[-1])}) attends to:")
    last_src  = src_attn[-1]
    last_base = base_attn[-1]
    top5_src  = np.argsort(last_src)[-5:][::-1]
    top5_base = np.argsort(last_base)[-5:][::-1]
    print(f"  SOURCE top-5: {[(i, repr(tok_strs[i]), f'{last_src[i]:.3f}') for i in top5_src]}")
    print(f"  BASE   top-5: {[(i, repr(tok_strs[i]), f'{last_base[i]:.3f}') for i in top5_base]}")
    print(f"  Diff (src-base) at diff_pos: "
          f"{[(i, repr(tok_strs[i]), f'{diff_attn[-1, i]:+.3f}') for i in diff_pos]}")

    # Also check which query rows show the largest attention diff at diff positions
    # This reveals which query positions (rows) drive the binding-specific attention
    diff_attn_at_name = diff_attn[:, diff_pos].sum(axis=1)  # [seq]: sum over diff cols
    top3_rows = np.argsort(np.abs(diff_attn_at_name))[-3:][::-1]
    print(f"\n[Attn] Query rows with largest diff at name/attr positions:")
    for r in top3_rows:
        print(f"  row {r:2d} {repr(tok_strs[r]):15s}: diff={diff_attn_at_name[r]:+.4f}")

    fig, axes = plt.subplots(1, 3,
                             figsize=(seq_len * 0.35 * 3, seq_len * 0.35 + 1.5))
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
        ax.set_xticks(range(seq_len)); ax.set_xticklabels(tok_strs, rotation=90, fontsize=5)
        ax.set_yticks(range(seq_len)); ax.set_yticklabels(tok_strs, fontsize=5)
        ax.set_xlabel("Key (attended to)"); ax.set_ylabel("Query (attending from)")
        for p in diff_pos:
            ax.axvline(x=p-0.5, color="orange", alpha=0.6, lw=1.5)
            ax.axvline(x=p+0.5, color="orange", alpha=0.6, lw=1.5)
            ax.axhline(y=p-0.5, color="orange", alpha=0.3, lw=1.0)
            ax.axhline(y=p+0.5, color="orange", alpha=0.3, lw=1.0)
        ax.set_title(f"L{LAYER}H{HEAD} | {title}", fontsize=9)
    fig.suptitle(
        f"Attention: L{LAYER}H{HEAD} [{args.model} {args.setup}]\n"
        f"orange=diff positions  src={exp['source_answer']!r}  base={exp['base_answer']!r}",
        fontsize=10)
    plt.tight_layout()
    plt.savefig(f"{outdir}/1_attn.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  → Saved 1_attn.png")

    # ── 2. Head output semantics ──────────────────────────────────────────────
    # head_out[pos] = z[pos, HEAD] @ W_O[HEAD]  →  this head's contribution to
    # the residual stream at each position.
    # Approximation: we dot with W_U direction without applying final layernorm,
    # valid for comparing positions and SOURCE vs BASE relative changes.

    W_O_head = model.W_O[LAYER, HEAD].float().cpu()           # [d_head, d_model]
    z_src    = src_cache["z",  LAYER][0, :, HEAD, :].float().cpu()  # [seq, d_head]
    z_base   = base_cache["z", LAYER][0, :, HEAD, :].float().cpu()

    head_out_src  = z_src  @ W_O_head   # [seq, d_model] — head's write to residual stream
    head_out_base = z_base @ W_O_head

    # Logit-diff contribution: how much does this head, at each position,
    # push the output toward src_answer vs base_answer?
    # Positive = helps the correct answer; negative = hurts it.
    ld_contrib_src  = (head_out_src  @ ld_dir).detach().numpy()   # [seq]
    ld_contrib_base = (head_out_base @ ld_dir).detach().numpy()

    print(f"\n[Output] Logit-diff contribution at LAST token (pos {seq_len-1}):")
    print(f"  SOURCE: {ld_contrib_src[-1]:+.4f}   "
          f"(positive = head promotes {exp['source_answer']!r})")
    print(f"  BASE:   {ld_contrib_base[-1]:+.4f}")
    print(f"  src−base diff: {ld_contrib_src[-1] - ld_contrib_base[-1]:+.4f}  "
          f"(positive = head is binding-specific at this position)")

    # Top positions by absolute LD contribution (SOURCE)
    top5_pos = np.argsort(np.abs(ld_contrib_src))[-5:][::-1]
    print(f"\n[Output] Top-5 positions by |LD contribution| (SOURCE run):")
    for p in top5_pos:
        print(f"  pos {p:2d} {repr(tok_strs[p]):18s}: {ld_contrib_src[p]:+.4f}  "
              f"{'← DIFF pos' if p in diff_pos else ''}")

    # Top-10 tokens promoted at last position — tells us what this head "wants to say"
    # at the answer boundary. Hypothesis: lookup/copy heads should promote src_answer token.
    last_logits_src  = (head_out_src[-1]  @ W_U).detach().numpy()   # [vocab]
    last_logits_base = (head_out_base[-1] @ W_U).detach().numpy()

    top10_src_idx  = np.argsort(last_logits_src)[-10:][::-1]
    top10_base_idx = np.argsort(last_logits_base)[-10:][::-1]
    top10_src_tok  = [model.tokenizer.decode(i) for i in top10_src_idx]
    top10_base_tok = [model.tokenizer.decode(i) for i in top10_base_idx]

    print(f"\n[Output] Top-10 tokens promoted at last pos (SOURCE): "
          f"{list(zip(top10_src_tok, [f'{last_logits_src[i]:.2f}' for i in top10_src_idx]))}")
    print(f"[Output] Top-10 tokens promoted at last pos (BASE):   "
          f"{list(zip(top10_base_tok, [f'{last_logits_base[i]:.2f}' for i in top10_base_idx]))}")

    # Direct check: does this head promote the correct answer token at the last position?
    src_score_src   = last_logits_src[src_id]
    base_score_src  = last_logits_src[base_id]
    src_score_base  = last_logits_base[src_id]
    base_score_base = last_logits_base[base_id]
    print(f"\n[Output] At last token — does head promote correct answer?")
    print(f"  SOURCE run: logit[{exp['source_answer']!r}]={src_score_src:.3f}  "
          f"logit[{exp['base_answer']!r}]={base_score_src:.3f}  "
          f"→ {'CORRECT ✓' if src_score_src > base_score_src else 'wrong ✗'}")
    print(f"  BASE   run: logit[{exp['source_answer']!r}]={src_score_base:.3f}  "
          f"logit[{exp['base_answer']!r}]={base_score_base:.3f}  "
          f"→ {'correct' if src_score_base > base_score_base else 'wrong or indifferent'}")

    # Output norm: where does this head write strongly?
    out_norm_src  = head_out_src.norm(dim=-1).detach().numpy()    # [seq]
    out_norm_base = head_out_base.norm(dim=-1).detach().numpy()

    # Plot
    fig, axes = plt.subplots(3, 1, figsize=(max(14, seq_len * 0.38), 11))

    # Row 1: logit-diff contribution per position
    ax = axes[0]
    x  = np.arange(seq_len)
    ax.bar(x - 0.2, ld_contrib_src,  width=0.4, color="steelblue", alpha=0.85, label="SOURCE")
    ax.bar(x + 0.2, ld_contrib_base, width=0.4, color="firebrick",  alpha=0.85, label="BASE")
    ax.axhline(0, color="black", lw=0.5)
    for p in diff_pos:
        ax.axvspan(p - 0.5, p + 0.5, alpha=0.15, color="orange")
    ax.set_xticks(range(seq_len))
    ax.set_xticklabels(tok_strs, rotation=90, fontsize=5)
    ax.set_ylabel("Logit-diff contribution")
    ax.set_title(
        f"WHAT does L{LAYER}H{HEAD} write? — logit-diff contrib per position\n"
        f"Positive = promotes {exp['source_answer']!r} over {exp['base_answer']!r}  "
        f"(approx., no LN)")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3, axis="y")

    # Row 2: output norm per position — WHERE does the head write strongly?
    ax = axes[1]
    ax.plot(out_norm_src,  color="steelblue", lw=1.5, marker="o", ms=3, label="SOURCE")
    ax.plot(out_norm_base, color="firebrick",  lw=1.5, marker="o", ms=3, label="BASE")
    for p in diff_pos:
        ax.axvspan(p - 0.5, p + 0.5, alpha=0.15, color="orange")
    ax.set_xticks(range(seq_len))
    ax.set_xticklabels(tok_strs, rotation=90, fontsize=5)
    ax.set_ylabel("||head output||")
    ax.set_title(f"WHERE does L{LAYER}H{HEAD} write? — output norm per position")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # Row 3: top-10 tokens at last position (SOURCE)
    ax = axes[2]
    vals_src_bar  = last_logits_src[top10_src_idx]
    vals_base_bar = last_logits_base[top10_src_idx]
    x10 = np.arange(10)
    ax.bar(x10 - 0.2, vals_src_bar,  width=0.4, color="steelblue", alpha=0.85, label="SOURCE")
    ax.bar(x10 + 0.2, vals_base_bar, width=0.4, color="firebrick",  alpha=0.85, label="BASE")
    ax.set_xticks(x10)
    ax.set_xticklabels([repr(t) for t in top10_src_tok], rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("Raw logit score (approx.)")
    ax.set_title(
        f"Top-10 tokens promoted at last position (sorted by SOURCE)\n"
        f"Hypothesis check: should include {exp['source_answer']!r}")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle(
        f"Head Output Semantics: L{LAYER}H{HEAD} [{args.model} {args.setup}]\n"
        f"src={exp['source_answer']!r}  base={exp['base_answer']!r}",
        fontsize=11)
    plt.tight_layout()
    plt.savefig(f"{outdir}/2_output.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  → Saved 2_output.png")

    # ── 3. Upstream: residual decomp ─────────────────────────────────────────
    # First, quick phase0: which component (Q or V) carries more of the binding signal?
    # We patch Q or V of this head from SOURCE into the BASE run and measure the metric.
    # Whichever gives a higher score "carries" the signal for this head.

    def patch_component(comp):
        """Patch one component of L{LAYER}H{HEAD} from SOURCE into BASE, return metric."""
        hook_name = tl_utils.get_act_name(comp, LAYER)
        def hook_fn(act, hook):
            if comp in ("k", "v"):
                # GQA: K and V indexed by kv_head, not full head
                kv_h = HEAD // GQA
                act[:, :, kv_h] = src_cache[hook.name][:, :, kv_h]
            else:   # q, z: indexed by full head
                act[:, :, HEAD] = src_cache[hook.name][:, :, HEAD]
            return act
        with torch.no_grad():
            logits = model.run_with_hooks(base_tok, fwd_hooks=[(hook_name, hook_fn)],
                                          return_type="logits")
        return metric(logits).item()

    q_score = patch_component("q")
    v_score = patch_component("v")
    z_score = patch_component("z")   # full head output — sanity check, should be ~1

    print(f"\n[Phase0] Patching L{LAYER}H{HEAD} from SOURCE into BASE:")
    print(f"  Q score = {q_score:.4f}  (high → binding signal flows through Q)")
    print(f"  V score = {v_score:.4f}  (high → binding signal flows through V)")
    print(f"  Z score = {z_score:.4f}  (full head output; ~1 if this head carries all signal)")
    DOMINANT = "Q" if q_score >= v_score else "V"
    print(f"  → dominant: {DOMINANT}")

    # Residual decomposition into the dominant component's projection matrix.
    # For Q: which earlier heads write into the residual stream such that,
    #        projected through W_Q, it aligns with the binding direction in Q-space?
    # For V: same but through W_V, at the most-attended position.
    #
    # Binding direction = unit vector of (src - base) in Q/V space of this head.
    # Score for earlier head (l, h) = (delta_out @ W_proj) · bind_dir
    # where delta_out = head_l_h_src_out - head_l_h_base_out  at proj_pos.

    if DOMINANT == "Q":
        W_proj   = model.W_Q[LAYER, HEAD].float().cpu()        # [d_model, d_head]
        proj_pos = seq_len - 1                                  # Q reads from last token
        dir_src  = src_cache["q",  LAYER][0, proj_pos, HEAD, :].float().cpu()
        dir_base = base_cache["q", LAYER][0, proj_pos, HEAD, :].float().cpu()
    else:
        kv_head  = HEAD // GQA
        W_proj   = model.W_V[LAYER, kv_head].float().cpu()     # [d_model, d_head]
        # V reads from positions attended-to from the last token
        attn_last = src_cache["pattern", LAYER][0, HEAD, -1, :].cpu()
        proj_pos  = int(attn_last.argmax().item())
        dir_src   = src_cache["v",  LAYER][0, proj_pos, kv_head, :].float().cpu()
        dir_base  = base_cache["v", LAYER][0, proj_pos, kv_head, :].float().cpu()

    bind_dir = (dir_src - dir_base)
    bind_dir_norm = bind_dir.norm().item()
    bind_dir = bind_dir / (bind_dir_norm + 1e-8)

    print(f"\n[Upstream] Residual decomp in {DOMINANT}-space")
    print(f"  projection position: {proj_pos} = {repr(tok_strs[proj_pos])}")
    print(f"  binding direction norm (pre-normalise): {bind_dir_norm:.4f}")
    print(f"  (larger norm → bigger src/base gap in {DOMINANT}-space → cleaner signal)")

    head_scores = np.zeros((LAYER, NH))
    mlp_scores  = np.zeros(LAYER)

    for l in range(LAYER):
        # head (l, h) output at proj_pos: z[l, proj_pos, h] @ W_O[l, h] → [d_model]
        z_l_src  = src_cache["z",  l][0, proj_pos, :, :].float().cpu()   # [n_heads, d_head]
        z_l_base = base_cache["z", l][0, proj_pos, :, :].float().cpu()
        W_O_l    = model.W_O[l].float().cpu()                             # [n_heads, d_head, d_model]
        for h in range(NH):
            delta = (z_l_src[h] @ W_O_l[h]) - (z_l_base[h] @ W_O_l[h])  # [d_model]
            # Project through W_proj of current head, dot with binding direction
            head_scores[l, h] = (delta @ W_proj).dot(bind_dir).item()
        # MLP
        mlp_delta = (src_cache["mlp_out",  l][0, proj_pos, :].float().cpu() -
                     base_cache["mlp_out", l][0, proj_pos, :].float().cpu())
        mlp_scores[l] = (mlp_delta @ W_proj).dot(bind_dir).item()

    # Print top contributors — tells us WHO sets up the binding signal
    flat = head_scores.flatten()
    print(f"\n[Upstream] Top-5 heads writing into L{LAYER}H{HEAD}.{DOMINANT}:")
    for idx in np.argsort(np.abs(flat))[-5:][::-1]:
        sl, sh = divmod(int(idx), NH)
        print(f"  L{sl}H{sh}: {flat[idx]:+.4f}")
    print(f"\n[Upstream] Top-3 MLPs writing into L{LAYER}H{HEAD}.{DOMINANT}:")
    for l in np.argsort(np.abs(mlp_scores))[-3:][::-1]:
        print(f"  L{int(l)} MLP: {mlp_scores[int(l)]:+.4f}")

    # Plot upstream heatmap + MLP bar
    vmax_h = max(abs(head_scores).max(), 0.01)
    vmax_m = max(abs(mlp_scores).max(),  0.01)
    fig, axes = plt.subplots(1, 2, figsize=(NH * 0.45, LAYER * 0.35 + 1.5),
                             gridspec_kw={"width_ratios": [5, 1]})

    ax = axes[0]
    im = ax.imshow(head_scores, aspect="auto", cmap="RdBu_r", origin="lower",
                   vmin=-vmax_h, vmax=vmax_h)
    plt.colorbar(im, ax=ax,
                 label=f"Contribution to {DOMINANT}-space binding dir of L{LAYER}H{HEAD}")
    ax.set_xlabel("Head"); ax.set_ylabel("Layer")
    ax.set_title(
        f"Earlier heads → L{LAYER}H{HEAD}.{DOMINANT}\n"
        f"proj_pos={proj_pos} ({repr(tok_strs[proj_pos])})  lime=top-5 by |score|")
    ax.set_xticks(range(NH)); ax.set_yticks(range(LAYER))
    for idx in np.argsort(np.abs(flat))[-5:]:
        sl, sh = divmod(int(idx), NH)
        ax.add_patch(plt.Rectangle((sh-0.5, sl-0.5), 1, 1,
                                    fill=False, edgecolor="lime", lw=2))

    ax = axes[1]
    colors = ["steelblue" if v >= 0 else "firebrick" for v in mlp_scores]
    ax.barh(range(LAYER), mlp_scores, color=colors)
    ax.axvline(0, color="black", lw=0.5)
    ax.set_yticks(range(LAYER))
    ax.set_yticklabels([f"L{l}" for l in range(LAYER)], fontsize=6)
    ax.set_xlabel("MLP contribution")
    ax.set_xlim(-vmax_m * 1.3, vmax_m * 1.3)
    ax.set_title("MLPs")
    ax.grid(True, alpha=0.3, axis="x")

    fig.suptitle(
        f"Upstream Attribution: WHO writes into L{LAYER}H{HEAD}.{DOMINANT}?\n"
        f"[{args.model} {args.setup}]  "
        f"(Q score={q_score:.3f}  V score={v_score:.3f})",
        fontsize=11)
    plt.tight_layout()
    plt.savefig(f"{outdir}/3_upstream.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  → Saved 3_upstream.png")
    print(f"\n  All plots saved to {outdir}/")

print(f"\n{'='*65}")
print(f"All done. Results: {out_base}/")
print(f"{'='*65}")
