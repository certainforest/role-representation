import random
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

random.seed(12)
torch.manual_seed(12)
torch.cuda.manual_seed(12)

def rindex(lst, value):
    return len(lst) - 1 - lst[::-1].index(value)

def find_first_diff(list1, list2):
    """Find the first index where two lists differ."""
    for i, (a, b) in enumerate(zip(list1, list2)):
        if a != b:
            return i
    return min(len(list1), len(list2))

# ══════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════

MODEL_NAME = "Qwen/Qwen3-8B"#"meta-llama/Llama-3.1-8B"

# Source prompt: provides the activations we patch FROM
PROMPT_SOURCE = """This is the transcript of a conversation.
"I am Alice."
"I am Bob."
"I live in France."
"I live in Thailand."
Question: Who lives in France? Answer:"""  # Alice

# Base prompt: the prompt we patch INTO (run with patched activations)
PROMPT_BASE = """This is the transcript of a conversation.
"I am Alice."
"I am Bob."
"I live in Thailand."
"I live in France."
Question: Who lives in France? Answer:"""  # Bob (swapped)

# Tokens to measure: P(source_answer) - P(base_answer)
SOURCE_ANSWER = "Alice"
BASE_ANSWER = "Bob"

OUTPUT_FILE = f"{MODEL_NAME.replace('/', '_')}_activation_patching_binding_person_country_1.png"

# ══════════════════════════════════════════════════════════════

import nnsight
model = nnsight.LanguageModel(MODEL_NAME, device_map="auto")

# Token IDs for the answers (add_special_tokens=False works for both Llama and Qwen)
source_answer_id = model.tokenizer.encode(f" {SOURCE_ANSWER}", add_special_tokens=False)[0]
base_answer_id = model.tokenizer.encode(f" {BASE_ANSWER}", add_special_tokens=False)[0]
print(f"Token IDs: {SOURCE_ANSWER}={source_answer_id}, {BASE_ANSWER}={base_answer_id}")

# Parse tokens
source_prompt_ids = model.tokenizer(PROMPT_SOURCE).input_ids
base_prompt_ids = model.tokenizer(PROMPT_BASE).input_ids

assert len(source_prompt_ids) == len(base_prompt_ids), \
    f"Prompt lengths differ: {len(source_prompt_ids)} vs {len(base_prompt_ids)}"

# Find differing positions
first_diff_index = find_first_diff(source_prompt_ids, base_prompt_ids)
diff_positions = set()
for i, (s_id, b_id) in enumerate(zip(source_prompt_ids, base_prompt_ids)):
    if s_id != b_id:
        diff_positions.add(i)

print(f"\nFirst differing token at index {first_diff_index}:")
print(f"  Source: {model.tokenizer.decode(source_prompt_ids[first_diff_index])!r}")
print(f"  Base:   {model.tokenizer.decode(base_prompt_ids[first_diff_index])!r}")

# Print all tokens with diff markers
print("\n" + "=" * 60)
print("TOKENS (*** = differs):")
print("=" * 60)
for i, (s_id, b_id) in enumerate(zip(source_prompt_ids, base_prompt_ids)):
    s_tok = model.tokenizer.decode(s_id)
    b_tok = model.tokenizer.decode(b_id)
    if s_id != b_id:
        print(f"{i:3d}: {s_tok!r} vs {b_tok!r} ***")
    else:
        print(f"{i:3d}: {s_tok!r}")

# ══════════════════════════════════════════════════════════════
# Get activations
# ══════════════════════════════════════════════════════════════
print("\nGetting activations...")

source_activations = []
with torch.no_grad():
    with model.trace(PROMPT_SOURCE) as tracer:
        for layer in model.model.layers:
            source_activations.append(layer.output[0].save())
        source_logits = model.output.logits.save()

source_pred_id = source_logits.argmax(dim=-1)[0, -1].item()
source_pred = model.tokenizer.decode(source_pred_id)
print(f"Source prediction: {source_pred!r}")
assert source_pred_id == source_answer_id, \
    f"Source prompt should predict '{SOURCE_ANSWER}' but got {source_pred!r} — patching would be meaningless"

base_activations = []
with torch.no_grad():
    with model.trace(PROMPT_BASE) as tracer:
        for layer in model.model.layers:
            base_activations.append(layer.output[0].save())
        base_logits = model.output.logits.save()

base_pred_id = base_logits.argmax(dim=-1)[0, -1].item()
base_pred = model.tokenizer.decode(base_pred_id)
print(f"Base prediction:   {base_pred!r}")
assert base_pred_id == base_answer_id, \
    f"Base prompt should predict '{BASE_ANSWER}' but got {base_pred!r} — patching would be meaningless"

# Confidence
source_probs = source_logits[0, -1].softmax(dim=-1)
base_probs = base_logits[0, -1].softmax(dim=-1)
print(f"\nSource: P({SOURCE_ANSWER})={source_probs[source_answer_id].item():.4f}, "
      f"P({BASE_ANSWER})={source_probs[base_answer_id].item():.4f}")
print(f"Base:   P({SOURCE_ANSWER})={base_probs[source_answer_id].item():.4f}, "
      f"P({BASE_ANSWER})={base_probs[base_answer_id].item():.4f}")

# Check shape
is_2d = len(source_activations[0].shape) == 2
print(f"\nActivation shape: {source_activations[0].shape}, is_2d: {is_2d}")

# ══════════════════════════════════════════════════════════════
# Patching
# ══════════════════════════════════════════════════════════════
from tqdm import trange

patching_results = []
for layer_index in trange(model.config.num_hidden_layers, desc="Patching"):
    patching_per_layer = []
    for token_index in range(first_diff_index, len(source_prompt_ids)):
        with torch.no_grad():
            with model.trace(PROMPT_BASE) as tracer:
                if is_2d:
                    model.model.layers[layer_index].output[0][token_index, :] = \
                        source_activations[layer_index][token_index, :]
                else:
                    model.model.layers[layer_index].output[0][0, token_index, :] = \
                        source_activations[layer_index][0, token_index, :]
                patched_logits = model.output.logits.save()

            patched_probs = patched_logits[:, -1].softmax(dim=-1)
            src_prob = patched_probs[0, source_answer_id].item()
            base_prob = patched_probs[0, base_answer_id].item()
            patching_per_layer.append(src_prob - base_prob)

    patching_results.append(patching_per_layer)

patching_results = np.array(patching_results, dtype=float)
print(f"Patching results shape: {patching_results.shape}")

# ══════════════════════════════════════════════════════════════
# Build token labels for x-axis (highlight diffs)
# ══════════════════════════════════════════════════════════════
token_strings = []
for i in range(first_diff_index, len(base_prompt_ids)):
    base_tok = model.tokenizer.decode(base_prompt_ids[i]).replace("\n", "\\n")
    if i in diff_positions:
        src_tok = model.tokenizer.decode(source_prompt_ids[i]).replace("\n", "\\n")
        token_strings.append(f"{base_tok} [{src_tok}]")
    else:
        token_strings.append(base_tok)

# Find completion boundary: first token after the last '"\n' in the prompt
import re
match = list(re.finditer(r'"\n', PROMPT_BASE))
assert match, "No closing quote + newline found in PROMPT_BASE"
completion_char_pos = match[-1].end()
# Tokenize up to that point to find the token index
prefix_tokens = model.tokenizer(PROMPT_BASE[:completion_char_pos]).input_ids
question_rel_pos = len(prefix_tokens) - first_diff_index

# ══════════════════════════════════════════════════════════════
# Plot
# ══════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(13, 10))
gs = gridspec.GridSpec(2, 1, height_ratios=[3.5, 1], hspace=0.35)

# --- Heatmap ---
ax = fig.add_subplot(gs[0])
im = ax.imshow(patching_results, aspect="auto", cmap="RdBu_r",
               origin="lower", vmin=-0.5, vmax=0.5)
fig.colorbar(im, ax=ax, label=f"P({SOURCE_ANSWER}) - P({BASE_ANSWER})")

# Vertical line at question boundary
ax.axvline(x=question_rel_pos - 0.5, color="green", linestyle="--",
           linewidth=2, label="Question starts")


ax.set_xticks(range(len(token_strings)))
xlabels = ax.set_xticklabels(token_strings, rotation=90, ha="center", fontsize=6)
for i, abs_pos in enumerate(range(first_diff_index, len(base_prompt_ids))):
    if abs_pos in diff_positions:
        xlabels[i].set_color("red")
        xlabels[i].set_fontweight("bold")

ax.set_yticks(range(0, patching_results.shape[0], 2))
ax.set_yticklabels(range(0, patching_results.shape[0], 2))
ax.set_xlabel("Token position (patched one at a time)")
ax.set_ylabel("Layer")
ax.set_title(
    f"Activation Patching: replace one (layer, token) in BASE with SOURCE activation\n"
    f"Red = shifted toward \"{SOURCE_ANSWER}\" | Blue = shifted toward \"{BASE_ANSWER}\" | "
    f"Orange highlight = tokens that differ"
)
ax.legend(loc="upper left", fontsize=8)

# --- Prompt text panel ---
ax_text = fig.add_subplot(gs[1])
ax_text.axis("off")

source_display = PROMPT_SOURCE.strip().replace("\n", "\n    ")
base_display = PROMPT_BASE.strip().replace("\n", "\n    ")

prompt_text = (
    f"SOURCE (activations taken FROM this prompt):          prediction: {source_pred!r}\n"
    f"    {source_display}\n\n"
    f"BASE (activations patched INTO this prompt):          prediction: {base_pred!r}\n"
    f"    {base_display}"
)

ax_text.text(
    0.02, 0.95, prompt_text,
    transform=ax_text.transAxes,
    fontsize=7, verticalalignment="top", fontfamily="monospace",
    bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8),
)

plt.savefig(OUTPUT_FILE, dpi=150, bbox_inches="tight")
plt.close()

print(f"\nSaved: {OUTPUT_FILE}")
