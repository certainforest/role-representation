import random
import torch
import numpy as np
import matplotlib.pyplot as plt

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

# Load model
import nnsight
model = nnsight.LanguageModel("meta-llama/Llama-3.1-8B", device_map="auto")

# Prompts: Different conversations, SAME question
PROMPT_SOURCE = """This is the transcript of a conversation.
"I am Alice."
"I am Bob."
"I like basketball."
"I heard that Claire likes basketball too!"
"Hi Alice! Yes, I like basketball. But I like tennis more!"
"Fun! What do you like, Bob?"
"I like soccer."
Question: Who likes tennis? Answer:"""  # Claire

PROMPT_BASE = """This is the transcript of a conversation.
"I am Alice."
"I am Bob."
"I like basketball."
"I heard that Claire likes basketball too!"
"Hi Alice! Yes, I like basketball. But I like soccer more!"
"Fun! What do you like, Bob?"
"I like tennis."
Question: Who likes tennis? Answer:"""  # Bob

# Token IDs
token_ids = {
    "Claire": model.tokenizer(" Claire").input_ids[1],
    "Bob": model.tokenizer(" Bob").input_ids[1],
}
print(f"Token IDs: {token_ids}")

# Parse tokens
source_prompt_ids = model.tokenizer(PROMPT_SOURCE).input_ids
base_prompt_ids = model.tokenizer(PROMPT_BASE).input_ids

assert len(source_prompt_ids) == len(base_prompt_ids), \
    f"Prompt lengths differ: {len(source_prompt_ids)} vs {len(base_prompt_ids)}"

# Find first differing position
first_diff_index = find_first_diff(source_prompt_ids, base_prompt_ids)
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
    marker = " ***" if s_id != b_id else ""
    if s_id == b_id:
        print(f"{i:3d}: {s_tok!r}{marker}")
    else:
        print(f"{i:3d}: {s_tok!r} vs {b_tok!r}{marker}")

# Get source activations and logits
print("\n" + "=" * 60)
print("Getting activations...")
print("=" * 60)

source_activations = []
with torch.no_grad():
    with model.trace(PROMPT_SOURCE) as tracer:
        for layer in model.model.layers:
            source_activations.append(layer.output[0].save())
        source_logits = model.output.logits.save()

source_pred_id = source_logits.argmax(dim=-1)[0, -1].item()
source_pred = model.tokenizer.decode(source_pred_id)
print(f"Source (Claire=tennis): {source_pred!r}")

base_activations = []
with torch.no_grad():
    with model.trace(PROMPT_BASE) as tracer:
        for layer in model.model.layers:
            base_activations.append(layer.output[0].save())
        base_logits = model.output.logits.save()

base_pred_id = base_logits.argmax(dim=-1)[0, -1].item()
base_pred = model.tokenizer.decode(base_pred_id)
print(f"Base (Bob=tennis):     {base_pred!r}")

# Confidence breakdown
print("\n" + "=" * 60)
print("Confidence:")
print("=" * 60)
source_probs = source_logits[0, -1].softmax(dim=-1)
base_probs = base_logits[0, -1].softmax(dim=-1)

print(f"Source: P(Claire)={source_probs[token_ids['Claire']].item():.4f}, P(Bob)={source_probs[token_ids['Bob']].item():.4f}")
print(f"Base:   P(Claire)={base_probs[token_ids['Claire']].item():.4f}, P(Bob)={base_probs[token_ids['Bob']].item():.4f}")

# Check tracking behavior
print("\n" + "=" * 60)
if source_pred_id == token_ids["Claire"] and base_pred_id == token_ids["Bob"]:
    print("✓ Model correctly tracks conversation!")
elif source_pred_id == token_ids["Claire"] and base_pred_id == token_ids["Claire"]:
    print("✗ Model uses SHORTCUT: Always says Claire")
else:
    print(f"? Unexpected: source={source_pred}, base={base_pred}")
print("=" * 60)

# Check shape
is_2d = len(source_activations[0].shape) == 2
print(f"\nActivation shape: {source_activations[0].shape}, is_2d: {is_2d}")

# Patching
from tqdm import trange

claire_token_id = token_ids["Claire"]
bob_token_id = token_ids["Bob"]

patching_results = []
for layer_index in trange(model.config.num_hidden_layers, desc="Patching"):
    patching_per_layer = []
    for token_index in range(first_diff_index,len(source_prompt_ids)):
        with torch.no_grad():
            with model.trace(PROMPT_BASE) as tracer:
                if is_2d:
                    model.model.layers[layer_index].output[0][token_index, :] = source_activations[layer_index][token_index, :]
                else:
                    model.model.layers[layer_index].output[0][0, token_index, :] = source_activations[layer_index][0, token_index, :]
                patched_logits = model.output.logits.save()
            
            patched_probs = patched_logits[:, -1].softmax(dim=-1)
            claire_prob = patched_probs[0, claire_token_id].item()
            bob_prob = patched_probs[0, bob_token_id].item()
            patching_per_layer.append(claire_prob - bob_prob)
    
    patching_results.append(patching_per_layer)

patching_results = np.array(patching_results, dtype=float)
print(f"Patching results shape: {patching_results.shape}")

# Token strings for x-axis
token_strings = [model.tokenizer.decode(tok_id) for tok_id in base_prompt_ids][first_diff_index:]

# Find question position
question_token_id = model.tokenizer('Question').input_ids[1]
question_pos = rindex(base_prompt_ids[first_diff_index:], question_token_id) + first_diff_index
# Plot
plt.figure(figsize=(10, 8))
plt.imshow(patching_results, aspect='auto', cmap='RdBu_r', origin='lower', vmin=-0.5, vmax=0.5)
plt.colorbar(label='P(Claire) - P(Bob)')

# Mark first diff and question boundary
plt.axvline(x=question_pos - first_diff_index - 0.5, color='green', linestyle='--', linewidth=2, label='Question starts')

plt.xticks(ticks=range(len(token_strings)), labels=token_strings, rotation=90, ha='center', fontsize=6)
plt.yticks(ticks=range(0, patching_results.shape[0], 2), labels=range(0, patching_results.shape[0], 2))
plt.xlabel('Tokens')
plt.ylabel('Layers')
plt.title('Activation Patching: Source (Claire=tennis) → Base (Bob=tennis)\nOrange = first diff | Green = question | Red = more Claire, Blue = more Bob')
plt.legend(loc='upper left')

plt.tight_layout()
plt.savefig("activation_patching_binding_tennis.png", dpi=150)
plt.close()

print("\nSaved: activation_patching_binding_tennis.png")