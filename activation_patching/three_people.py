import random
import torch
import numpy as np
import matplotlib.pyplot as plt
from itertools import permutations

random.seed(12)
torch.manual_seed(12)
torch.cuda.manual_seed(12)

def rindex(lst, value):
    """get the rightmost index of a value in a list."""
    return len(lst) - 1 - lst[::-1].index(value)

# load model
import nnsight
model = nnsight.LanguageModel("meta-llama/Llama-3.1-8B", device_map="auto")

PROMPT_TEMPLATE = """This is the transcript of a conversation.
"I am Alice."
"I am Bob."
"I like basketball."
"I heard that Claire likes basketball too!"
"Hi Alice! Yes, I like basketball. But I like tennis more!"
"Fun! What do you like?"
"I like soccer."
Question: {question} Answer:"""

# Define all prompts with their expected answers
prompts = {
    "Alice": PROMPT_TEMPLATE.format(question="Who likes basketball?"),
    "Bob": PROMPT_TEMPLATE.format(question="Who likes soccer?"),
    "Claire": PROMPT_TEMPLATE.format(question="Who likes tennis?"),
}

# Token IDs
token_ids = {
    "Alice": model.tokenizer(" Alice").input_ids[1],
    "Bob": model.tokenizer(" Bob").input_ids[1],
    "Claire": model.tokenizer(" Claire").input_ids[1],
}

print("Expected token IDs:", token_ids)

# Verify model produces correct answers and get activations
all_activations = {}
for name, prompt in prompts.items():
    activations = []
    with torch.no_grad():
        with model.trace(prompt):
            for layer in model.model.layers:
                activations.append(layer.output[0].save())
            logits = model.output.logits.save()
    
    predicted_token_id = logits.argmax(dim=-1)[0, -1].item()
    predicted_token = model.tokenizer.decode(predicted_token_id)
    expected_token_id = token_ids[name]
    
    print(f"{name}: predicted={predicted_token!r} (id={predicted_token_id}), expected id={expected_token_id}")
    
    # Assert model produces correct answer
    assert predicted_token_id == expected_token_id, \
        f"Model prediction mismatch for {name}! " \
        f"Expected token id {expected_token_id} (' {name}'), " \
        f"but got {predicted_token_id} ({predicted_token!r})"
    
    all_activations[name] = activations

print("\n✓ All model predictions match expected answers. Proceeding with patching.\n")
for name, prompt in prompts.items():
    with torch.no_grad():
        with model.trace(prompt):
            logits = model.output.logits.save()
    probs = logits[0, -1].softmax(dim=-1)
    print(f"\n{name} prompt:")
    for n, tid in token_ids.items():
        print(f"  P({n}) = {probs[tid].item():.4f}")

# Check activation shape
is_2d = len(all_activations["Alice"][0].shape) == 2
print(f"Activation shape: {all_activations['Alice'][0].shape}, is_2d: {is_2d}")

# Get token indices for patching
source_prompt_ids = model.tokenizer(prompts["Alice"]).input_ids
question_token_id = model.tokenizer('Question').input_ids[1]
last_example_index = rindex(source_prompt_ids, question_token_id)

print("last_example_index:", last_example_index)

from tqdm import trange

# Pairwise patching: source -> base for all pairs
pairs = list(permutations(prompts.keys(), 2))
print(f"Pairs to patch: {pairs}")

all_results = {}

for source_name, base_name in pairs:
    source_activations = all_activations[source_name]
    base_prompt = prompts[base_name]
    source_token_id = token_ids[source_name]
    base_token_id = token_ids[base_name]
    
    patching_results = []
    
    for layer_index in trange(model.config.num_hidden_layers, desc=f"{source_name}→{base_name}"):
        patching_per_layer = []
        for token_index in range(last_example_index, len(source_prompt_ids)):
            with torch.no_grad():
                with model.trace(base_prompt):
                    if is_2d:
                        model.model.layers[layer_index].output[0][token_index, :] = source_activations[layer_index][token_index, :]
                    else:
                        model.model.layers[layer_index].output[0][0, token_index, :] = source_activations[layer_index][0, token_index, :]
                    
                    patched_logits = model.output.logits.save()
                
                patched_probs = patched_logits[:, -1].softmax(dim=-1)
                source_prob = patched_probs[0, source_token_id].item()
                base_prob = patched_probs[0, base_token_id].item()
                patched_diff = source_prob - base_prob
                patching_per_layer.append(patched_diff)
        
        patching_results.append(patching_per_layer)
    
    all_results[(source_name, base_name)] = np.array(patching_results, dtype=float)

# Plot all pairs in a grid
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()

# Token strings for x-axis
base_token_ids = model.tokenizer(prompts["Alice"]).input_ids
token_strings = [
    model.tokenizer.decode(base_token_ids[t])
    for t in range(last_example_index, len(base_token_ids))
]

for idx, ((source_name, base_name), results) in enumerate(all_results.items()):
    ax = axes[idx]
    im = ax.imshow(results, aspect='auto', cmap='RdBu_r', origin='lower', 
                   vmin=-1, vmax=1)
    ax.set_title(f'{source_name} → {base_name}\nP({source_name}) - P({base_name})')
    ax.set_xticks(range(len(token_strings)))
    ax.set_xticklabels(token_strings, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Layers')
    ax.set_xlabel('Tokens')

plt.colorbar(im, ax=axes, label='Probability Difference', shrink=0.8)
plt.suptitle('Pairwise Activation Patching', fontsize=14)
plt.tight_layout()
plt.savefig("activation_patching_pairwise.png", dpi=150)
plt.close()

print("Done!")