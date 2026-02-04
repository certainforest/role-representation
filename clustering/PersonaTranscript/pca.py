import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import nnsight

# ====================
# CONFIGURE HERE
# ====================
SPEAKERS = ["Alice", "Bob"]
# SPEAKERS = ["OBAMA", "MCCAIN", "LEHRER"]

# TRANSCRIPT_PATH = "2008debate_clean.txt"
TRANSCRIPT_PATH = "transcript1_funny.txt"
MODEL_NAME = "meta-llama/Llama-3.1-8B"

# Colors will cycle if you have many speakers
COLORS = ["blue", "red", "green", "orange", "purple", "brown", "pink", "gray"]

# ====================
# Setup
# ====================
SPEAKER_TO_ID = {s: i for i, s in enumerate(SPEAKERS)}

model = nnsight.LanguageModel(MODEL_NAME, device_map="auto")
tokenizer = model.tokenizer

# ====================
# Load and parse transcript
# ====================
with open(TRANSCRIPT_PATH) as f:
    lines = f.readlines()

full_text = ""
speaker_per_char = []

for line in lines:
    line = line.strip()
    if not line:
        continue
    
    # Find which speaker this line belongs to
    matched_speaker = None
    for speaker in SPEAKERS:
        if line.startswith(f"{speaker}:"):
            matched_speaker = speaker
            content = line[len(speaker)+1:].strip()
            break
    
    if matched_speaker is None:
        continue
    
    speaker_id = SPEAKER_TO_ID[matched_speaker]
    text = content + "\n"
    full_text += text
    speaker_per_char += [speaker_id] * len(text)

print(f"Total characters: {len(full_text)}")
print(f"Characters per speaker: {[(s, speaker_per_char.count(i)) for s, i in SPEAKER_TO_ID.items()]}")

# ====================
# Tokenize and map tokens → speaker
# ====================
enc = tokenizer(full_text, return_offsets_mapping=True)
input_ids = enc["input_ids"]
offsets = enc["offset_mapping"]

# Filter valid offsets (skip special tokens with (0,0))
valid_mask = [end > 0 for start, end in offsets]
valid_offsets = [off for off, v in zip(offsets, valid_mask) if v]
valid_input_ids = [tid for tid, v in zip(input_ids, valid_mask) if v]

speaker_per_token = np.array([speaker_per_char[start] for start, end in valid_offsets])

# Filter out special tokens
special_tokens = {tokenizer.bos_token_id, tokenizer.eos_token_id, tokenizer.pad_token_id}
special_tokens.discard(None)

mask_non_special = np.array([tid not in special_tokens for tid in valid_input_ids])
speaker_filtered = speaker_per_token[mask_non_special]
offsets_filtered = [off for off, keep in zip(valid_offsets, mask_non_special) if keep]

print(f"Tokens after filtering: {len(speaker_filtered)}")

# ====================
# Trace activations
# ====================
layer_outputs = {}

with torch.no_grad():
    with model.trace(full_text):
        for i, layer in enumerate(model.model.layers):
            layer_outputs[i] = layer.output[0].save()

# ====================
# PCA + plotting
# ====================
LAYERS = list(range(0, len(model.model.layers), 3))  # layers to plot
N_COLS = 3  # number of columns in the subplot grid
N_ROWS = (len(LAYERS) + N_COLS - 1) // N_COLS  # auto rows

# Create figure
fig, axes = plt.subplots(N_ROWS, N_COLS, figsize=(5*N_COLS, 5*N_ROWS))
axes = axes.flatten()  # flatten in case of 2D grid

for idx, LAYER in enumerate(LAYERS):
    hidden = layer_outputs[LAYER].detach().cpu().numpy()
    
    if hidden.ndim == 3:
        hidden = hidden[0]  # remove batch dim if present
    
    # Align filtered tokens if you did filtering
    hidden_valid = hidden[valid_mask]
    hidden_filtered = hidden_valid[mask_non_special]

    # Normalize
    hidden_scaled = (hidden_filtered - hidden_filtered.mean(0)) / (hidden_filtered.std(0) + 1e-6)

    # PCA
    proj = PCA(n_components=2).fit_transform(hidden_scaled)

    ax = axes[idx]
    
    for speaker_id, speaker_name in enumerate(SPEAKERS):
        mask = speaker_filtered == speaker_id
        if mask.sum() > 0:
            ax.scatter(
                proj[mask, 0], proj[mask, 1],
                label=f"{speaker_name} ({mask.sum()})",
                alpha=0.5,
                s=10,
                c=COLORS[speaker_id % len(COLORS)]
            )
    
    ax.set_title(f"Layer {LAYER}")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend(fontsize=6)

# Hide any empty subplots
for i in range(len(LAYERS), len(axes)):
    axes[i].axis('off')

plt.tight_layout()
plt.savefig("pca_all_layers_funny.png", dpi=300)
plt.show()
