import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import nnsight

# ====================
# CONFIG
# ====================
SPEAKERS = ["OBAMA", "MCCAIN", "LEHRER"]
TRANSCRIPT_PATH = "2008debate_clean.txt"
MODEL_NAME = "meta-llama/Llama-3.1-8B"
CHUNK_SIZE = 512
COLORS = ["blue", "red", "green", "orange", "purple", "brown", "pink", "gray"]

SPEAKER_TO_ID = {s: i for i, s in enumerate(SPEAKERS)}

# ====================
# Setup
# ====================
model = nnsight.LanguageModel(MODEL_NAME, device_map="auto")
tokenizer = model.tokenizer

# ====================
# Load & parse transcript
# ====================
with open(TRANSCRIPT_PATH) as f:
    lines = f.readlines()

full_text = ""
speaker_per_char = []

for line in lines:
    line = line.strip()
    if not line:
        continue
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

# ====================
# Tokenize
# ====================
enc = tokenizer(full_text, return_offsets_mapping=True)
input_ids = enc["input_ids"]
offsets = enc["offset_mapping"]

# Filter special tokens
special_tokens = {tokenizer.bos_token_id, tokenizer.eos_token_id, tokenizer.pad_token_id}
special_tokens.discard(None)

valid_mask = [(end > 0) and (tid not in special_tokens) 
              for (start, end), tid in zip(offsets, input_ids)]

valid_input_ids = [tid for tid, v in zip(input_ids, valid_mask) if v]
valid_offsets = [off for off, v in zip(offsets, valid_mask) if v]
speaker_per_token = np.array([speaker_per_char[start] for start, end in valid_offsets])
tokens_text = [full_text[start:end] for start, end in valid_offsets]

print(f"Total tokens after filtering: {len(valid_input_ids)}")

# ====================
# Process in chunks
# ====================
def chunk_indices(seq_len, chunk_size):
    for start in range(0, seq_len, chunk_size):
        yield start, min(start + chunk_size, seq_len)

num_layers = len(model.model.layers)
layer_activations = {i: [] for i in range(num_layers)}

num_chunks = (len(valid_input_ids) + CHUNK_SIZE - 1) // CHUNK_SIZE
print(f"Processing {num_chunks} chunks...")

for chunk_idx, (start, end) in enumerate(chunk_indices(len(valid_input_ids), CHUNK_SIZE)):
    chunk_ids = torch.tensor([valid_input_ids[start:end]])  # shape (1, chunk_len)
    
    with torch.no_grad():
        with model.trace(chunk_ids):
            for i, layer in enumerate(model.model.layers):
                # layer.output[0] is hidden states: (batch, seq, hidden)
                layer_activations[i].append(layer.output[0].save())
    
    if (chunk_idx + 1) % 10 == 0:
        print(f"  Processed chunk {chunk_idx + 1}/{num_chunks}")
        torch.cuda.empty_cache()

print("Concatenating activations...")

# ====================
# Concatenate chunks
# ====================
layer_outputs = {}
for LAYER in range(num_layers):
    chunks = layer_activations[LAYER]
    # Each chunk is (1, chunk_len, hidden_dim), squeeze batch dim then concat
    tensors = [c.squeeze(0).detach().cpu() for c in chunks]  # each is (chunk_len, hidden_dim)
    layer_outputs[LAYER] = torch.cat(tensors, dim=0).numpy()  # (total_len, hidden_dim)

# Verify shape
print(f"Layer 0 shape: {layer_outputs[0].shape}")  # should be (num_tokens, hidden_dim)
assert layer_outputs[0].shape[0] == len(valid_input_ids), "Shape mismatch!"

# ====================
# PCA + Plot (all layers in one figure)
# ====================
LAYERS = list(range(0, num_layers, 4))
N_COLS = 3
N_ROWS = (len(LAYERS) + N_COLS - 1) // N_COLS

fig, axes = plt.subplots(N_ROWS, N_COLS, figsize=(5*N_COLS, 5*N_ROWS))
axes = axes.flatten()

for idx, LAYER in enumerate(LAYERS):
    hidden = layer_outputs[LAYER]  # (seq_len, hidden_dim)
    hidden_scaled = (hidden - hidden.mean(0)) / (hidden.std(0) + 1e-6)
    
    proj = PCA(n_components=2).fit_transform(hidden_scaled)
    
    ax = axes[idx]
    for speaker_id, speaker_name in enumerate(SPEAKERS):
        mask = speaker_per_token == speaker_id
        if mask.sum() > 0:
            ax.scatter(
                proj[mask, 0], proj[mask, 1],
                label=f"{speaker_name} ({mask.sum()})",
                alpha=0.5, s=10,
                c=COLORS[speaker_id % len(COLORS)]
            )
    
    ax.set_title(f"Layer {LAYER}")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend(fontsize=6)

# Hide empty subplots
for i in range(len(LAYERS), len(axes)):
    axes[i].axis('off')

plt.tight_layout()
plt.savefig("pca_all_layers.png", dpi=300)
plt.show()

print("✅ Done!")