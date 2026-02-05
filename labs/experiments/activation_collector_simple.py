"""
Simple Activation Collector for Demo
Truncates inputs to fit in memory and uses CPU
"""

import torch
import h5py
import numpy as np
from pathlib import Path
from transformers import AutoModel, AutoTokenizer
import json
import sys

def collect_activations_simple(condition_dir, condition, model_name, output_file, max_length=512):
    """
    Simplified activation collection that truncates input and uses CPU.

    Args:
        condition_dir: Directory containing condition files
        condition: "labeled", "unlabeled", or "corrupted"
        model_name: HuggingFace model name
        output_file: Where to save HDF5
        max_length: Maximum sequence length (default 512 for memory constraints)
    """
    print(f"\n{'='*70}")
    print("SIMPLE ACTIVATION COLLECTOR")
    print(f"{'='*70}")

    # Load condition text
    condition_path = Path(condition_dir) / f"{condition}.txt"
    with open(condition_path, 'r') as f:
        text = f.read()

    # Load metadata
    metadata_path = Path(condition_dir) / "token_mapping.json"
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    transcript_id = metadata['transcript_id']
    speaker_spans = metadata['speaker_spans']

    print(f"\nTranscript: {transcript_id}")
    print(f"Condition: {condition}")
    print(f"Model: {model_name}")
    print(f"Original text length: {len(text)} chars")

    # Load model and tokenizer
    print("\nLoading model...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()
    model.to('cpu')  # Use CPU to avoid memory issues

    # Tokenize and truncate
    print(f"\nTokenizing (max_length={max_length})...")
    encoding = tokenizer(text, truncation=True, max_length=max_length, return_tensors='pt')
    input_ids = encoding['input_ids'][0]
    tokens = tokenizer.convert_ids_to_tokens(input_ids)

    print(f"Tokens: {len(tokens)} (truncated from {len(text.split())} words)")

    # Collect activations for every other layer
    print("\nCollecting activations...")
    num_layers = model.config.num_hidden_layers
    layers_to_collect = list(range(0, num_layers, 2))

    print(f"Collecting layers: {layers_to_collect}")

    with torch.no_grad():
        outputs = model(input_ids.unsqueeze(0), output_hidden_states=True)
        hidden_states = outputs.hidden_states  # Tuple of (batch, seq_len, hidden_dim)

    # Extract activations
    activations = {}
    for layer_idx in layers_to_collect:
        act = hidden_states[layer_idx][0].cpu().numpy()  # Remove batch dim
        activations[layer_idx] = act
        print(f"  Layer {layer_idx:2d}: {act.shape}")

    # Map speakers to token positions (simplified - just use "UNKNOWN" for truncated)
    speaker_labels = ["UNKNOWN"] * len(tokens)
    for start_idx, end_idx, speaker in speaker_spans:
        for i in range(start_idx, min(end_idx, len(tokens))):
            speaker_labels[i] = speaker

    # Save to HDF5
    print(f"\nSaving to {output_file}...")
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(output_path, 'w') as f:
        # Metadata
        f.attrs['transcript_id'] = transcript_id
        f.attrs['condition'] = condition
        f.attrs['model_name'] = model_name
        f.attrs['layers'] = layers_to_collect
        f.attrs['truncated'] = True
        f.attrs['max_length'] = max_length

        # Tokens and labels (encode strings as UTF-8)
        f.create_dataset('tokens', data=np.array([t.encode('utf-8') for t in tokens], dtype='S'))
        f.create_dataset('token_ids', data=input_ids.numpy())
        f.create_dataset('speaker_labels', data=np.array([s.encode('utf-8') for s in speaker_labels], dtype='S'))

        # Activations
        for layer_idx, acts in activations.items():
            layer_group = f.create_group(f'layer_{layer_idx}')
            layer_group.create_dataset('activations', data=acts, compression='gzip')

    print(f"\n✓ Saved successfully!")
    print(f"  File size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")
    print(f"{'='*70}\n")

    return str(output_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--condition-dir', required=True)
    parser.add_argument('--condition', required=True)
    parser.add_argument('--model', default='gpt2')
    parser.add_argument('--output', required=True)
    parser.add_argument('--max-length', type=int, default=512)

    args = parser.parse_args()

    collect_activations_simple(
        args.condition_dir,
        args.condition,
        args.model,
        args.output,
        args.max_length
    )
