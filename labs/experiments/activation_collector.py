"""
Activation Collector

Extracts hidden states from language models for speaker binding analysis.
Supports both local execution and remote execution via NDIF.
"""

import torch
import h5py
import numpy as np
from pathlib import Path
from typing import List, Optional, Dict, Tuple
from dataclasses import dataclass, asdict
import json

try:
    from nnsight import LanguageModel
except ImportError:
    print("Warning: nnsight not installed. Install with: pip install nnsight")
    LanguageModel = None


@dataclass
class ActivationData:
    """Container for collected activations."""
    transcript_id: str
    condition: str  # "labeled", "unlabeled", "corrupted"
    model_name: str
    layers: List[int]  # Which layers were collected
    tokens: List[str]  # Decoded tokens
    token_ids: List[int]  # Token IDs
    speaker_labels: List[str]  # Speaker for each token position
    activations: Dict[int, np.ndarray]  # layer -> (seq_len, hidden_dim)

    def save_hdf5(self, output_path: str):
        """Save activations to HDF5 format for efficient storage."""
        with h5py.File(output_path, 'w') as f:
            # Save metadata
            f.attrs['transcript_id'] = self.transcript_id
            f.attrs['condition'] = self.condition
            f.attrs['model_name'] = self.model_name
            f.attrs['layers'] = self.layers

            # Save tokens and labels
            f.create_dataset('tokens', data=np.array(self.tokens, dtype='S'))
            f.create_dataset('token_ids', data=np.array(self.token_ids))
            f.create_dataset('speaker_labels', data=np.array(self.speaker_labels, dtype='S'))

            # Save activations for each layer
            for layer_idx, acts in self.activations.items():
                layer_group = f.create_group(f'layer_{layer_idx}')
                layer_group.create_dataset('activations', data=acts, compression='gzip')

    @classmethod
    def load_hdf5(cls, input_path: str) -> 'ActivationData':
        """Load activations from HDF5 file."""
        with h5py.File(input_path, 'r') as f:
            # Load metadata
            transcript_id = f.attrs['transcript_id']
            condition = f.attrs['condition']
            model_name = f.attrs['model_name']
            layers = list(f.attrs['layers'])

            # Load tokens and labels
            tokens = [t.decode() for t in f['tokens'][:]]
            token_ids = list(f['token_ids'][:])
            speaker_labels = [s.decode() for s in f['speaker_labels'][:]]

            # Load activations
            activations = {}
            for layer_idx in layers:
                layer_group = f[f'layer_{layer_idx}']
                activations[layer_idx] = layer_group['activations'][:]

            return cls(
                transcript_id=transcript_id,
                condition=condition,
                model_name=model_name,
                layers=layers,
                tokens=tokens,
                token_ids=token_ids,
                speaker_labels=speaker_labels,
                activations=activations
            )


class ActivationCollector:
    """Collect hidden state activations from language models."""

    def __init__(self, model_name: str, layers: Optional[List[int]] = None):
        """
        Args:
            model_name: HuggingFace model identifier
            layers: Which layers to collect. Default: every other layer
                    e.g., [0, 2, 4, ..., 30] for 32-layer model
        """
        if LanguageModel is None:
            raise ImportError("nnsight not installed. Install with: pip install nnsight")

        self.model_name = model_name
        self.model = LanguageModel(model_name, device_map="auto")

        # Detect model architecture
        if hasattr(self.model, 'transformer') and hasattr(self.model.transformer, 'h'):
            # GPT-2 style architecture
            self.layer_accessor = lambda idx: self.model.transformer.h[idx].output[0]
            self.model_type = 'gpt2'
        elif hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
            # Llama style architecture
            self.layer_accessor = lambda idx: self.model.model.layers[idx].output[0]
            self.model_type = 'llama'
        else:
            raise ValueError(f"Unknown model architecture for {model_name}")

        # Determine layers to collect
        if layers is None:
            num_layers = self.model.config.num_hidden_layers
            self.layers = list(range(0, num_layers, 2))  # Every other layer
        else:
            self.layers = layers

        print(f"ActivationCollector initialized:")
        print(f"  Model: {model_name}")
        print(f"  Model type: {self.model_type}")
        print(f"  Layers: {self.layers}")
        print(f"  Total layers: {self.model.config.num_hidden_layers}")

    def collect_activations(
        self,
        prompt: str,
        condition: str,
        speaker_spans: List[Tuple[int, int, str]],
        transcript_id: str,
        remote: bool = False,
        scan: bool = True
    ) -> ActivationData:
        """
        Collect hidden states at specified layers.

        Args:
            prompt: Input text
            condition: "labeled", "unlabeled", or "corrupted"
            speaker_spans: List of (start_idx, end_idx, speaker_name)
            transcript_id: Identifier for transcript
            remote: Use NDIF remote execution
            scan: Use scan mode for remote execution (required for NDIF)

        Returns:
            ActivationData with activations organized by layer
        """
        print(f"\nCollecting activations for {transcript_id} ({condition})...")

        # Tokenize to get token count
        token_ids = self.model.tokenizer.encode(prompt)
        tokens = self.model.tokenizer.convert_ids_to_tokens(token_ids)

        print(f"  Tokens: {len(tokens)}")
        print(f"  Remote: {remote}")

        # Collect activations using nnsight
        with self.model.trace(prompt, remote=remote, scan=scan) as tracer:
            saved_acts = {}
            for layer_idx in self.layers:
                # Access hidden states for this layer (architecture-specific)
                hidden = self.layer_accessor(layer_idx)
                saved_acts[layer_idx] = hidden.save()

        # Extract values from saved tensors
        activations = {}
        for layer_idx, saved in saved_acts.items():
            act = self._get_value(saved)

            # Handle batch dimension: (batch, seq_len, hidden_dim) -> (seq_len, hidden_dim)
            if len(act.shape) == 3:
                act = act[0]  # Take first batch item

            # Convert to numpy
            if isinstance(act, torch.Tensor):
                act = act.cpu().numpy()

            activations[layer_idx] = act
            print(f"  Layer {layer_idx:2d}: {act.shape}")

        # Map speaker labels to token positions
        speaker_labels = self._map_speakers_to_tokens(
            len(tokens), speaker_spans
        )

        return ActivationData(
            transcript_id=transcript_id,
            condition=condition,
            model_name=self.model_name,
            layers=self.layers,
            tokens=tokens,
            token_ids=token_ids,
            speaker_labels=speaker_labels,
            activations=activations
        )

    def _get_value(self, saved):
        """Extract value from saved tensor (handles both local and remote)."""
        try:
            return saved.value
        except AttributeError:
            return saved

    def _map_speakers_to_tokens(
        self,
        num_tokens: int,
        speaker_spans: List[Tuple[int, int, str]]
    ) -> List[str]:
        """Map speaker labels to each token position."""
        speaker_labels = ["UNKNOWN"] * num_tokens

        for start_idx, end_idx, speaker in speaker_spans:
            for i in range(start_idx, min(end_idx, num_tokens)):
                speaker_labels[i] = speaker

        return speaker_labels

    def collect_from_condition_dir(
        self,
        condition_dir: str,
        condition: str,
        output_dir: str,
        remote: bool = False
    ) -> str:
        """
        Collect activations from a condition directory.

        Args:
            condition_dir: Path to directory with condition files
            condition: "labeled", "unlabeled", or "corrupted"
            output_dir: Where to save HDF5 files
            remote: Use NDIF remote execution

        Returns:
            Path to saved HDF5 file
        """
        condition_path = Path(condition_dir)

        # Load condition file
        condition_file = condition_path / f"{condition}.txt"
        with open(condition_file, 'r') as f:
            prompt = f.read()

        # Load token mapping metadata
        metadata_file = condition_path / "token_mapping.json"
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)

        transcript_id = metadata['transcript_id']
        speaker_spans = [tuple(span) for span in metadata['speaker_spans']]

        # Collect activations
        activation_data = self.collect_activations(
            prompt=prompt,
            condition=condition,
            speaker_spans=speaker_spans,
            transcript_id=transcript_id,
            remote=remote
        )

        # Save to HDF5
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        model_short = self.model_name.split('/')[-1].replace('.', '-')
        output_file = output_path / f"{transcript_id}_{condition}_{model_short}.h5"

        activation_data.save_hdf5(str(output_file))
        print(f"\n✓ Saved: {output_file}")

        return str(output_file)


def main():
    """Test activation collection."""
    import argparse

    parser = argparse.ArgumentParser(description="Collect activations from language model")
    parser.add_argument('--prompt', help='Input prompt text')
    parser.add_argument('--condition-dir', help='Directory with condition files')
    parser.add_argument('--condition', default='labeled',
                       choices=['labeled', 'unlabeled', 'corrupted'])
    parser.add_argument('--model', default='meta-llama/Llama-3.1-8B')
    parser.add_argument('--layers', type=int, nargs='+',
                       help='Specific layers to collect (default: every other)')
    parser.add_argument('--output', required=True, help='Output HDF5 file path')
    parser.add_argument('--remote', action='store_true',
                       help='Use NDIF remote execution')

    args = parser.parse_args()

    # Initialize collector
    collector = ActivationCollector(
        model_name=args.model,
        layers=args.layers
    )

    # Collect from condition directory
    if args.condition_dir:
        output_file = collector.collect_from_condition_dir(
            condition_dir=args.condition_dir,
            condition=args.condition,
            output_dir=Path(args.output).parent,
            remote=args.remote
        )
        print(f"\n✓ Complete: {output_file}")

    # Direct prompt collection (for testing)
    elif args.prompt:
        activation_data = collector.collect_activations(
            prompt=args.prompt,
            condition=args.condition,
            speaker_spans=[],
            transcript_id="test",
            remote=args.remote
        )
        activation_data.save_hdf5(args.output)
        print(f"\n✓ Saved: {args.output}")

    else:
        parser.error("Must provide either --prompt or --condition-dir")


if __name__ == "__main__":
    main()
