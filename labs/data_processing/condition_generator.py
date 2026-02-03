"""
Three Label Conditions Generator

Generates three experimental conditions from clean transcripts:
1. Labeled - Normal transcript with speaker markers
2. Unlabeled - Dialogue only, model must infer speakers
3. Corrupted - Randomly misattributed labels
"""

import argparse
from pathlib import Path
from speaker_token_mapper import SpeakerTokenMapper, TranscriptTokens
import json


class ConditionGenerator:
    """Generate experimental conditions from transcripts."""

    def __init__(self, model_name: str, corruption_rate: float = 0.3, seed: int = 42):
        self.mapper = SpeakerTokenMapper(model_name)
        self.corruption_rate = corruption_rate
        self.seed = seed

    def generate_all_conditions(
        self,
        jsonl_path: str,
        output_dir: str
    ) -> dict:
        """
        Generate all three conditions and save to output directory.

        Args:
            jsonl_path: Path to input JSONL transcript
            output_dir: Directory to save condition files

        Returns:
            dict with paths to generated files
        """
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Map transcript to tokens
        print(f"Processing: {jsonl_path}")
        transcript_tokens = self.mapper.map_transcript(jsonl_path)

        # Generate three conditions
        conditions = {}

        # 1. Labeled
        labeled_text = self.mapper.create_labeled_version(transcript_tokens)
        labeled_path = output_path / "labeled.txt"
        with open(labeled_path, 'w') as f:
            f.write(labeled_text)
        conditions['labeled'] = str(labeled_path)
        print(f"  ✓ Labeled: {labeled_path}")

        # 2. Unlabeled
        unlabeled_text = self.mapper.create_unlabeled_version(transcript_tokens)
        unlabeled_path = output_path / "unlabeled.txt"
        with open(unlabeled_path, 'w') as f:
            f.write(unlabeled_text)
        conditions['unlabeled'] = str(unlabeled_path)
        print(f"  ✓ Unlabeled: {unlabeled_path}")

        # 3. Corrupted
        corrupted_text = self.mapper.create_corrupted_version(
            transcript_tokens,
            corruption_rate=self.corruption_rate,
            seed=self.seed
        )
        corrupted_path = output_path / "corrupted.txt"
        with open(corrupted_path, 'w') as f:
            f.write(corrupted_text)
        conditions['corrupted'] = str(corrupted_path)
        print(f"  ✓ Corrupted: {corrupted_path} (rate={self.corruption_rate})")

        # Save token mapping metadata
        metadata_path = output_path / "token_mapping.json"
        with open(metadata_path, 'w') as f:
            json.dump(transcript_tokens.to_dict(), f, indent=2)
        conditions['metadata'] = str(metadata_path)
        print(f"  ✓ Metadata: {metadata_path}")

        # Save condition info
        info_path = output_path / "condition_info.json"
        info = {
            'transcript_id': transcript_tokens.transcript_id,
            'model': transcript_tokens.model,
            'corruption_rate': self.corruption_rate,
            'seed': self.seed,
            'num_tokens': len(transcript_tokens.tokens),
            'num_speakers': len(set(s for _, _, s in transcript_tokens.speaker_spans)),
            'num_turns': len(transcript_tokens.turn_boundaries),
            'files': conditions
        }
        with open(info_path, 'w') as f:
            json.dump(info, f, indent=2)

        print(f"\n✓ Generated all conditions in: {output_dir}")
        return conditions


def main():
    parser = argparse.ArgumentParser(
        description="Generate experimental conditions from transcript"
    )
    parser.add_argument('--input', required=True,
                       help='Path to JSONL transcript')
    parser.add_argument('--output', required=True,
                       help='Output directory for condition files')
    parser.add_argument('--model', default='meta-llama/Llama-3.1-8B',
                       help='Model name for tokenizer')
    parser.add_argument('--corruption-rate', type=float, default=0.3,
                       help='Corruption rate for corrupted condition (0-1)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for corruption')

    args = parser.parse_args()

    # Generate conditions
    generator = ConditionGenerator(
        model_name=args.model,
        corruption_rate=args.corruption_rate,
        seed=args.seed
    )

    conditions = generator.generate_all_conditions(
        jsonl_path=args.input,
        output_dir=args.output
    )

    print(f"\n{'='*60}")
    print("GENERATION COMPLETE")
    print(f"{'='*60}")
    print(f"Input: {args.input}")
    print(f"Output: {args.output}")
    print(f"Conditions: {', '.join(conditions.keys())}")


if __name__ == "__main__":
    main()
