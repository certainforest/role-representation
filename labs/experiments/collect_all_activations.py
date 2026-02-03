"""
Batch Activation Collection

Process all transcripts × conditions × models to collect activations.
Supports parallel processing and resumable execution.
"""

import argparse
from pathlib import Path
from typing import List
import json
import time
from activation_collector import ActivationCollector


class BatchCollector:
    """Batch process multiple transcripts, conditions, and models."""

    def __init__(
        self,
        transcript_dirs: List[str],
        models: List[str],
        conditions: List[str],
        output_dir: str,
        remote: bool = False
    ):
        self.transcript_dirs = [Path(d) for d in transcript_dirs]
        self.models = models
        self.conditions = conditions
        self.output_dir = Path(output_dir)
        self.remote = remote

        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Track progress
        self.completed = self._load_progress()
        self.progress_file = self.output_dir / "collection_progress.json"

    def _load_progress(self) -> set:
        """Load previously completed collections."""
        progress_file = self.output_dir / "collection_progress.json"
        if progress_file.exists():
            with open(progress_file, 'r') as f:
                data = json.load(f)
                return set(tuple(item) for item in data['completed'])
        return set()

    def _save_progress(self):
        """Save progress to resume later."""
        with open(self.progress_file, 'w') as f:
            json.dump({
                'completed': [list(item) for item in self.completed],
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            }, f, indent=2)

    def collect_all(self):
        """Collect activations for all combinations."""
        # Count total tasks
        total_tasks = (
            len(self.transcript_dirs) *
            len(self.conditions) *
            len(self.models)
        )

        completed_count = len(self.completed)
        remaining = total_tasks - completed_count

        print(f"\n{'='*70}")
        print(f"BATCH ACTIVATION COLLECTION")
        print(f"{'='*70}")
        print(f"Transcripts: {len(self.transcript_dirs)}")
        print(f"Conditions: {', '.join(self.conditions)}")
        print(f"Models: {', '.join(self.models)}")
        print(f"Total tasks: {total_tasks}")
        print(f"Completed: {completed_count}")
        print(f"Remaining: {remaining}")
        print(f"Remote execution: {self.remote}")
        print(f"Output: {self.output_dir}")
        print(f"{'='*70}\n")

        # Process each combination
        task_num = 0
        for transcript_dir in self.transcript_dirs:
            transcript_id = transcript_dir.name

            for condition in self.conditions:
                for model in self.models:
                    task_num += 1
                    task_key = (transcript_id, condition, model)

                    # Skip if already completed
                    if task_key in self.completed:
                        print(f"[{task_num}/{total_tasks}] SKIPPING (already completed): "
                              f"{transcript_id} / {condition} / {model}")
                        continue

                    print(f"\n[{task_num}/{total_tasks}] PROCESSING: "
                          f"{transcript_id} / {condition} / {model}")

                    try:
                        self._collect_one(transcript_dir, condition, model)

                        # Mark as completed
                        self.completed.add(task_key)
                        self._save_progress()

                        print(f"  ✓ SUCCESS")

                    except Exception as e:
                        print(f"  ✗ FAILED: {e}")
                        # Continue with next task
                        continue

        print(f"\n{'='*70}")
        print(f"BATCH COLLECTION COMPLETE")
        print(f"{'='*70}")
        print(f"Completed: {len(self.completed)} / {total_tasks}")
        print(f"Output directory: {self.output_dir}")
        print(f"{'='*70}\n")

    def _collect_one(self, transcript_dir: Path, condition: str, model: str):
        """Collect activations for one combination."""
        # Initialize collector for this model
        collector = ActivationCollector(model_name=model)

        # Collect activations
        output_file = collector.collect_from_condition_dir(
            condition_dir=str(transcript_dir),
            condition=condition,
            output_dir=str(self.output_dir),
            remote=self.remote
        )

        return output_file


def main():
    parser = argparse.ArgumentParser(
        description="Batch collect activations for all transcripts and conditions"
    )
    parser.add_argument('--transcripts', required=True,
                       help='Directory containing transcript condition folders')
    parser.add_argument('--models', nargs='+',
                       default=['meta-llama/Llama-3.1-8B'],
                       help='List of model names to process')
    parser.add_argument('--conditions', nargs='+',
                       default=['labeled', 'unlabeled', 'corrupted'],
                       help='Which conditions to process')
    parser.add_argument('--output', required=True,
                       help='Output directory for HDF5 files')
    parser.add_argument('--remote', action='store_true',
                       help='Use NDIF remote execution')
    parser.add_argument('--resume', action='store_true',
                       help='Resume from previous run (automatically handled)')

    args = parser.parse_args()

    # Find all transcript directories
    transcripts_path = Path(args.transcripts)
    transcript_dirs = [d for d in transcripts_path.iterdir() if d.is_dir()]

    if not transcript_dirs:
        print(f"Error: No transcript directories found in {args.transcripts}")
        return

    print(f"\nFound {len(transcript_dirs)} transcript directories:")
    for d in transcript_dirs:
        print(f"  - {d.name}")

    # Initialize batch collector
    collector = BatchCollector(
        transcript_dirs=[str(d) for d in transcript_dirs],
        models=args.models,
        conditions=args.conditions,
        output_dir=args.output,
        remote=args.remote
    )

    # Run collection
    collector.collect_all()


if __name__ == "__main__":
    main()
