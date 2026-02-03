"""
Speaker-Token Mapping System

Maps transcript turns to precise token sequences with offset tracking.
Handles tokenization-aware speaker span identification.
"""

import json
from dataclasses import dataclass, asdict
from typing import List, Tuple, Optional, Dict
from pathlib import Path
from transformers import AutoTokenizer
import argparse


@dataclass
class TranscriptTokens:
    """Token-level representation of a transcript."""
    transcript_id: str
    model: str
    tokens: List[str]
    token_ids: List[int]
    speaker_spans: List[Tuple[int, int, str]]  # (start_idx, end_idx, speaker)
    turn_boundaries: List[int]  # token indices where turns start
    offset_mapping: List[Tuple[int, int]]  # char offsets for each token

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> 'TranscriptTokens':
        return cls(**data)


class SpeakerTokenMapper:
    """Maps transcript turns to token-level structure with precise offsets."""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Ensure tokenizer has padding token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def map_transcript(self, jsonl_path: str) -> TranscriptTokens:
        """
        Maps JSONL transcript to token-level structure.

        Args:
            jsonl_path: Path to JSONL transcript file

        Returns:
            TranscriptTokens with precise token-level speaker spans
        """
        # Load transcript
        turns = []
        with open(jsonl_path, 'r') as f:
            for line in f:
                turns.append(json.loads(line))

        # Build full text with speaker labels
        full_text = self._build_labeled_text(turns)

        # Tokenize with offset mapping
        encoding = self.tokenizer(
            full_text,
            return_offsets_mapping=True,
            add_special_tokens=True
        )

        tokens = self.tokenizer.convert_ids_to_tokens(encoding['input_ids'])
        token_ids = encoding['input_ids']
        offset_mapping = encoding['offset_mapping']

        # Map character positions to speaker spans
        speaker_spans, turn_boundaries = self._map_speaker_spans(
            turns, full_text, offset_mapping
        )

        # Extract transcript ID from filename
        transcript_id = Path(jsonl_path).stem

        return TranscriptTokens(
            transcript_id=transcript_id,
            model=self.model_name,
            tokens=tokens,
            token_ids=token_ids,
            speaker_spans=speaker_spans,
            turn_boundaries=turn_boundaries,
            offset_mapping=offset_mapping
        )

    def _build_labeled_text(self, turns: List[Dict]) -> str:
        """Build labeled text: 'Speaker: dialogue\\n' format."""
        lines = []
        for turn in turns:
            speaker = turn['speaker']
            text = turn['text'].strip()
            lines.append(f"{speaker}: {text}")
        return '\n'.join(lines)

    def _map_speaker_spans(
        self,
        turns: List[Dict],
        full_text: str,
        offset_mapping: List[Tuple[int, int]]
    ) -> Tuple[List[Tuple[int, int, str]], List[int]]:
        """
        Map token positions to speakers using character offsets.

        Returns:
            (speaker_spans, turn_boundaries)
            speaker_spans: List of (start_token_idx, end_token_idx, speaker)
            turn_boundaries: List of token indices where turns start
        """
        # Build character-level speaker mapping
        char_to_speaker = {}
        char_pos = 0

        for turn in turns:
            speaker = turn['speaker']
            text = turn['text'].strip()
            turn_str = f"{speaker}: {text}\n"

            # Map characters in this turn to speaker
            for i in range(len(turn_str) - 1):  # exclude final \n
                char_to_speaker[char_pos + i] = speaker

            char_pos += len(turn_str)

        # Convert to token-level spans
        speaker_spans = []
        turn_boundaries = []
        current_speaker = None
        span_start = 0

        for token_idx, (start_char, end_char) in enumerate(offset_mapping):
            # Skip special tokens with (0, 0) offsets
            if start_char == end_char == 0:
                continue

            # Get speaker for this token (use start character)
            token_speaker = char_to_speaker.get(start_char, None)

            # Check if this starts a new turn
            if token_speaker != current_speaker:
                # Close previous span
                if current_speaker is not None:
                    speaker_spans.append((span_start, token_idx, current_speaker))

                # Start new span
                span_start = token_idx
                current_speaker = token_speaker

                if token_speaker is not None:
                    turn_boundaries.append(token_idx)

        # Close final span
        if current_speaker is not None:
            speaker_spans.append((span_start, len(offset_mapping), current_speaker))

        return speaker_spans, turn_boundaries

    def create_labeled_version(self, transcript_tokens: TranscriptTokens) -> str:
        """Create 'Speaker: dialogue' format with labels."""
        # Reconstruct from original JSONL
        jsonl_path = f"labs/transcripts/jsonl/{transcript_tokens.transcript_id}.jsonl"

        turns = []
        with open(jsonl_path, 'r') as f:
            for line in f:
                turns.append(json.loads(line))

        return self._build_labeled_text(turns)

    def create_unlabeled_version(self, transcript_tokens: TranscriptTokens) -> str:
        """Just dialogue, no speaker markers."""
        jsonl_path = f"labs/transcripts/jsonl/{transcript_tokens.transcript_id}.jsonl"

        turns = []
        with open(jsonl_path, 'r') as f:
            for line in f:
                turns.append(json.loads(line))

        lines = [turn['text'].strip() for turn in turns]
        return '\n'.join(lines)

    def create_corrupted_version(
        self,
        transcript_tokens: TranscriptTokens,
        corruption_rate: float = 0.3,
        seed: int = 42
    ) -> str:
        """Randomly swap speaker labels."""
        import random
        random.seed(seed)

        jsonl_path = f"labs/transcripts/jsonl/{transcript_tokens.transcript_id}.jsonl"

        turns = []
        with open(jsonl_path, 'r') as f:
            for line in f:
                turns.append(json.loads(line))

        # Get unique speakers
        speakers = list(set(turn['speaker'] for turn in turns))

        # Corrupt speaker labels
        lines = []
        for turn in turns:
            original_speaker = turn['speaker']
            text = turn['text'].strip()

            # Randomly swap speaker
            if random.random() < corruption_rate:
                # Choose different speaker
                other_speakers = [s for s in speakers if s != original_speaker]
                corrupted_speaker = random.choice(other_speakers) if other_speakers else original_speaker
                lines.append(f"{corrupted_speaker}: {text}")
            else:
                lines.append(f"{original_speaker}: {text}")

        return '\n'.join(lines)

    def visualize_mapping(self, transcript_tokens: TranscriptTokens, max_tokens: int = 50):
        """Print visualization of token-speaker mapping (for debugging)."""
        print(f"\nTranscript: {transcript_tokens.transcript_id}")
        print(f"Model: {transcript_tokens.model}")
        print(f"Total tokens: {len(transcript_tokens.tokens)}")
        print(f"Turn boundaries: {len(transcript_tokens.turn_boundaries)}")
        print(f"\nFirst {max_tokens} tokens:\n")

        # Build token -> speaker lookup
        token_to_speaker = {}
        for start, end, speaker in transcript_tokens.speaker_spans:
            for i in range(start, end):
                token_to_speaker[i] = speaker

        # Print tokens with speaker labels
        for i in range(min(max_tokens, len(transcript_tokens.tokens))):
            token = transcript_tokens.tokens[i]
            speaker = token_to_speaker.get(i, "NONE")
            boundary_marker = " <TURN>" if i in transcript_tokens.turn_boundaries else ""
            print(f"{i:3d} | {speaker:20s} | {token:20s}{boundary_marker}")


def main():
    parser = argparse.ArgumentParser(description="Map transcript to token-level structure")
    parser.add_argument('--input', required=True, help='Path to JSONL transcript')
    parser.add_argument('--model', default='meta-llama/Llama-3.1-8B',
                       help='Model name for tokenizer')
    parser.add_argument('--output', help='Output JSON path (optional)')
    parser.add_argument('--visualize', action='store_true',
                       help='Print token mapping visualization')

    args = parser.parse_args()

    # Create mapper
    mapper = SpeakerTokenMapper(args.model)

    # Map transcript
    print(f"Mapping transcript: {args.input}")
    transcript_tokens = mapper.map_transcript(args.input)

    # Visualize if requested
    if args.visualize:
        mapper.visualize_mapping(transcript_tokens)

    # Save if output specified
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(transcript_tokens.to_dict(), f, indent=2)

        print(f"\nSaved to: {args.output}")

    # Print summary
    print(f"\nSummary:")
    print(f"  Tokens: {len(transcript_tokens.tokens)}")
    print(f"  Speakers: {len(set(s for _, _, s in transcript_tokens.speaker_spans))}")
    print(f"  Turns: {len(transcript_tokens.turn_boundaries)}")


if __name__ == "__main__":
    main()
