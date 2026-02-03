#!/usr/bin/env python3
"""
Convert interview transcripts from plain text format to JSONL entries.

Each speaker turn becomes a JSON object with metadata and sequential context.
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import List, Dict, Optional, Tuple


class TranscriptParser:
    """Main parser for converting transcript text to structured JSON."""

    # Pattern matches: "Speaker Name  0:00  " (speaker followed by timestamp)
    SPEAKER_PATTERN = re.compile(r'^([A-Za-z\s.]+?)\s+(\d{1,2}:\d{2})\s*$')
    FOOTER_MARKER = "Transcribed by https://otter.ai"

    def __init__(self, verbose: bool = False):
        self.verbose = verbose

    def parse_transcript(
        self,
        filepath: Path,
        transcript_id: Optional[str] = None,
        topic_override: Optional[str] = None
    ) -> List[Dict]:
        """
        Parse a transcript file into structured turn entries.

        Args:
            filepath: Path to transcript file
            transcript_id: Override ID (defaults to filename stem)
            topic_override: Override auto-detected topic

        Returns:
            List of turn dictionaries
        """
        if not filepath.exists():
            raise FileNotFoundError(f"Transcript file not found: {filepath}")

        transcript_id = transcript_id or filepath.stem

        # Read file with encoding fallback
        try:
            content = filepath.read_text(encoding='utf-8')
        except UnicodeDecodeError:
            content = filepath.read_text(encoding='latin-1')

        lines = content.split('\n')
        turns = []
        current_turn = None
        current_dialogue_lines = []

        for line_num, line in enumerate(lines, 1):
            # Strip line numbers added by tool (format: "  123→")
            line = re.sub(r'^\s*\d+→', '', line)

            # Stop at footer
            if self.FOOTER_MARKER in line:
                break

            # Check for speaker line
            match = self.SPEAKER_PATTERN.match(line.strip())

            if match:
                # Finalize previous turn
                if current_turn is not None:
                    current_turn['text'] = self._extract_dialogue_text(current_dialogue_lines)
                    if self._validate_turn(current_turn, line_num):
                        turns.append(current_turn)
                    current_dialogue_lines = []

                # Start new turn
                speaker, timestamp = match.groups()
                current_turn = {
                    'speaker': speaker.strip(),
                    'timestamp': timestamp.strip(),
                    'turn_number': len(turns) + 1,
                }

            elif line.strip() and current_turn is not None:
                # Accumulate dialogue lines
                current_dialogue_lines.append(line.strip())

        # Finalize last turn
        if current_turn is not None:
            current_turn['text'] = self._extract_dialogue_text(current_dialogue_lines)
            if self._validate_turn(current_turn, len(lines)):
                turns.append(current_turn)

        # Post-process: add sequential links and metadata
        turns = self._add_sequential_context(turns)
        metadata = self._extract_metadata(turns, filepath, topic_override)

        for turn in turns:
            turn.update(metadata)

        if self.verbose:
            print(f"Parsed {len(turns)} turns from {filepath.name}")

        return turns

    def _extract_dialogue_text(self, lines: List[str]) -> str:
        """Combine multi-line dialogue into single text field."""
        return ' '.join(lines).strip()

    def _validate_turn(self, turn: Dict, line_num: int) -> bool:
        """Validate turn has required fields and content."""
        if not turn.get('speaker'):
            if self.verbose:
                print(f"Warning (line {line_num}): Empty speaker name, skipping turn")
            return False

        if not turn.get('text') or len(turn['text']) < 10:
            if self.verbose:
                print(f"Warning (line {line_num}): Insufficient dialogue for {turn['speaker']}, skipping")
            return False

        # Validate timestamp format
        timestamp = turn.get('timestamp', '')
        if not re.match(r'^\d{1,2}:\d{2}$', timestamp):
            if self.verbose:
                print(f"Warning (line {line_num}): Invalid timestamp format '{timestamp}'")

        return True

    def _add_sequential_context(self, turns: List[Dict]) -> List[Dict]:
        """Add previous_speaker and next_speaker fields."""
        for i, turn in enumerate(turns):
            turn['previous_speaker'] = turns[i-1]['speaker'] if i > 0 else None
            turn['next_speaker'] = turns[i+1]['speaker'] if i < len(turns) - 1 else None
            # Ensure turn_number is correct after any filtering
            turn['turn_number'] = i + 1

        return turns

    def _extract_metadata(
        self,
        turns: List[Dict],
        filepath: Path,
        topic_override: Optional[str] = None
    ) -> Dict:
        """Extract metadata from turns."""
        participants = []
        seen_speakers = set()

        for turn in turns:
            speaker = turn['speaker']
            if speaker not in seen_speakers:
                participants.append(speaker)
                seen_speakers.add(speaker)

        total_duration = turns[-1]['timestamp'] if turns else "0:00"

        # Auto-detect topic if not overridden
        if topic_override:
            topic = topic_override
        else:
            topic = self._infer_topic(turns, filepath.stem)

        return {
            'transcript_id': filepath.stem,
            'topic': topic,
            'participants': participants,
            'total_duration': total_duration
        }

    def _infer_topic(self, turns: List[Dict], transcript_id: str) -> str:
        """Infer topic from transcript content."""
        # Known topics for transcripts 1-3
        known_topics = {
            '1': 'Artificial turf vs natural grass sports injuries',
            '2': 'Artificial turf vs natural grass sports injuries',
            '3': 'Open-source AI and closed-source models'
        }

        if transcript_id in known_topics:
            return known_topics[transcript_id]

        # Fallback: use keywords from first few substantial turns
        keywords = []
        for turn in turns[:5]:
            text = turn.get('text', '').lower()
            if len(text) > 50:  # Only substantial turns
                # Extract potential keywords
                if 'turf' in text or 'grass' in text or 'injury' in text:
                    return 'Sports surface safety research'
                elif 'open source' in text or 'model' in text or 'ai' in text:
                    return 'AI model development and deployment'

        return 'Interview transcript'


class JSONLWriter:
    """Write turns to JSONL format."""

    @staticmethod
    def write(turns: List[Dict], output_path: Path, verbose: bool = False):
        """Write turns to JSONL file."""
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with output_path.open('w', encoding='utf-8') as f:
            for turn in turns:
                json.dump(turn, f, ensure_ascii=False)
                f.write('\n')

        if verbose:
            print(f"Wrote {len(turns)} entries to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Convert interview transcripts to JSONL format',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process single file
  %(prog)s labs/transcripts/1.txt -o labs/transcripts/jsonl/1.jsonl

  # Batch process directory
  %(prog)s labs/transcripts/ -o labs/transcripts/jsonl/

  # Validate without writing
  %(prog)s labs/transcripts/1.txt --validate-only -v
        """
    )

    parser.add_argument(
        'input',
        type=Path,
        help='Input transcript file or directory'
    )

    parser.add_argument(
        '-o', '--output',
        type=Path,
        help='Output file or directory'
    )

    parser.add_argument(
        '--transcript-id',
        help='Override transcript ID (default: filename)'
    )

    parser.add_argument(
        '--topic',
        help='Override auto-detected topic'
    )

    parser.add_argument(
        '--validate-only',
        action='store_true',
        help='Validate without writing output'
    )

    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Verbose logging'
    )

    args = parser.parse_args()

    transcript_parser = TranscriptParser(verbose=args.verbose)

    # Determine input files
    if args.input.is_file():
        input_files = [args.input]
    elif args.input.is_dir():
        input_files = sorted(args.input.glob('*.txt'))
        if not input_files:
            print(f"No .txt files found in {args.input}", file=sys.stderr)
            sys.exit(1)
    else:
        print(f"Input not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    # Process each file
    for input_file in input_files:
        try:
            turns = transcript_parser.parse_transcript(
                input_file,
                transcript_id=args.transcript_id,
                topic_override=args.topic
            )

            if not args.validate_only:
                # Determine output path
                if args.output:
                    if args.output.is_dir() or (not args.output.exists() and len(input_files) > 1):
                        output_file = args.output / f"{input_file.stem}.jsonl"
                    else:
                        output_file = args.output
                else:
                    output_file = input_file.with_suffix('.jsonl')

                JSONLWriter.write(turns, output_file, verbose=args.verbose)
            else:
                print(f"✓ Validated {input_file.name}: {len(turns)} turns")

        except Exception as e:
            print(f"Error processing {input_file}: {e}", file=sys.stderr)
            if args.verbose:
                import traceback
                traceback.print_exc()
            sys.exit(1)

    if args.verbose:
        print(f"\nProcessed {len(input_files)} file(s)")


if __name__ == '__main__':
    main()
