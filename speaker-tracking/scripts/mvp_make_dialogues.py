#!/usr/bin/env python3
"""Build MVP dialogues from MeetingBank transcripts."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import random
import re
from pathlib import Path
from typing import Any


# MeetingBank commonly uses labels like "Speaker 4:".
_SPEAKER_LABEL = r"[A-Za-z][A-Za-z0-9_ .'\-]{0,59}"
SPEAKER_UTTERANCE_RE = re.compile(
    rf"(?P<speaker>{_SPEAKER_LABEL}):\s*(?P<text>.+?)(?=(?:\s+{_SPEAKER_LABEL}:\s)|$)",
    re.DOTALL,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-dialogues", type=int, default=20)
    parser.add_argument(
        "--dataset-id",
        type=str,
        default="lytang/MeetingBank-transcript",
        help="Hugging Face dataset id.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Dataset split to sample from.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--min-turns",
        type=int,
        default=8,
        help="Minimum number of turns after filtering to top-2 speakers.",
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=80,
        help="Maximum turns kept per dialogue after filtering.",
    )
    parser.add_argument(
        "--max-chars-per-turn",
        type=int,
        default=320,
        help="Truncate each turn text to this many characters.",
    )
    parser.add_argument(
        "--speaker-text-mode",
        type=str,
        choices=("omit", "keep", "anonymize"),
        default="omit",
        help=(
            "How to render speaker IDs in turn text: "
            "'omit' keeps only utterance text, "
            "'keep' prepends original speaker IDs, "
            "'anonymize' prepends stable SPEAKER_1/SPEAKER_2 labels."
        ),
    )
    return parser.parse_args()


def _clean_text(text: str, max_chars: int) -> str:
    compact = " ".join(text.split())
    return compact[:max_chars].strip()


def _is_plausible_speaker(speaker: str) -> bool:
    s = speaker.strip()
    if not s or len(s) > 60:
        return False
    tokens = s.split()
    if len(tokens) > 6:
        return False
    # Avoid obvious non-speaker labels such as section headers.
    if s.upper() == s and len(s) > 24:
        return False
    return True


def _extract_turns_from_source(source: str, max_chars_per_turn: int) -> list[dict[str, str]]:
    turns: list[dict[str, str]] = []
    for match in SPEAKER_UTTERANCE_RE.finditer(source):
        speaker = match.group("speaker").strip()
        text = _clean_text(match.group("text"), max_chars=max_chars_per_turn)
        if not _is_plausible_speaker(speaker) or not text:
            continue
        if turns and turns[-1]["speaker"] == speaker:
            turns[-1]["text"] = _clean_text(
                f'{turns[-1]["text"]} {text}',
                max_chars=max_chars_per_turn,
            )
            continue
        turns.append({"speaker": speaker, "text": text})
    return turns


def _apply_speaker_text_mode(
    turns: list[dict[str, str]],
    mode: str,
    ordered_speakers: list[str],
    max_chars_per_turn: int,
) -> list[dict[str, str]]:
    if mode == "omit":
        return turns

    anon_map = {
        speaker: f"SPEAKER_{idx + 1}" for idx, speaker in enumerate(ordered_speakers)
    }
    rendered: list[dict[str, str]] = []
    for turn in turns:
        if mode == "keep":
            prefix = turn["speaker"]
        elif mode == "anonymize":
            prefix = anon_map[turn["speaker"]]
        else:
            raise ValueError(f"Unsupported --speaker-text-mode: {mode}")
        text = _clean_text(f"{prefix}: {turn['text']}", max_chars=max_chars_per_turn)
        rendered.append({"speaker": turn["speaker"], "text": text})
    return rendered


def _map_to_alice_bob(
    turns: list[dict[str, str]],
    min_turns: int,
    max_turns: int,
) -> tuple[list[dict[str, str]], dict[str, str]] | tuple[None, None]:
    speaker_counts = Counter(turn["speaker"] for turn in turns)
    top_two = [speaker for speaker, _ in speaker_counts.most_common(2)]
    if len(top_two) < 2:
        return None, None

    speaker_map = {
        top_two[0]: "Alice",
        top_two[1]: "Bob",
    }
    filtered = [
        {"speaker": speaker_map[turn["speaker"]], "text": turn["text"]}
        for turn in turns
        if turn["speaker"] in speaker_map
    ][:max_turns]
    if len(filtered) < min_turns:
        return None, None

    count_after_filter = Counter(turn["speaker"] for turn in filtered)
    if count_after_filter.get("Alice", 0) < 2 or count_after_filter.get("Bob", 0) < 2:
        return None, None
    return filtered, speaker_map


def speaker_swap(turns: list[dict[str, str]]) -> list[dict[str, str]]:
    swapped: list[dict[str, str]] = []
    for turn in turns:
        if turn["speaker"] == "Alice":
            speaker = "Bob"
        elif turn["speaker"] == "Bob":
            speaker = "Alice"
        else:
            speaker = turn["speaker"]
        swapped.append({"speaker": speaker, "text": turn["text"]})
    return swapped


def main() -> None:
    args = parse_args()
    if args.num_dialogues <= 0:
        raise ValueError("--num-dialogues must be positive.")
    if args.min_turns < 4:
        raise ValueError("--min-turns must be >= 4.")
    if args.max_turns < args.min_turns:
        raise ValueError("--max-turns must be >= --min-turns.")

    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "Install huggingface datasets first: pip install datasets"
        ) from exc

    split = load_dataset(args.dataset_id, split=args.split)
    row_indices = list(range(len(split)))
    random.Random(args.seed).shuffle(row_indices)

    dialogues: list[dict[str, object]] = []
    skipped = 0
    for row_idx in row_indices:
        if len(dialogues) >= args.num_dialogues:
            break
        row: dict[str, Any] = split[row_idx]
        source = row.get("source", "")
        if not isinstance(source, str) or not source.strip():
            skipped += 1
            continue
        raw_turns = _extract_turns_from_source(
            source=source,
            max_chars_per_turn=args.max_chars_per_turn,
        )
        base_turns, speaker_map = _map_to_alice_bob(
            turns=raw_turns,
            min_turns=args.min_turns,
            max_turns=args.max_turns,
        )
        if not base_turns or not speaker_map:
            skipped += 1
            continue

        ordered_original_speakers = [
            original_speaker
            for original_speaker, role_name in sorted(
                speaker_map.items(), key=lambda item: item[1]
            )
        ]
        base_turns = _apply_speaker_text_mode(
            turns=base_turns,
            mode=args.speaker_text_mode,
            ordered_speakers=ordered_original_speakers,
            max_chars_per_turn=args.max_chars_per_turn,
        )

        transcript_id = str(row.get("meeting_id", f"{args.split}_{row_idx}"))
        alias_by_role = {
            role_name: original_name for original_name, role_name in speaker_map.items()
        }
        dialogues.append(
            {
                "transcript_id": transcript_id,
                "topic": str(row.get("type", "")),
                "city": str(row.get("city", "")),
                "base": base_turns,
                "speaker_swapped": speaker_swap(base_turns),
                "speaker_aliases": alias_by_role,
            }
        )

    if len(dialogues) < args.num_dialogues:
        raise ValueError(
            f"Could only build {len(dialogues)} dialogues from split='{args.split}' "
            f"(requested {args.num_dialogues}, skipped {skipped})."
        )

    payload = {
        "metadata": {
            "generator": "mvp_make_dialogues.py",
            "dataset_id": args.dataset_id,
            "split": args.split,
            "seed": args.seed,
            "num_dialogues": args.num_dialogues,
            "min_turns": args.min_turns,
            "max_turns": args.max_turns,
            "max_chars_per_turn": args.max_chars_per_turn,
            "speaker_text_mode": args.speaker_text_mode,
            "skipped_rows": skipped,
        },
        "dialogues": dialogues,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=True)


if __name__ == "__main__":
    main()
