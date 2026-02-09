#!/usr/bin/env python3
"""Generate a small controlled dialogue set for MVP role tests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


BASE_TEMPLATES = [
    (
        "I believe {topic} is urgent.",
        "I think {counter_topic} should be prioritized.",
    ),
    (
        "There are long-term costs if we ignore {topic}.",
        "Short-term tradeoffs still matter for {counter_topic}.",
    ),
    (
        "Policy should protect future outcomes for {topic}.",
        "Policy should avoid harming current outcomes for {counter_topic}.",
    ),
]

TOPIC_PAIRS = [
    ("climate action", "economic growth"),
    ("education reform", "budget discipline"),
    ("public health", "private sector flexibility"),
    ("housing access", "local zoning control"),
    ("AI safety", "innovation speed"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-dialogues", type=int, default=20)
    return parser.parse_args()


def build_dialogue(dialogue_id: str, topic: str, counter_topic: str) -> list[dict[str, str]]:
    turns: list[dict[str, str]] = []
    for a_tmpl, b_tmpl in BASE_TEMPLATES:
        turns.append({"speaker": "Alice", "text": a_tmpl.format(topic=topic, counter_topic=counter_topic)})
        turns.append({"speaker": "Bob", "text": b_tmpl.git format(topic=topic, counter_topic=counter_topic)})
    return turns


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
    dialogues: list[dict[str, object]] = []
    for idx in range(args.num_dialogues):
        topic, counter_topic = TOPIC_PAIRS[idx % len(TOPIC_PAIRS)]
        transcript_id = f"d_{idx:03d}"
        base_turns = build_dialogue(transcript_id, topic=topic, counter_topic=counter_topic)
        dialogues.append(
            {
                "transcript_id": transcript_id,
                "topic": topic,
                "base": base_turns,
                "speaker_swapped": speaker_swap(base_turns),
            }
        )

    payload = {"dialogues": dialogues}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=True)


if __name__ == "__main__":
    main()
