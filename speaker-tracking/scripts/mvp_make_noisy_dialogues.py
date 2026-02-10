#!/usr/bin/env python3
"""Generate noisy dialogue variants for robustness testing."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

# Paraphrase pools (same semantic stance, different wording)
PARAPHRASES_A = {
    "I believe {topic} is urgent.": [
        "{topic} needs immediate attention.",
        "We can't afford to delay on {topic}.",
        "The urgency of {topic} is clear.",
        "I think {topic} should be addressed now.",
        "{topic} is a pressing issue.",
    ],
    "There are long-term costs if we ignore {topic}.": [
        "Ignoring {topic} will have serious long-term consequences.",
        "The long-term price of neglecting {topic} is too high.",
        "We'll pay a steep price later if we don't act on {topic}.",
        "Failing to address {topic} creates future problems.",
        "The costs of inaction on {topic} compound over time.",
    ],
    "Policy should protect future outcomes for {topic}.": [
        "Our policies need to safeguard long-term results for {topic}.",
        "We should design policy that secures future gains in {topic}.",
        "Policy must prioritize future success on {topic}.",
        "Protecting future outcomes for {topic} should guide policy.",
        "Long-term policy for {topic} is what matters.",
    ],
}

PARAPHRASES_B = {
    "I think {counter_topic} should be prioritized.": [
        "{counter_topic} deserves priority right now.",
        "We need to focus on {counter_topic} first.",
        "I believe {counter_topic} is the top priority.",
        "Prioritizing {counter_topic} makes sense.",
        "{counter_topic} should come first in my view.",
    ],
    "Short-term tradeoffs still matter for {counter_topic}.": [
        "We can't ignore short-term impacts on {counter_topic}.",
        "Short-term considerations for {counter_topic} are important too.",
        "The immediate costs to {counter_topic} matter.",
        "You have to account for near-term tradeoffs with {counter_topic}.",
        "Short-term effects on {counter_topic} shouldn't be dismissed.",
    ],
    "Policy should avoid harming current outcomes for {counter_topic}.": [
        "We need policy that doesn't damage current {counter_topic} results.",
        "Policy shouldn't hurt today's outcomes for {counter_topic}.",
        "Protecting present gains in {counter_topic} matters.",
        "Current {counter_topic} outcomes must be preserved by policy.",
        "Policy should safeguard existing {counter_topic} progress.",
    ],
}

FILLERS = [
    "Well, ",
    "I mean, ",
    "You know, ",
    "Honestly, ",
    "Look, ",
    "So, ",
    "",
    "",
    "",
]

HEDGES = [
    " I think",
    " in my view",
    " it seems to me",
    "",
    "",
]

CONVERSATIONAL_PREFIXES = [
    "I hear you, but ",
    "Sure, but ",
    "That's fair, though ",
    "I see that, but ",
    "",
    "",
    "",
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
    parser.add_argument(
        "--paraphrase-prob",
        type=float,
        default=0.7,
        help="Probability of paraphrasing a template (0.0-1.0).",
    )
    parser.add_argument(
        "--filler-prob",
        type=float,
        default=0.5,
        help="Probability of adding filler/hedge to a turn.",
    )
    parser.add_argument(
        "--conversational-prob",
        type=float,
        default=0.3,
        help="Probability of adding conversational prefix after first turn.",
    )
    parser.add_argument(
        "--non-alternating-prob",
        type=float,
        default=0.2,
        help="Probability of speaker repeating (non-alternating turns).",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _apply_paraphrase(
    template: str,
    paraphrase_pool: dict[str, list[str]],
    topic: str,
    counter_topic: str,
    prob: float,
) -> str:
    if random.random() < prob and template in paraphrase_pool:
        variant = random.choice(paraphrase_pool[template])
        return variant.format(topic=topic, counter_topic=counter_topic)
    return template.format(topic=topic, counter_topic=counter_topic)


def _apply_filler(text: str, prob: float) -> str:
    if random.random() < prob:
        prefix = random.choice(FILLERS)
        hedge = random.choice(HEDGES)
        return f"{prefix}{text}{hedge}."
    return text


def _apply_conversational_prefix(text: str, prob: float, turn_idx: int) -> str:
    if turn_idx > 0 and random.random() < prob:
        prefix = random.choice(CONVERSATIONAL_PREFIXES)
        if prefix:
            return f"{prefix}{text[0].lower()}{text[1:]}"
    return text


def build_noisy_dialogue(
    dialogue_id: str,
    topic: str,
    counter_topic: str,
    paraphrase_prob: float,
    filler_prob: float,
    conversational_prob: float,
    non_alternating_prob: float,
) -> list[dict[str, str]]:
    base_templates_a = list(PARAPHRASES_A.keys())
    base_templates_b = list(PARAPHRASES_B.keys())

    turns: list[dict[str, str]] = []
    current_speaker = "Alice"

    for idx, (a_tmpl, b_tmpl) in enumerate(zip(base_templates_a, base_templates_b)):
        # Alice's turn
        text_a = _apply_paraphrase(
            a_tmpl, PARAPHRASES_A, topic, counter_topic, paraphrase_prob
        )
        text_a = _apply_filler(text_a, filler_prob)
        text_a = _apply_conversational_prefix(text_a, conversational_prob, idx)
        turns.append({"speaker": "Alice", "text": text_a})

        # Decide if Bob responds or Alice continues
        if random.random() < non_alternating_prob:
            current_speaker = "Alice"
        else:
            current_speaker = "Bob"

        # Bob's turn (or Alice's continuation)
        if current_speaker == "Bob":
            text_b = _apply_paraphrase(
                b_tmpl, PARAPHRASES_B, topic, counter_topic, paraphrase_prob
            )
            text_b = _apply_filler(text_b, filler_prob)
            text_b = _apply_conversational_prefix(text_b, conversational_prob, idx)
            turns.append({"speaker": "Bob", "text": text_b})
        else:
            # Alice speaks again (use next Alice template if available)
            if idx + 1 < len(base_templates_a):
                next_a_tmpl = base_templates_a[idx + 1]
                text_a2 = _apply_paraphrase(
                    next_a_tmpl, PARAPHRASES_A, topic, counter_topic, paraphrase_prob
                )
                text_a2 = _apply_filler(text_a2, filler_prob)
                turns.append({"speaker": "Alice", "text": text_a2})

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
    random.seed(args.seed)

    dialogues: list[dict[str, object]] = []
    for idx in range(args.num_dialogues):
        topic, counter_topic = TOPIC_PAIRS[idx % len(TOPIC_PAIRS)]
        transcript_id = f"d_{idx:03d}"
        base_turns = build_noisy_dialogue(
            transcript_id,
            topic=topic,
            counter_topic=counter_topic,
            paraphrase_prob=args.paraphrase_prob,
            filler_prob=args.filler_prob,
            conversational_prob=args.conversational_prob,
            non_alternating_prob=args.non_alternating_prob,
        )
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
