#!/usr/bin/env python3
"""Generate long Alice/Bob transcripts using an LLM."""

from __future__ import annotations

import argparse
import json
import os
import random
import re
from pathlib import Path
from typing import Any
from urllib import error as urlerror
from urllib import request as urlrequest


DEFAULT_TOPICS = [
    "whether cities should prioritize climate adaptation funding over new infrastructure expansion",
    "the tradeoff between strict AI safety regulation and open innovation",
    "how public schools should balance standardized testing with project-based learning",
    "whether universal basic income improves labor outcomes in the long run",
    "the role of zoning reform in reducing housing costs",
    "whether governments should phase out fossil-fuel subsidies immediately",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-dialogues", type=int, default=20)
    parser.add_argument(
        "--num-turns",
        type=int,
        default=120,
        help="Number of turns per dialogue.",
    )
    parser.add_argument(
        "--min-words-per-turn",
        type=int,
        default=30,
        help="Target minimum words for each turn.",
    )
    parser.add_argument(
        "--max-words-per-turn",
        type=int,
        default=80,
        help="Target maximum words for each turn.",
    )
    parser.add_argument(
        "--topics",
        type=str,
        default="",
        help="Optional comma-separated topic list. Defaults to built-in topics.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--api-base",
        type=str,
        default="https://api.openai.com/v1",
        help="OpenAI-compatible API base URL.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o-mini",
        help="Chat model name at the selected API.",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default="",
        help="API key (defaults to OPENAI_API_KEY env var).",
    )
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument(
        "--max-completion-tokens",
        type=int,
        default=16000,
        help="Maximum completion tokens per generation request.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Retries if output is invalid.",
    )
    return parser.parse_args()


def speaker_swap(turns: list[dict[str, str]]) -> list[dict[str, str]]:
    swapped: list[dict[str, str]] = []
    for turn in turns:
        speaker = "Bob" if turn["speaker"] == "Alice" else "Alice"
        swapped.append({"speaker": speaker, "text": turn["text"]})
    return swapped


def _strip_code_fences(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```[a-zA-Z0-9_-]*\n?", "", stripped, count=1).strip()
        stripped = re.sub(r"\n?```$", "", stripped, count=1).strip()
    return stripped


def _extract_text_from_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        chunks: list[str] = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str) and text.strip():
                    chunks.append(text)
        return "\n".join(chunks).strip()
    return ""


def _extract_json_candidate(raw_text: str) -> Any:
    text = _strip_code_fences(raw_text).strip()
    if not text:
        raise ValueError("Model output is empty.")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Fallback: parse first balanced JSON array from free-form text.
    start = text.find("[")
    if start == -1:
        raise ValueError("No JSON array start '[' found in model output.")
    depth = 0
    in_string = False
    escaped = False
    for idx in range(start, len(text)):
        ch = text[idx]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
            if depth == 0:
                return json.loads(text[start : idx + 1])
    raise ValueError("Could not find balanced JSON array in model output.")


def _word_count(text: str) -> int:
    return len([w for w in text.split() if w.strip()])


def _validate_turns(
    turns: list[dict[str, Any]],
    num_turns: int,
    min_words_per_turn: int,
) -> list[dict[str, str]]:
    if len(turns) < num_turns:
        raise ValueError(f"Expected at least {num_turns} turns, got {len(turns)}.")
    if len(turns) > num_turns:
        turns = turns[:num_turns]

    validated: list[dict[str, str]] = []
    for idx, turn in enumerate(turns):
        if not isinstance(turn, dict):
            raise ValueError(f"Turn {idx} is not an object.")
        speaker = str(turn.get("speaker", "")).strip()
        text = " ".join(str(turn.get("text", "")).split()).strip()
        if speaker not in {"Alice", "Bob"}:
            raise ValueError(f"Turn {idx} has invalid speaker '{speaker}'.")
        if not text:
            raise ValueError(f"Turn {idx} has empty text.")
        if _word_count(text) < min_words_per_turn:
            raise ValueError(
                f"Turn {idx} too short ({_word_count(text)} words, need >= {min_words_per_turn})."
            )
        validated.append({"speaker": speaker, "text": text})
    return validated


def _build_messages(
    topic: str,
    num_turns: int,
    min_words_per_turn: int,
    max_words_per_turn: int,
) -> list[dict[str, str]]:
    system = (
        "You generate realistic long-form two-speaker transcripts for research. "
        "You must return only valid JSON and no markdown."
    )
    user = (
        "Generate a transcript between Alice and Bob.\n"
        f"Topic: {topic}\n"
        f"Turns: exactly {num_turns}\n"
        f"Each turn length: roughly {min_words_per_turn} to {max_words_per_turn} words\n"
        "Constraints:\n"
        "- Output a JSON array only.\n"
        '- Each element is {"speaker": "...", "text": "..."}.\n'
        '- speaker must be exactly "Alice" or "Bob".\n'
        "- Do not include speaker names inside text.\n"
        "- The conversation should be coherent, detailed, and naturally argumentative.\n"
        "- No preamble, no explanation, no code fences."
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _call_openai_compatible(
    api_base: str,
    api_key: str,
    model: str,
    messages: list[dict[str, str]],
    temperature: float,
    max_completion_tokens: int,
) -> str:
    if not api_key:
        raise ValueError("Missing API key. Pass --api-key or set OPENAI_API_KEY.")
    endpoint = api_base.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_completion_tokens": max_completion_tokens,
        # Many OpenAI-compatible providers use max_tokens.
        "max_tokens": max_completion_tokens,
    }
    if "openrouter.ai" in api_base:
        payload["response_format"] = {"type": "json_object"}
    req = urlrequest.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        method="POST",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": os.getenv("OPENROUTER_SITE_URL", "https://github.com/certainforest/role-representation"),
            "X-Title": os.getenv("OPENROUTER_APP_NAME", "role-representation"),
        },
    )
    try:
        with urlrequest.urlopen(req, timeout=180) as response:
            body = response.read().decode("utf-8")
    except urlerror.HTTPError as exc:  # pragma: no cover
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"LLM API HTTP {exc.code}: {detail}") from exc
    except urlerror.URLError as exc:  # pragma: no cover
        raise RuntimeError(f"LLM API request failed: {exc}") from exc

    data = json.loads(body)
    choices = data.get("choices", [])
    if not choices:
        raise RuntimeError("LLM API returned no choices.")
    message = choices[0].get("message", {})
    content = _extract_text_from_content(message.get("content", ""))
    if not content.strip():
        raise RuntimeError("LLM API returned empty content.")
    return content


def _generate_one_dialogue(
    api_base: str,
    api_key: str,
    model: str,
    topic: str,
    num_turns: int,
    min_words_per_turn: int,
    max_words_per_turn: int,
    temperature: float,
    max_completion_tokens: int,
    max_retries: int,
) -> list[dict[str, str]]:
    messages = _build_messages(
        topic=topic,
        num_turns=num_turns,
        min_words_per_turn=min_words_per_turn,
        max_words_per_turn=max_words_per_turn,
    )
    for attempt in range(1, max_retries + 1):
        content = _call_openai_compatible(
            api_base=api_base,
            api_key=api_key,
            model=model,
            messages=messages,
            temperature=temperature,
            max_completion_tokens=max_completion_tokens,
        )
        raw = _strip_code_fences(content)
        try:
            parsed = _extract_json_candidate(raw)
            if isinstance(parsed, dict):
                # Permit object wrappers from strict JSON mode.
                if isinstance(parsed.get("turns"), list):
                    parsed = parsed["turns"]
                elif isinstance(parsed.get("dialogue"), list):
                    parsed = parsed["dialogue"]
            if not isinstance(parsed, list):
                raise ValueError("Top-level JSON must be an array.")
            return _validate_turns(
                turns=parsed,
                num_turns=num_turns,
                min_words_per_turn=min_words_per_turn,
            )
        except Exception as exc:
            if attempt == max_retries:
                raise RuntimeError(
                    f"Failed to parse/validate model output after {max_retries} attempts: {exc}"
                ) from exc
            messages = messages + [
                {"role": "assistant", "content": content},
                {
                    "role": "user",
                    "content": (
                        "Your prior output was invalid. "
                        "Return only a valid JSON array with exactly the required schema and length."
                    ),
                },
            ]
    raise RuntimeError("Unreachable.")


def main() -> None:
    args = parse_args()
    if args.num_dialogues <= 0:
        raise ValueError("--num-dialogues must be positive.")
    if args.num_turns < 2:
        raise ValueError("--num-turns must be >= 2.")
    if args.min_words_per_turn <= 0 or args.max_words_per_turn < args.min_words_per_turn:
        raise ValueError("Invalid word range. Ensure 0 < min <= max.")

    api_key = args.api_key.strip() or os.getenv("OPENAI_API_KEY", "").strip()
    topics = [t.strip() for t in args.topics.split(",") if t.strip()] or DEFAULT_TOPICS
    rng = random.Random(args.seed)

    dialogues: list[dict[str, object]] = []
    for idx in range(args.num_dialogues):
        topic = topics[idx % len(topics)]
        # Topic jitter to reduce near-duplicates across many generated dialogues.
        topic_variant = topic
        if len(topics) == 1:
            topic_variant = f"{topic} (dialogue perspective {idx + 1})"
        turns = _generate_one_dialogue(
            api_base=args.api_base,
            api_key=api_key,
            model=args.model,
            topic=topic_variant,
            num_turns=args.num_turns,
            min_words_per_turn=args.min_words_per_turn,
            max_words_per_turn=args.max_words_per_turn,
            temperature=max(0.0, min(2.0, args.temperature + rng.uniform(-0.1, 0.1))),
            max_completion_tokens=args.max_completion_tokens,
            max_retries=args.max_retries,
        )
        transcript_id = f"llm_{idx:03d}"
        dialogues.append(
            {
                "transcript_id": transcript_id,
                "topic": topic_variant,
                "base": turns,
                "speaker_swapped": speaker_swap(turns),
            }
        )
        print(f"[{idx + 1}/{args.num_dialogues}] generated {transcript_id}")

    payload = {
        "metadata": {
            "generator": "mvp_generate_llm_dialogues.py",
            "model": args.model,
            "api_base": args.api_base,
            "num_dialogues": args.num_dialogues,
            "num_turns": args.num_turns,
            "min_words_per_turn": args.min_words_per_turn,
            "max_words_per_turn": args.max_words_per_turn,
            "seed": args.seed,
            "topics": topics,
        },
        "dialogues": dialogues,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=True)


if __name__ == "__main__":
    main()
