#!/usr/bin/env python3
"""Generate experiment-specific Alice/Bob dialogues using an LLM.

Three experiment modes:

  agreement_pivot   – Standard argumentative debate where Alice and Bob converge
                      to agreement around a configurable turn window.
  similar_neutral   – Alice and Bob share similar demographic/ideological profiles
                      and have a neutral, non-argumentative conversation.
  similar_polarize  – Alice and Bob share similar profiles but reinforce each
                      other's views, escalating in the same ideological direction.

Output format is identical to mvp_generate_llm_dialogues.py so all downstream
analysis scripts (extraction, geometry, probes) work unchanged.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import time
from pathlib import Path
from typing import Any
from urllib import error as urlerror
from urllib import request as urlrequest


# ---------------------------------------------------------------------------
# Default profiles & topics
# ---------------------------------------------------------------------------

DEFAULT_TOPICS = [
    "whether cities should prioritize climate adaptation funding over new infrastructure expansion",
    "the tradeoff between strict AI safety regulation and open innovation",
    "how public schools should balance standardized testing with project-based learning",
    "whether universal basic income improves labor outcomes in the long run",
    "the role of zoning reform in reducing housing costs",
    "whether governments should phase out fossil-fuel subsidies immediately",
]

SIMILAR_PROFILES = [
    {
        "alice": "Alice is a 34-year-old progressive Democrat living in Los Angeles. She works in urban planning, values sustainability, and volunteers for local community gardens.",
        "bob": "Bob is a 36-year-old progressive Democrat living in Los Angeles. He works in environmental policy, values sustainability, and coaches youth soccer in his neighborhood.",
    },
    {
        "alice": "Alice is a 29-year-old left-leaning independent living in Brooklyn, NY. She is a public-school teacher, advocates for education equity, and enjoys cycling.",
        "bob": "Bob is a 31-year-old left-leaning independent living in Brooklyn, NY. He is a social worker, advocates for housing justice, and is an avid runner.",
    },
    {
        "alice": "Alice is a 40-year-old liberal Democrat living in San Francisco. She is a tech ethics researcher, supports universal healthcare, and volunteers at a food bank.",
        "bob": "Bob is a 42-year-old liberal Democrat living in San Francisco. He is a data scientist at a nonprofit, supports universal healthcare, and mentors first-generation college students.",
    },
    {
        "alice": "Alice is a 38-year-old center-left Democrat living in Austin, TX. She runs a small bookshop, supports public transit expansion, and is active in local arts advocacy.",
        "bob": "Bob is a 37-year-old center-left Democrat living in Austin, TX. He is a civil engineer, supports public transit expansion, and plays in a community jazz band.",
    },
]

NEUTRAL_TOPICS = [
    "how their neighborhood has changed over the past five years and what they hope to see next",
    "their favorite local restaurants and what makes a great neighborhood food scene",
    "weekend routines, hobbies, and how they recharge after a busy work week",
    "the best parks and outdoor spaces in their city and how they use them",
    "what they've been reading or watching lately and why it resonated with them",
    "how they got into their current careers and what keeps them motivated",
]

POLARIZE_SAME_TOPICS = [
    "how corporate lobbying is the root cause of climate inaction and what radical policy should replace the status quo",
    "why the US healthcare system is fundamentally broken and needs single-payer reform immediately",
    "how tech monopolies are eroding democracy and what aggressive antitrust measures are needed",
    "why the criminal justice system must be completely restructured around restorative practices",
    "how wealth inequality has reached a tipping point and what redistribution policies are morally required",
    "why standardized testing is destroying public education and what should replace it",
]


# ---------------------------------------------------------------------------
# Individuation experiment profiles & topics (paired-comparison design)
# ---------------------------------------------------------------------------

INDIVIDUATION_PROFILES = [
    {
        "alice_bio": (
            "{name} is a 34-year-old software engineer from Portland who owns a golden "
            "retriever named Blueberry, is married to Alex (a librarian), enjoys hiking "
            "in Forest Park on weekends, and recently started learning to bake sourdough."
        ),
        "bob_bio": (
            "{name} is a 41-year-old veterinarian from Portland who has a german shepherd "
            "mix named Frosting, is married to Kristi (a physical therapist), coaches youth "
            "soccer on Saturdays, and is an avid home cook who specializes in Thai food."
        ),
        "collision_name": "Michael",
    },
    {
        "alice_bio": (
            "{name} is a 29-year-old graphic designer from Chicago who plays guitar in a "
            "local band called The Ruminants, recently moved to Logan Square, has a cat "
            "named Pixel, and is training for her first triathlon."
        ),
        "bob_bio": (
            "{name} is a 32-year-old emergency room nurse from Chicago who plays piano, "
            "has lived in Logan Square for eight years, volunteers at the PAWS animal "
            "shelter, and just finished building a home recording studio."
        ),
        "collision_name": "Sarah",
    },
    {
        "alice_bio": (
            "{name} is a 38-year-old high school English teacher from Denver who runs "
            "marathons, has two kids named Leo and Maya, is restoring a 1970s Volkswagen "
            "Beetle in the garage, and chairs the neighborhood book club."
        ),
        "bob_bio": (
            "{name} is a 36-year-old freelance journalist from Denver who does rock "
            "climbing, recently adopted a rescue greyhound named Dash, brews kombucha "
            "at home, and is writing a book about Colorado's water politics."
        ),
        "collision_name": "David",
    },
    {
        "alice_bio": (
            "{name} is a 40-year-old restaurant owner from Seattle whose partner Roni is "
            "a jazz musician, has maintained the same sourdough starter for eight years, "
            "loves kayaking on Puget Sound, and collects vintage cookbooks."
        ),
        "bob_bio": (
            "{name} is a 43-year-old architect from Seattle whose wife Linda runs a "
            "bookshop, brews craft beer at home, recently took up pottery at a local "
            "studio, and serves on the city's design review board."
        ),
        "collision_name": "Jasmine",
    },
]

INDIVIDUATION_TOPICS = [
    "their pets, favorite neighborhood spots, and how their part of the city has changed recently",
    "weekend routines, hobbies they have picked up lately, and how they unwind after busy weeks",
    "career paths, memorable moments at work, and what keeps them motivated day to day",
    "favorite local restaurants, cooking adventures at home, and food trends they have noticed",
    "travel stories, upcoming trip plans, and how travel has shaped their perspectives",
    "how technology has changed their daily routines and the tools or apps they rely on most",
]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate experiment-specific Alice/Bob dialogues.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--experiment",
        type=str,
        required=True,
        choices=(
            "agreement_pivot", "similar_neutral", "similar_polarize",
            "distinct_names", "name_collision", "quote_intrusion", "cue_corrupted",
        ),
        help="Experiment type to generate.",
    )
    parser.add_argument("--num-dialogues", type=int, default=20)
    parser.add_argument("--num-turns", type=int, default=120)
    parser.add_argument("--min-words-per-turn", type=int, default=30)
    parser.add_argument("--max-words-per-turn", type=int, default=80)
    parser.add_argument(
        "--agreement-turn-start",
        type=int,
        default=40,
        help="(agreement_pivot) Turn at which convergence begins.",
    )
    parser.add_argument(
        "--agreement-turn-end",
        type=int,
        default=50,
        help="(agreement_pivot) Turn by which full agreement is reached.",
    )
    parser.add_argument(
        "--quote-frequency",
        type=int,
        default=12,
        help="(quote_intrusion) Approx turn interval between explicit quotes.",
    )
    parser.add_argument(
        "--topics",
        type=str,
        default="",
        help="Comma-separated topic override. Defaults depend on experiment type.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--api-base", type=str, default="https://api.openai.com/v1")
    parser.add_argument("--model", type=str, default="gpt-4o-mini")
    parser.add_argument(
        "--api-key",
        type=str,
        default="",
        help="API key (defaults to OPENAI_API_KEY or OPENROUTER_API_KEY env var).",
    )
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--max-completion-tokens", type=int, default=16000)
    parser.add_argument("--max-retries", type=int, default=6)
    parser.add_argument("--min-words-tolerance", type=int, default=3)
    parser.add_argument("--chunk-turns", type=int, default=20)
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Shared LLM plumbing (mirrors mvp_generate_llm_dialogues.py)
# ---------------------------------------------------------------------------

def speaker_swap(turns: list[dict[str, str]]) -> list[dict[str, str]]:
    return [
        {"speaker": ("Bob" if t["speaker"] == "Alice" else "Alice"), "text": t["text"]}
        for t in turns
    ]


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


def _normalize_api_key(raw_value: str) -> str:
    """Defensively normalize accidentally pasted multi-line API keys."""
    token = (raw_value or "").strip()
    if not token:
        return ""
    # Keep only the first logical line and first token on that line.
    # This avoids invalid header values if users paste notes after a key.
    token = token.splitlines()[0].strip()
    token = token.split()[0].strip()
    return token


def _word_count(text: str) -> int:
    return len([w for w in text.split() if w.strip()])


def _validate_turns(
    turns: list[dict[str, Any]],
    num_turns: int,
    min_words_per_turn: int,
    min_words_tolerance: int,
    min_required_turns: int | None = None,
) -> list[dict[str, str]]:
    if min_required_turns is None:
        min_required_turns = num_turns
    min_required_turns = max(1, min_required_turns)
    if len(turns) < min_required_turns:
        raise ValueError(f"Expected at least {min_required_turns} turns, got {len(turns)}.")
    if len(turns) > num_turns:
        turns = turns[:num_turns]

    validated: list[dict[str, str]] = []
    effective_min_words = max(1, min_words_per_turn - max(0, min_words_tolerance))
    # Small slack avoids brittle failures when the model misses by a single word.
    effective_min_words_with_slack = max(1, effective_min_words - 1)
    word_counts: list[int] = []
    for idx, turn in enumerate(turns):
        if not isinstance(turn, dict):
            raise ValueError(f"Turn {idx} is not an object.")
        speaker = str(turn.get("speaker", "")).strip()
        text = " ".join(str(turn.get("text", "")).split()).strip()
        if speaker not in {"Alice", "Bob"}:
            raise ValueError(f"Turn {idx} has invalid speaker '{speaker}'.")
        if not text:
            raise ValueError(f"Turn {idx} has empty text.")
        wc = _word_count(text)
        word_counts.append(wc)
        if wc < effective_min_words_with_slack:
            raise ValueError(f"Turn {idx} too short ({wc} words, need >= {effective_min_words}).")
        validated.append({"speaker": speaker, "text": text})
    mean_words = sum(word_counts) / float(len(word_counts))
    if mean_words < float(effective_min_words):
        raise ValueError(f"Average turn length too short ({mean_words:.1f}, need >= {min_words_per_turn}).")
    return validated


def _call_openai_compatible(
    api_base: str,
    api_key: str,
    model: str,
    messages: list[dict[str, str]],
    temperature: float,
    max_completion_tokens: int,
) -> str:
    api_key = _normalize_api_key(api_key)
    if not api_key:
        raise ValueError(
            "Missing API key. Pass --api-key or set OPENAI_API_KEY/OPENROUTER_API_KEY."
        )
    endpoint = api_base.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_completion_tokens": max_completion_tokens,
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
            "HTTP-Referer": os.getenv(
                "OPENROUTER_SITE_URL",
                "https://github.com/certainforest/role-representation",
            ),
            "X-Title": os.getenv("OPENROUTER_APP_NAME", "role-representation"),
        },
    )
    try:
        with urlrequest.urlopen(req, timeout=180) as response:
            body = response.read().decode("utf-8")
    except urlerror.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"LLM API HTTP {exc.code}: {detail}") from exc
    except urlerror.URLError as exc:
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


# ---------------------------------------------------------------------------
# Prompt builders (experiment-specific)
# ---------------------------------------------------------------------------

def _build_messages_agreement_pivot(
    *,
    topic: str,
    num_turns: int,
    min_words: int,
    max_words: int,
    agree_start: int,
    agree_end: int,
    prior_turns: list[dict[str, str]] | None = None,
    required_first_speaker: str = "",
    current_turn_offset: int = 0,
) -> list[dict[str, str]]:
    """Prompt for a debate that converges to agreement at a specified turn window."""
    system = (
        "You generate realistic long-form two-speaker transcripts for research. "
        "You must return only valid JSON and no markdown."
    )
    phase_instructions = (
        f"The dialogue has THREE phases:\n"
        f"  Phase 1 (turns 1–{agree_start - 1}): Alice and Bob DISAGREE on the topic. "
        f"They argue from opposing sides with substantive, detailed points. "
        f"The tone is respectful but firm; neither concedes.\n"
        f"  Phase 2 (turns {agree_start}–{agree_end}): A TRANSITION occurs. "
        f"One speaker acknowledges a strong point from the other. "
        f"They find common ground and gradually converge. "
        f"The shift should feel organic, not abrupt.\n"
        f"  Phase 3 (turns {agree_end + 1}–{num_turns}): Alice and Bob are IN AGREEMENT. "
        f"They build on each other's ideas collaboratively, exploring implications "
        f"and next steps. The tone is cooperative and constructive.\n"
    )
    continuation = ""
    if prior_turns:
        continuation = (
            "\nContinuation context (already generated turns, do not repeat):\n"
            f"{json.dumps(prior_turns[-6:], ensure_ascii=True)}\n"
            f"The next turn to generate is turn {current_turn_offset + 1} "
            f"(1-indexed). Follow the phase rules above for this turn number.\n"
            "Generate the next turns only.\n"
        )
    first_speaker_req = (
        f'- The first generated turn must have speaker "{required_first_speaker}".\n'
        if required_first_speaker
        else ""
    )
    user = (
        "Generate a transcript between Alice and Bob.\n"
        f"Topic: {topic}\n"
        f"Turns: exactly {num_turns}\n"
        f"Each turn length: roughly {min_words} to {max_words} words\n"
        f"\n{phase_instructions}\n"
        f"{continuation}"
        "Constraints:\n"
        "- Output a JSON array only.\n"
        '- Each element is {{"speaker": "...", "text": "..."}}.\n'
        '- speaker must be exactly "Alice" or "Bob".\n'
        f"{first_speaker_req}"
        "- Do not include speaker names inside text.\n"
        "- No preamble, no explanation, no code fences."
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _build_messages_similar_neutral(
    *,
    topic: str,
    num_turns: int,
    min_words: int,
    max_words: int,
    alice_profile: str,
    bob_profile: str,
    prior_turns: list[dict[str, str]] | None = None,
    required_first_speaker: str = "",
) -> list[dict[str, str]]:
    """Prompt for a neutral conversation between two demographically similar people."""
    system = (
        "You generate realistic long-form two-speaker transcripts for research. "
        "You must return only valid JSON and no markdown."
    )
    continuation = ""
    if prior_turns:
        continuation = (
            "\nContinuation context (already generated turns, do not repeat):\n"
            f"{json.dumps(prior_turns[-6:], ensure_ascii=True)}\n"
            "Generate the next turns only.\n"
        )
    first_speaker_req = (
        f'- The first generated turn must have speaker "{required_first_speaker}".\n'
        if required_first_speaker
        else ""
    )
    user = (
        "Generate a transcript between Alice and Bob.\n\n"
        f"ALICE'S PROFILE: {alice_profile}\n"
        f"BOB'S PROFILE: {bob_profile}\n\n"
        f"Topic: {topic}\n"
        f"Turns: exactly {num_turns}\n"
        f"Each turn length: roughly {min_words} to {max_words} words\n\n"
        "IMPORTANT STYLE INSTRUCTIONS:\n"
        "- This is a casual, friendly, NEUTRAL conversation. NOT a debate.\n"
        "- Alice and Bob are like-minded neighbors or colleagues chatting.\n"
        "- They share observations, personal anecdotes, ask each other questions, "
        "and occasionally mildly disagree on details but never argue.\n"
        "- The tone is warm, relaxed, and cooperative throughout.\n"
        "- They should reference aspects of their profiles naturally (work, neighborhood, hobbies).\n"
        f"\n{continuation}"
        "Constraints:\n"
        "- Output a JSON array only.\n"
        '- Each element is {{"speaker": "...", "text": "..."}}.\n'
        '- speaker must be exactly "Alice" or "Bob".\n'
        f"{first_speaker_req}"
        "- Do not include speaker names inside text.\n"
        "- No preamble, no explanation, no code fences."
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _build_messages_similar_polarize(
    *,
    topic: str,
    num_turns: int,
    min_words: int,
    max_words: int,
    alice_profile: str,
    bob_profile: str,
    prior_turns: list[dict[str, str]] | None = None,
    required_first_speaker: str = "",
) -> list[dict[str, str]]:
    """Prompt for an echo-chamber dialogue where similar people reinforce each other."""
    system = (
        "You generate realistic long-form two-speaker transcripts for research. "
        "You must return only valid JSON and no markdown."
    )
    continuation = ""
    if prior_turns:
        continuation = (
            "\nContinuation context (already generated turns, do not repeat):\n"
            f"{json.dumps(prior_turns[-6:], ensure_ascii=True)}\n"
            "Generate the next turns only.\n"
        )
    first_speaker_req = (
        f'- The first generated turn must have speaker "{required_first_speaker}".\n'
        if required_first_speaker
        else ""
    )
    user = (
        "Generate a transcript between Alice and Bob.\n\n"
        f"ALICE'S PROFILE: {alice_profile}\n"
        f"BOB'S PROFILE: {bob_profile}\n\n"
        f"Topic: {topic}\n"
        f"Turns: exactly {num_turns}\n"
        f"Each turn length: roughly {min_words} to {max_words} words\n\n"
        "IMPORTANT STYLE INSTRUCTIONS:\n"
        "- Alice and Bob fundamentally AGREE on this topic from the start.\n"
        "- Over the course of the conversation they REINFORCE and ESCALATE each other's views.\n"
        "- Each speaker builds on the other's points, adds more extreme examples, "
        "and pushes the shared position further.\n"
        "- This is an echo-chamber dynamic: mutual validation, growing conviction, "
        "increasing rhetorical intensity.\n"
        "- They may start moderate but by the end should hold a strongly amplified shared position.\n"
        "- They should reference aspects of their profiles naturally.\n"
        "- Despite agreeing, each turn should add NEW substantive content (facts, anecdotes, "
        "policy proposals) rather than just echoing.\n"
        f"\n{continuation}"
        "Constraints:\n"
        "- Output a JSON array only.\n"
        '- Each element is {{"speaker": "...", "text": "..."}}.\n'
        '- speaker must be exactly "Alice" or "Bob".\n'
        f"{first_speaker_req}"
        "- Do not include speaker names inside text.\n"
        "- No preamble, no explanation, no code fences."
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


# ---------------------------------------------------------------------------
# Individuation prompt builders (paired-comparison experiments)
# ---------------------------------------------------------------------------


def _build_messages_distinct_names(
    *,
    topic: str,
    num_turns: int,
    min_words: int,
    max_words: int,
    alice_profile: str,
    bob_profile: str,
    prior_turns: list[dict[str, str]] | None = None,
    required_first_speaker: str = "",
) -> list[dict[str, str]]:
    """Reference condition: two speakers with distinct names and rich profiles."""
    system = (
        "You generate realistic long-form two-speaker transcripts for research. "
        "You must return only valid JSON and no markdown."
    )
    continuation = ""
    if prior_turns:
        continuation = (
            "\nContinuation context (already generated turns, do not repeat):\n"
            f"{json.dumps(prior_turns[-6:], ensure_ascii=True)}\n"
            "Generate the next turns only.\n"
        )
    first_speaker_req = (
        f'- The first generated turn must have speaker "{required_first_speaker}".\n'
        if required_first_speaker
        else ""
    )
    user = (
        "Generate a transcript between Alice and Bob.\n\n"
        f"ALICE'S PROFILE: {alice_profile}\n"
        f"BOB'S PROFILE: {bob_profile}\n\n"
        f"Topic: {topic}\n"
        f"Turns: exactly {num_turns}\n"
        f"Each turn length: roughly {min_words} to {max_words} words\n\n"
        "STYLE INSTRUCTIONS:\n"
        "- This is a natural, casual conversation between two acquaintances.\n"
        "- Each speaker has their own distinct personality and speech patterns "
        "reflecting their background.\n"
        "- They share anecdotes, ask questions, and react to each other naturally.\n"
        "- The tone is warm and conversational throughout.\n"
        "- Speakers should reference aspects of their profiles (work, family, pets, "
        "hobbies) naturally and frequently.\n"
        "- They should address each other by name occasionally.\n"
        f"\n{continuation}"
        "Constraints:\n"
        "- Output a JSON array only.\n"
        '- Each element is {{"speaker": "...", "text": "..."}}.\n'
        '- speaker must be exactly "Alice" or "Bob".\n'
        f"{first_speaker_req}"
        "- Do not include speaker labels inside text.\n"
        "- No preamble, no explanation, no code fences."
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _build_messages_name_collision(
    *,
    topic: str,
    num_turns: int,
    min_words: int,
    max_words: int,
    alice_profile: str,
    bob_profile: str,
    collision_name: str,
    prior_turns: list[dict[str, str]] | None = None,
    required_first_speaker: str = "",
) -> list[dict[str, str]]:
    """Two speakers sharing the same first name, distinguished only by context."""
    system = (
        "You generate realistic long-form two-speaker transcripts for research. "
        "You must return only valid JSON and no markdown."
    )
    continuation = ""
    if prior_turns:
        continuation = (
            "\nContinuation context (already generated turns, do not repeat):\n"
            f"{json.dumps(prior_turns[-6:], ensure_ascii=True)}\n"
            "Generate the next turns only.\n"
        )
    first_speaker_req = (
        f'- The first generated turn must have speaker "{required_first_speaker}".\n'
        if required_first_speaker
        else ""
    )
    user = (
        f"Generate a conversation between two people BOTH named {collision_name}.\n\n"
        f"FIRST {collision_name.upper()}: {alice_profile}\n"
        f"SECOND {collision_name.upper()}: {bob_profile}\n\n"
        f"Topic: {topic}\n"
        f"Turns: exactly {num_turns}\n"
        f"Each turn length: roughly {min_words} to {max_words} words\n\n"
        "NAME COLLISION INSTRUCTIONS:\n"
        f"- Both speakers share the SAME first name: {collision_name}.\n"
        f'- In the dialogue text, both are called "{collision_name}".\n'
        "- They must be distinguished ONLY through contextual cues: their jobs, "
        "pets, families, hobbies, and past experiences.\n"
        f'- They occasionally address each other as "{collision_name}", creating '
        "natural ambiguity that context resolves.\n"
        "- Each speaker should build a distinct life context through stories, "
        "references to third parties (spouses, friends, pets by name), and "
        "personal details.\n"
        "- Do NOT use nicknames, last names, or titles to disambiguate.\n"
        f"\n{continuation}"
        "Constraints:\n"
        "- Output a JSON array only.\n"
        '- Each element is {{"speaker": "...", "text": "..."}}.\n'
        f'- Use "Alice" for the first {collision_name} and "Bob" for the second '
        f"{collision_name} in the speaker field.\n"
        f'- In the TEXT, both speakers are called "{collision_name}".\n'
        f"{first_speaker_req}"
        "- No preamble, no explanation, no code fences."
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _build_messages_quote_intrusion(
    *,
    topic: str,
    num_turns: int,
    min_words: int,
    max_words: int,
    alice_profile: str,
    bob_profile: str,
    quote_frequency: int,
    prior_turns: list[dict[str, str]] | None = None,
    required_first_speaker: str = "",
) -> list[dict[str, str]]:
    """Dialogue with periodic verbatim quotes of the other speaker."""
    system = (
        "You generate realistic long-form two-speaker transcripts for research. "
        "You must return only valid JSON and no markdown."
    )
    continuation = ""
    if prior_turns:
        continuation = (
            "\nContinuation context (already generated turns, do not repeat):\n"
            f"{json.dumps(prior_turns[-6:], ensure_ascii=True)}\n"
            "Generate the next turns only.\n"
        )
    first_speaker_req = (
        f'- The first generated turn must have speaker "{required_first_speaker}".\n'
        if required_first_speaker
        else ""
    )
    user = (
        "Generate a transcript between Alice and Bob.\n\n"
        f"ALICE'S PROFILE: {alice_profile}\n"
        f"BOB'S PROFILE: {bob_profile}\n\n"
        f"Topic: {topic}\n"
        f"Turns: exactly {num_turns}\n"
        f"Each turn length: roughly {min_words} to {max_words} words\n\n"
        "QUOTE INTRUSION INSTRUCTIONS:\n"
        f"- Approximately every {quote_frequency} turns, one speaker must EXPLICITLY "
        "QUOTE something the other speaker said earlier in the conversation.\n"
        "- Use direct quotation with clear attribution, for example:\n"
        '  - \'You mentioned earlier that "exact previous quote"...\'\n'
        '  - \'As you said, "exact previous quote", and I think...\'\n'
        '  - \'I keep thinking about what you said: "exact previous quote"\'\n'
        "- Quoted text must be a substantial phrase (8+ words) that the other "
        "speaker ACTUALLY said earlier.\n"
        "- Between quotes, the conversation flows naturally.\n"
        "- Both speakers should quote each other roughly equally.\n"
        "- Speakers should also reference their own profiles naturally.\n"
        f"\n{continuation}"
        "Constraints:\n"
        "- Output a JSON array only.\n"
        '- Each element is {{"speaker": "...", "text": "..."}}.\n'
        '- speaker must be exactly "Alice" or "Bob".\n'
        f"{first_speaker_req}"
        "- Do not include speaker labels inside text.\n"
        "- No preamble, no explanation, no code fences."
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _build_messages_cue_corrupted(
    *,
    topic: str,
    num_turns: int,
    min_words: int,
    max_words: int,
    alice_profile: str,
    bob_profile: str,
    prior_turns: list[dict[str, str]] | None = None,
    required_first_speaker: str = "",
) -> list[dict[str, str]]:
    """Dialogue where speakers use identical speech styles, removing stylistic cues."""
    system = (
        "You generate realistic long-form two-speaker transcripts for research. "
        "You must return only valid JSON and no markdown."
    )
    continuation = ""
    if prior_turns:
        continuation = (
            "\nContinuation context (already generated turns, do not repeat):\n"
            f"{json.dumps(prior_turns[-6:], ensure_ascii=True)}\n"
            "Generate the next turns only.\n"
        )
    first_speaker_req = (
        f'- The first generated turn must have speaker "{required_first_speaker}".\n'
        if required_first_speaker
        else ""
    )
    user = (
        "Generate a transcript between Alice and Bob.\n\n"
        f"ALICE'S PROFILE: {alice_profile}\n"
        f"BOB'S PROFILE: {bob_profile}\n\n"
        f"Topic: {topic}\n"
        f"Turns: exactly {num_turns}\n"
        f"Each turn length: roughly {min_words} to {max_words} words\n\n"
        "CRITICAL STYLE-MATCHING INSTRUCTIONS:\n"
        "- Both speakers must use IDENTICAL speech styles.\n"
        "- Same vocabulary level, sentence structure, tone, and formality.\n"
        "- No distinctive verbal tics, catchphrases, filler words, or patterns.\n"
        "- Sentence lengths should be similar across speakers.\n"
        "- The two speakers should sound stylistically INTERCHANGEABLE.\n"
        '- Do NOT use the other speaker\'s name in the text. Refer to them only as '
        '"you" or via indirect references.\n'
        "- They still have different profiles and share different life experiences, "
        "but HOW they speak must be identical.\n"
        "- Avoid gendered pronouns or other surface-level identifying markers.\n"
        f"\n{continuation}"
        "Constraints:\n"
        "- Output a JSON array only.\n"
        '- Each element is {{"speaker": "...", "text": "..."}}.\n'
        '- speaker must be exactly "Alice" or "Bob".\n'
        f"{first_speaker_req}"
        "- Do not include speaker names inside text.\n"
        "- No preamble, no explanation, no code fences."
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


# ---------------------------------------------------------------------------
# Generation core
# ---------------------------------------------------------------------------

def _generate_one_chunk(
    *,
    api_base: str,
    api_key: str,
    model: str,
    messages: list[dict[str, str]],
    num_turns: int,
    min_words_per_turn: int,
    min_words_tolerance: int,
    temperature: float,
    max_completion_tokens: int,
    max_retries: int,
    min_required_turns: int | None = None,
    required_first_speaker: str = "",
) -> list[dict[str, str]]:
    for attempt in range(1, max_retries + 1):
        content = ""
        try:
            content = _call_openai_compatible(
                api_base=api_base,
                api_key=api_key,
                model=model,
                messages=messages,
                temperature=temperature,
                max_completion_tokens=max_completion_tokens,
            )
            raw = _strip_code_fences(content)
            parsed = _extract_json_candidate(raw)
            if isinstance(parsed, dict):
                if isinstance(parsed.get("turns"), list):
                    parsed = parsed["turns"]
                elif isinstance(parsed.get("dialogue"), list):
                    parsed = parsed["dialogue"]
            if not isinstance(parsed, list):
                raise ValueError("Top-level JSON must be an array.")
            validated = _validate_turns(
                turns=parsed,
                num_turns=num_turns,
                min_words_per_turn=min_words_per_turn,
                min_words_tolerance=min_words_tolerance,
                min_required_turns=min_required_turns,
            )
            if required_first_speaker and validated[0]["speaker"] != required_first_speaker:
                raise ValueError(
                    f'First turn must be "{required_first_speaker}", got "{validated[0]["speaker"]}".'
                )
            return validated
        except Exception as exc:
            if attempt == max_retries:
                raise RuntimeError(
                    f"Failed to parse/validate model output after {max_retries} attempts: {exc}"
                ) from exc
            backoff_s = min(20, 2 ** (attempt - 1))
            print(
                f"[retry {attempt}/{max_retries}] generation failed: {exc}. "
                f"Sleeping {backoff_s}s before retry..."
            )
            time.sleep(backoff_s)
            messages = messages + [
                {"role": "assistant", "content": content},
                {
                    "role": "user",
                    "content": (
                        "Your prior output was invalid. "
                        f"Validation error: {exc}. "
                        "Fix the error exactly. "
                        "Return only a valid JSON array with exactly the required schema and length."
                    ),
                },
            ]
    raise RuntimeError("Unreachable.")


def _generate_dialogue_chunked(
    *,
    experiment: str,
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
    chunk_turns: int,
    min_words_tolerance: int,
    agree_start: int = 40,
    agree_end: int = 50,
    alice_profile: str = "",
    bob_profile: str = "",
    quote_frequency: int = 12,
    collision_name: str = "",
) -> list[dict[str, str]]:
    collected: list[dict[str, str]] = []

    while len(collected) < num_turns:
        remaining = num_turns - len(collected)
        target = min(chunk_turns, remaining) if chunk_turns > 0 else remaining

        required_first_speaker = ""
        if collected:
            required_first_speaker = "Bob" if collected[-1]["speaker"] == "Alice" else "Alice"

        current_offset = len(collected)

        if experiment == "agreement_pivot":
            messages = _build_messages_agreement_pivot(
                topic=topic,
                num_turns=target,
                min_words=min_words_per_turn,
                max_words=max_words_per_turn,
                agree_start=agree_start,
                agree_end=agree_end,
                prior_turns=collected if collected else None,
                required_first_speaker=required_first_speaker,
                current_turn_offset=current_offset,
            )
        elif experiment == "similar_neutral":
            messages = _build_messages_similar_neutral(
                topic=topic,
                num_turns=target,
                min_words=min_words_per_turn,
                max_words=max_words_per_turn,
                alice_profile=alice_profile,
                bob_profile=bob_profile,
                prior_turns=collected if collected else None,
                required_first_speaker=required_first_speaker,
            )
        elif experiment == "similar_polarize":
            messages = _build_messages_similar_polarize(
                topic=topic,
                num_turns=target,
                min_words=min_words_per_turn,
                max_words=max_words_per_turn,
                alice_profile=alice_profile,
                bob_profile=bob_profile,
                prior_turns=collected if collected else None,
                required_first_speaker=required_first_speaker,
            )
        elif experiment == "distinct_names":
            messages = _build_messages_distinct_names(
                topic=topic,
                num_turns=target,
                min_words=min_words_per_turn,
                max_words=max_words_per_turn,
                alice_profile=alice_profile,
                bob_profile=bob_profile,
                prior_turns=collected if collected else None,
                required_first_speaker=required_first_speaker,
            )
        elif experiment == "name_collision":
            messages = _build_messages_name_collision(
                topic=topic,
                num_turns=target,
                min_words=min_words_per_turn,
                max_words=max_words_per_turn,
                alice_profile=alice_profile,
                bob_profile=bob_profile,
                collision_name=collision_name,
                prior_turns=collected if collected else None,
                required_first_speaker=required_first_speaker,
            )
        elif experiment == "quote_intrusion":
            messages = _build_messages_quote_intrusion(
                topic=topic,
                num_turns=target,
                min_words=min_words_per_turn,
                max_words=max_words_per_turn,
                alice_profile=alice_profile,
                bob_profile=bob_profile,
                quote_frequency=quote_frequency,
                prior_turns=collected if collected else None,
                required_first_speaker=required_first_speaker,
            )
        elif experiment == "cue_corrupted":
            messages = _build_messages_cue_corrupted(
                topic=topic,
                num_turns=target,
                min_words=min_words_per_turn,
                max_words=max_words_per_turn,
                alice_profile=alice_profile,
                bob_profile=bob_profile,
                prior_turns=collected if collected else None,
                required_first_speaker=required_first_speaker,
            )
        else:
            raise ValueError(f"Unknown experiment: {experiment}")

        min_required = min(target, 2)
        chunk = _generate_one_chunk(
            api_base=api_base,
            api_key=api_key,
            model=model,
            messages=messages,
            num_turns=target,
            min_words_per_turn=min_words_per_turn,
            min_words_tolerance=min_words_tolerance,
            temperature=temperature,
            max_completion_tokens=max_completion_tokens,
            max_retries=max_retries,
            min_required_turns=min_required,
            required_first_speaker=required_first_speaker,
        )
        collected.extend(chunk)

    return collected[:num_turns]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _build_payload(
    *,
    args: argparse.Namespace,
    topics: list[str],
    profiles: list[dict[str, str]],
    needs_profiles: bool,
    dialogues: list[dict[str, object]],
) -> dict[str, object]:
    payload: dict[str, object] = {
        "metadata": {
            "generator": "mvp_generate_experiment_dialogues.py",
            "experiment": args.experiment,
            "model": args.model,
            "api_base": args.api_base,
            "num_dialogues": args.num_dialogues,
            "num_turns": args.num_turns,
            "min_words_per_turn": args.min_words_per_turn,
            "max_words_per_turn": args.max_words_per_turn,
            "min_words_tolerance": args.min_words_tolerance,
            "chunk_turns": args.chunk_turns,
            "seed": args.seed,
            "topics": topics,
        },
        "dialogues": dialogues,
    }
    if args.experiment == "agreement_pivot":
        payload["metadata"]["agreement_turn_start"] = args.agreement_turn_start
        payload["metadata"]["agreement_turn_end"] = args.agreement_turn_end
    if needs_profiles:
        payload["metadata"]["profiles_used"] = profiles
    return payload


def _load_existing_dialogues(output_path: Path, experiment: str) -> list[dict[str, object]]:
    if not output_path.exists():
        return []
    try:
        with output_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except Exception:
        return []
    existing = payload.get("dialogues", [])
    if not isinstance(existing, list):
        return []
    out: list[dict[str, object]] = []
    prefix = f"{experiment}_"
    for row in existing:
        if not isinstance(row, dict):
            continue
        tid = str(row.get("transcript_id", ""))
        if tid.startswith(prefix):
            out.append(row)
    out.sort(key=lambda r: str(r.get("transcript_id", "")))
    return out

def main() -> None:
    args = parse_args()
    if args.num_dialogues <= 0:
        raise ValueError("--num-dialogues must be positive.")
    if args.num_turns < 2:
        raise ValueError("--num-turns must be >= 2.")
    if args.min_words_per_turn <= 0 or args.max_words_per_turn < args.min_words_per_turn:
        raise ValueError("Invalid word range. Ensure 0 < min <= max.")

    api_key = _normalize_api_key(
        args.api_key.strip()
        or os.getenv("OPENAI_API_KEY", "").strip()
        or os.getenv("OPENROUTER_API_KEY", "").strip()
    )
    rng = random.Random(args.seed)

    # Resolve topics and profiles based on experiment type.
    if args.topics.strip():
        topics = [t.strip() for t in args.topics.split(",") if t.strip()]
    elif args.experiment == "agreement_pivot":
        topics = list(DEFAULT_TOPICS)
    elif args.experiment == "similar_neutral":
        topics = list(NEUTRAL_TOPICS)
    elif args.experiment == "similar_polarize":
        topics = list(POLARIZE_SAME_TOPICS)
    elif args.experiment in ("distinct_names", "name_collision", "quote_intrusion", "cue_corrupted"):
        topics = list(INDIVIDUATION_TOPICS)
    else:
        topics = list(DEFAULT_TOPICS)

    needs_profiles = args.experiment in (
        "similar_neutral", "similar_polarize",
        "distinct_names", "name_collision", "quote_intrusion", "cue_corrupted",
    )
    if args.experiment in ("distinct_names", "name_collision", "quote_intrusion", "cue_corrupted"):
        profiles = list(INDIVIDUATION_PROFILES)
    else:
        profiles = list(SIMILAR_PROFILES)

    dialogues = _load_existing_dialogues(args.output, args.experiment)
    if dialogues:
        print(f"Resuming from existing output: {len(dialogues)} dialogues already present.")
    if len(dialogues) >= args.num_dialogues:
        payload = _build_payload(
            args=args,
            topics=topics,
            profiles=profiles,
            needs_profiles=needs_profiles,
            dialogues=dialogues[: args.num_dialogues],
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=True)
        print(f"Output already has {args.num_dialogues} dialogues: {args.output}")
        return

    start_idx = len(dialogues)
    for idx in range(start_idx, args.num_dialogues):
        topic = topics[idx % len(topics)]
        if len(topics) == 1:
            topic = f"{topic} (dialogue perspective {idx + 1})"

        profile = profiles[idx % len(profiles)] if needs_profiles else {}

        collision_name = ""
        if args.experiment == "name_collision":
            collision_name = profile.get("collision_name", "Michael")
            alice_profile = profile.get("alice_bio", "").format(name=collision_name)
            bob_profile = profile.get("bob_bio", "").format(name=collision_name)
        elif args.experiment in ("distinct_names", "quote_intrusion", "cue_corrupted"):
            alice_profile = profile.get("alice_bio", "").format(name="Alice")
            bob_profile = profile.get("bob_bio", "").format(name="Bob")
        else:
            alice_profile = profile.get("alice", "")
            bob_profile = profile.get("bob", "")

        turns = _generate_dialogue_chunked(
            experiment=args.experiment,
            api_base=args.api_base,
            api_key=api_key,
            model=args.model,
            topic=topic,
            num_turns=args.num_turns,
            min_words_per_turn=args.min_words_per_turn,
            max_words_per_turn=args.max_words_per_turn,
            temperature=max(0.0, min(2.0, args.temperature + rng.uniform(-0.1, 0.1))),
            max_completion_tokens=args.max_completion_tokens,
            max_retries=args.max_retries,
            chunk_turns=args.chunk_turns,
            min_words_tolerance=args.min_words_tolerance,
            agree_start=args.agreement_turn_start,
            agree_end=args.agreement_turn_end,
            alice_profile=alice_profile,
            bob_profile=bob_profile,
            quote_frequency=args.quote_frequency,
            collision_name=collision_name,
        )

        transcript_id = f"{args.experiment}_{idx:03d}"
        dialogue_entry: dict[str, object] = {
            "transcript_id": transcript_id,
            "topic": topic,
            "experiment": args.experiment,
            "base": turns,
            "speaker_swapped": speaker_swap(turns),
        }
        if needs_profiles:
            dialogue_entry["profiles"] = {
                "alice": alice_profile,
                "bob": bob_profile,
            }
        if args.experiment == "agreement_pivot":
            dialogue_entry["agreement_window"] = {
                "start": args.agreement_turn_start,
                "end": args.agreement_turn_end,
            }
        if args.experiment in ("distinct_names", "name_collision", "quote_intrusion", "cue_corrupted"):
            dialogue_entry["condition_family"] = "individuation"
            dialogue_entry["perturbation"] = (
                "none" if args.experiment == "distinct_names" else args.experiment
            )
        if args.experiment == "name_collision":
            dialogue_entry["collision_name"] = collision_name
        if args.experiment == "quote_intrusion":
            dialogue_entry["quote_frequency"] = args.quote_frequency

        dialogues.append(dialogue_entry)
        print(f"[{idx + 1}/{args.num_dialogues}] generated {transcript_id}")
        # Checkpoint every completed dialogue to survive transient API/network failures.
        payload = _build_payload(
            args=args,
            topics=topics,
            profiles=profiles,
            needs_profiles=needs_profiles,
            dialogues=dialogues,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=True)

    payload = _build_payload(
        args=args,
        topics=topics,
        profiles=profiles,
        needs_profiles=needs_profiles,
        dialogues=dialogues,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=True)
    print(f"Wrote {len(dialogues)} dialogues to {args.output}")


if __name__ == "__main__":
    main()
