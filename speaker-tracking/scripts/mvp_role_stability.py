#!/usr/bin/env python3
"""Compute MVP role-direction stability metrics."""

from __future__ import annotations

import argparse
import json
import math
from itertools import combinations
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _mean(vectors: list[list[float]]) -> list[float]:
    if not vectors:
        return []
    d = len(vectors[0])
    acc = [0.0] * d
    for vec in vectors:
        for i, x in enumerate(vec):
            acc[i] += x
    n = float(len(vectors))
    return [x / n for x in acc]


def _sub(a: list[float], b: list[float]) -> list[float]:
    return [x - y for x, y in zip(a, b)]


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def _norm(a: list[float]) -> float:
    return math.sqrt(sum(x * x for x in a))


def _cos(a: list[float], b: list[float], eps: float = 1e-8) -> float:
    return _dot(a, b) / max(_norm(a) * _norm(b), eps)


def main() -> None:
    args = parse_args()
    with args.embeddings.open("r", encoding="utf-8") as handle:
        rows = json.load(handle)["turn_embeddings"]

    per_model: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        per_model.setdefault(row["model_id"], []).append(row)

    results_by_model: dict[str, dict[str, object]] = {}
    for model_id, model_rows in per_model.items():
        per_transcript: dict[str, dict[str, dict[str, list[list[float]]]]] = {}
        for row in model_rows:
            t_id = row["transcript_id"]
            variant = row["variant"]
            speaker = row["speaker"]
            per_transcript.setdefault(t_id, {}).setdefault(variant, {}).setdefault(
                speaker, []
            ).append(row["vector"])

        base_role_vectors: dict[str, list[float]] = {}
        swap_role_vectors: dict[str, list[float]] = {}
        role_norms: dict[str, float] = {}
        for t_id, by_variant in per_transcript.items():
            base = by_variant.get("base", {})
            if "Alice" in base and "Bob" in base:
                v = _sub(_mean(base["Alice"]), _mean(base["Bob"]))
                base_role_vectors[t_id] = v
                role_norms[t_id] = _norm(v)
            swapped = by_variant.get("speaker_swapped", {})
            if "Alice" in swapped and "Bob" in swapped:
                swap_role_vectors[t_id] = _sub(
                    _mean(swapped["Alice"]), _mean(swapped["Bob"])
                )

        pairwise = [
            _cos(v1, v2)
            for (_, v1), (_, v2) in combinations(base_role_vectors.items(), 2)
        ]
        mean_pairwise_cosine = sum(pairwise) / float(len(pairwise)) if pairwise else 0.0

        flip_scores = []
        for t_id, v_base in base_role_vectors.items():
            if t_id in swap_role_vectors:
                flip_scores.append(_cos(v_base, swap_role_vectors[t_id]))
        mean_swap_cosine = sum(flip_scores) / float(len(flip_scores)) if flip_scores else 0.0

        results_by_model[model_id] = {
            "metrics": {
                "num_transcripts": len(base_role_vectors),
                "mean_pairwise_role_cosine": mean_pairwise_cosine,
                "mean_swap_cosine_should_be_negative": mean_swap_cosine,
            },
            "per_transcript_role_norm": role_norms,
        }

    payload = {
        "results_by_model": results_by_model,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=True)


if __name__ == "__main__":
    main()
