#!/usr/bin/env python3
"""Evaluate role-direction/function-vector style signals with controls."""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--split-mode",
        type=str,
        choices=("transcript", "topic"),
        default="transcript",
        help="Group-aware split key used for held-out evaluation.",
    )
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--num-seeds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--num-random-directions",
        type=int,
        default=20,
        help="Number of random directions for baseline comparison.",
    )
    return parser.parse_args()


def _load_rows(path: Path) -> list[dict[str, object]]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    rows = payload.get("turn_embeddings", [])
    if not isinstance(rows, list) or not rows:
        raise ValueError("No rows found under 'turn_embeddings'.")
    return rows


def _mean(vectors: list[list[float]]) -> list[float]:
    if not vectors:
        return []
    d = len(vectors[0])
    out = [0.0] * d
    for vec in vectors:
        for i, x in enumerate(vec):
            out[i] += x
    n = float(len(vectors))
    return [x / n for x in out]


def _sub(a: list[float], b: list[float]) -> list[float]:
    return [x - y for x, y in zip(a, b)]


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def _norm(a: list[float]) -> float:
    return math.sqrt(sum(x * x for x in a))


def _normalize(a: list[float], eps: float = 1e-8) -> list[float]:
    n = max(_norm(a), eps)
    return [x / n for x in a]


def _cos(a: list[float], b: list[float], eps: float = 1e-8) -> float:
    return _dot(a, b) / max(_norm(a) * _norm(b), eps)


def _project(a: list[float], direction: list[float]) -> float:
    return _dot(a, direction)


def _group_key(row: dict[str, object], split_mode: str) -> str:
    if split_mode == "transcript":
        return str(row.get("transcript_id", ""))
    if split_mode == "topic":
        return str(row.get("topic", ""))
    raise ValueError(f"Unsupported split mode '{split_mode}'.")


def _split_groups(
    groups: list[str], test_size: float, seed: int
) -> tuple[set[str], set[str]]:
    rng = random.Random(seed)
    uniq = sorted(set(groups))
    if len(uniq) < 2:
        raise ValueError("Need at least 2 groups.")
    rng.shuffle(uniq)
    test_n = max(1, int(math.ceil(len(uniq) * test_size)))
    test = set(uniq[:test_n])
    train = set(uniq[test_n:])
    if not train:
        last = uniq[-1]
        test.remove(last)
        train.add(last)
    return train, test


def _role_accuracy(rows: list[dict[str, object]], direction: list[float]) -> float | None:
    if not rows:
        return None
    correct = 0
    total = 0
    for row in rows:
        speaker = str(row.get("speaker", ""))
        if speaker not in {"Alice", "Bob"}:
            continue
        sign = 1.0 if speaker == "Alice" else -1.0
        pred = _project(row["vector"], direction)
        if sign * pred > 0:
            correct += 1
        total += 1
    if total == 0:
        return None
    return float(correct) / float(total)


def _build_direction_from_rows(rows: list[dict[str, object]]) -> list[float] | None:
    alice = [row["vector"] for row in rows if row.get("speaker") == "Alice"]
    bob = [row["vector"] for row in rows if row.get("speaker") == "Bob"]
    if not alice or not bob:
        return None
    return _normalize(_sub(_mean(alice), _mean(bob)))


def _swap_flip_score(rows: list[dict[str, object]], direction: list[float]) -> float | None:
    keyed: dict[tuple[str, int], dict[str, dict[str, object]]] = defaultdict(dict)
    for row in rows:
        transcript_id = str(row.get("transcript_id", ""))
        turn_id = int(row.get("turn_id", -1))
        variant = str(row.get("variant", ""))
        keyed[(transcript_id, turn_id)][variant] = row

    flips = []
    for pair in keyed.values():
        base = pair.get("base")
        swapped = pair.get("speaker_swapped")
        if not base or not swapped:
            continue
        s1 = _project(base["vector"], direction)
        s2 = _project(swapped["vector"], direction)
        flips.append(1.0 if s1 * s2 < 0 else 0.0)
    if not flips:
        return None
    return sum(flips) / float(len(flips))


def main() -> None:
    args = parse_args()
    if args.num_seeds <= 0:
        raise ValueError("--num-seeds must be positive.")
    if args.num_random_directions <= 0:
        raise ValueError("--num-random-directions must be positive.")
    if not (0.0 < args.test_size < 1.0):
        raise ValueError("--test-size must be in (0,1).")

    rows = _load_rows(args.embeddings)
    model_ids = sorted({str(row.get("model_id", "")) for row in rows if row.get("model_id")})
    if not model_ids:
        raise ValueError("No model_id values in embeddings.")

    payload: dict[str, object] = {
        "metadata": {
            "script": "mvp_role_direction_eval.py",
            "split_mode": args.split_mode,
            "test_size": args.test_size,
            "num_seeds": args.num_seeds,
            "seed": args.seed,
            "num_random_directions": args.num_random_directions,
        },
        "results_by_model": {},
    }

    for model_id in model_ids:
        model_rows = [row for row in rows if str(row.get("model_id", "")) == model_id]
        groups = [_group_key(row, args.split_mode) for row in model_rows]
        if len(set(groups)) < 2:
            payload["results_by_model"][model_id] = {"error": "Not enough split groups."}
            continue

        seed_metrics = []
        for seed_offset in range(args.num_seeds):
            split_seed = args.seed + seed_offset
            train_groups, test_groups = _split_groups(groups, args.test_size, split_seed)
            train_rows = [row for row in model_rows if _group_key(row, args.split_mode) in train_groups]
            test_rows = [row for row in model_rows if _group_key(row, args.split_mode) in test_groups]

            direction = _build_direction_from_rows(train_rows)
            if direction is None:
                continue

            acc = _role_accuracy(test_rows, direction)
            flip = _swap_flip_score(test_rows, direction)

            # Label-shuffle control computed on same test rows.
            rng = random.Random(split_seed)
            shuffled_rows = [dict(row) for row in test_rows]
            labels = [row.get("speaker", "") for row in shuffled_rows]
            rng.shuffle(labels)
            for row, shuffled_label in zip(shuffled_rows, labels):
                row["speaker"] = shuffled_label
            shuffled_acc = _role_accuracy(shuffled_rows, direction)

            # Random-direction baseline (same dimensionality).
            dim = len(direction)
            random_accs = []
            for j in range(args.num_random_directions):
                local_rng = random.Random(split_seed * 1000 + j)
                rand = [local_rng.uniform(-1.0, 1.0) for _ in range(dim)]
                rand = _normalize(rand)
                val = _role_accuracy(test_rows, rand)
                if val is not None:
                    random_accs.append(val)
            random_acc = (
                sum(random_accs) / float(len(random_accs))
                if random_accs
                else None
            )

            seed_metrics.append(
                {
                    "seed": split_seed,
                    "num_train_groups": len(train_groups),
                    "num_test_groups": len(test_groups),
                    "num_train_rows": len(train_rows),
                    "num_test_rows": len(test_rows),
                    "heldout_role_accuracy": acc,
                    "heldout_swap_flip_rate": flip,
                    "label_shuffle_accuracy": shuffled_acc,
                    "random_direction_accuracy": random_acc,
                }
            )

        if not seed_metrics:
            payload["results_by_model"][model_id] = {
                "error": "Could not produce valid splits for evaluation."
            }
            continue

        def _agg(name: str) -> dict[str, float | None]:
            vals = [row[name] for row in seed_metrics if row[name] is not None]
            if not vals:
                return {"mean": None, "std": None}
            mean = sum(vals) / float(len(vals))
            var = sum((x - mean) ** 2 for x in vals) / float(len(vals))
            return {"mean": mean, "std": math.sqrt(var)}

        all_rows_direction = _build_direction_from_rows(model_rows)
        pairwise_transcript_cos = []
        if all_rows_direction is not None:
            per_transcript: dict[str, list[dict[str, object]]] = defaultdict(list)
            for row in model_rows:
                per_transcript[str(row.get("transcript_id", ""))].append(row)
            dirs = []
            for t_rows in per_transcript.values():
                d = _build_direction_from_rows(t_rows)
                if d is not None:
                    dirs.append(d)
            for i in range(len(dirs)):
                for j in range(i + 1, len(dirs)):
                    pairwise_transcript_cos.append(_cos(dirs[i], dirs[j]))

        payload["results_by_model"][model_id] = {
            "aggregate": {
                "heldout_role_accuracy": _agg("heldout_role_accuracy"),
                "heldout_swap_flip_rate": _agg("heldout_swap_flip_rate"),
                "label_shuffle_accuracy": _agg("label_shuffle_accuracy"),
                "random_direction_accuracy": _agg("random_direction_accuracy"),
                "mean_pairwise_transcript_direction_cosine": (
                    sum(pairwise_transcript_cos) / float(len(pairwise_transcript_cos))
                    if pairwise_transcript_cos
                    else None
                ),
            },
            "per_seed": seed_metrics,
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=True)


if __name__ == "__main__":
    main()
