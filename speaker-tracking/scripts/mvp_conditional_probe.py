#!/usr/bin/env python3
"""Per-transcript linear probes with 2/3 vs 1/3 splits and condition-aware splits.

Strategies
----------
- random_2_3                  : random 2/3 train, 1/3 test within each transcript (over unique turn_ids).
- first_vs_second_half        : train on first half of turns, test on last half.
- pivot_window                : (agreement_pivot) train on turns OUTSIDE the agreement_window, test INSIDE.
- pivot_first_vs_second_half  : (agreement_pivot) train on first half of window, test on second half of window.
- quote_noquote_vs_quote      : (quote_intrusion) train on non-quote turns, test on quote turns (heuristic).

Inputs
------
--embeddings : mvp_extract_turn_embeddings.py output JSON.
--dialogues  : matching exp_*_dialogues.json (needed for condition metadata).

Outputs a JSON with aggregate stats across transcripts + per-transcript metrics.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import sys
from collections import Counter
from pathlib import Path

# Share helpers with the main probe script.
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))
from mvp_linear_probe import (  # type: ignore  # noqa: E402
    _fit_and_eval,
    _label_for_task,
    _make_control_labels,
    _metric_diff,
    _summarize,
)

ALLOWED_STRATEGIES = (
    "random_2_3",
    "first_vs_second_half",
    "pivot_window",
    "pivot_first_vs_second_half",
    "quote_noquote_vs_quote",
)

# Heuristic: a single-quoted span of ≥10 chars with at least one space is almost always
# an intruded quote in this dataset (contractions like 'don't' are shorter / no space).
_QUOTE_INTRUSION_RE = re.compile(r"'[^'\n]{10,}'")


def has_quote_intrusion(text: str) -> bool:
    m = _QUOTE_INTRUSION_RE.search(text)
    if m is None:
        return False
    # Require at least one whitespace inside the quoted span.
    return " " in m.group(0)[1:-1]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--embeddings", type=Path, required=True)
    p.add_argument("--dialogues", type=Path, required=True,
                   help="Matching exp_*_dialogues.json for condition metadata.")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--strategy", type=str, required=True,
                   choices=ALLOWED_STRATEGIES)
    p.add_argument("--tasks", type=str, default="role",
                   help="Comma-separated: role,variant,topic")
    p.add_argument("--test-size", type=float, default=1.0 / 3.0,
                   help="For random_2_3; fraction of unique turn_ids held out.")
    p.add_argument("--num-seeds", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-iter", type=int, default=20000)
    p.add_argument(
        "--variants",
        type=str,
        default="base",
        choices=("base", "speaker_swapped", "both"),
        help=(
            "Which variants to include. Default 'base' only. base and speaker_swapped share "
            "identical token sequences but opposite speaker labels given include_speaker_prefix=False, "
            "so mixing them caps role accuracy at 50%%."
        ),
    )
    return p.parse_args()


def _load_embeddings(path: Path) -> tuple[list[dict[str, object]], dict[str, object]]:
    with path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    rows = payload.get("turn_embeddings", [])
    if not rows:
        raise ValueError(f"No embeddings in {path}")
    return rows, payload.get("metadata", {})


def _load_dialogue_meta(path: Path) -> dict[str, dict[str, object]]:
    """Return {transcript_id -> dialogue_entry}."""
    with path.open("r", encoding="utf-8") as fh:
        dl = json.load(fh)["dialogues"]
    return {str(d["transcript_id"]): d for d in dl}


def _parse_tasks(raw: str) -> list[str]:
    allowed = {"role", "variant", "topic"}
    tasks = [x.strip().lower() for x in raw.split(",") if x.strip()]
    bad = [t for t in tasks if t not in allowed]
    if bad:
        raise ValueError(f"Unknown tasks: {bad}")
    return tasks


def _turn_ids_in_transcript(t_rows: list[dict[str, object]]) -> list[int]:
    return sorted({int(r.get("turn_id", -1)) for r in t_rows if int(r.get("turn_id", -1)) >= 0})


def _split_random_2_3(
    turn_ids: list[int], test_size: float, seed: int,
) -> tuple[set[int], set[int]]:
    rng = random.Random(seed)
    ts = list(turn_ids)
    rng.shuffle(ts)
    n_test = max(1, int(math.ceil(len(ts) * test_size)))
    test = set(ts[:n_test])
    train = set(ts[n_test:])
    if not train:
        last = ts[-1]
        test.discard(last)
        train.add(last)
    return train, test


def _split_first_vs_second_half(turn_ids: list[int]) -> tuple[set[int], set[int]]:
    n = len(turn_ids)
    if n < 2:
        return set(), set()
    mid = n // 2
    return set(turn_ids[:mid]), set(turn_ids[mid:])


def _split_pivot_window(
    turn_ids: list[int], window: dict[str, int],
) -> tuple[set[int], set[int]]:
    start, end = int(window["start"]), int(window["end"])
    in_window = {t for t in turn_ids if start <= t <= end}
    out_window = {t for t in turn_ids if t not in in_window}
    return out_window, in_window  # train out, test in


def _split_pivot_window_halves(
    turn_ids: list[int], window: dict[str, int],
) -> tuple[set[int], set[int]]:
    start, end = int(window["start"]), int(window["end"])
    window_turns = sorted(t for t in turn_ids if start <= t <= end)
    if len(window_turns) < 2:
        return set(), set()
    mid = len(window_turns) // 2
    return set(window_turns[:mid]), set(window_turns[mid:])


def _split_quote(
    t_rows: list[dict[str, object]],
) -> tuple[set[int], set[int]]:
    """Return (non_quote_turn_ids, quote_turn_ids)."""
    per_turn: dict[int, bool] = {}
    for r in t_rows:
        tid = int(r.get("turn_id", -1))
        if tid < 0:
            continue
        # A turn has a "quote" if its own text contains an intruded quoted span.
        txt = str(r.get("text", ""))
        per_turn[tid] = per_turn.get(tid, False) or has_quote_intrusion(txt)
    quote_tids = {t for t, q in per_turn.items() if q}
    no_quote_tids = {t for t, q in per_turn.items() if not q}
    return no_quote_tids, quote_tids


def _run_probe(
    t_rows: list[dict[str, object]],
    train_turns: set[int],
    test_turns: set[int],
    tasks: list[str],
    seed: int,
    max_iter: int,
) -> dict[str, dict[str, object]] | None:
    if not train_turns or not test_turns:
        return None

    train_idx = [i for i, r in enumerate(t_rows) if int(r.get("turn_id", -1)) in train_turns]
    test_idx = [i for i, r in enumerate(t_rows) if int(r.get("turn_id", -1)) in test_turns]
    if not train_idx or not test_idx:
        return None

    out: dict[str, dict[str, object]] = {}
    for task in tasks:
        y_train = [_label_for_task(t_rows[i], task) for i in train_idx]
        y_test = [_label_for_task(t_rows[i], task) for i in test_idx]
        X_train = [t_rows[i]["vector"] for i in train_idx]
        X_test = [t_rows[i]["vector"] for i in test_idx]
        if len(set(y_train)) < 2 or len(set(y_test)) < 2:
            out[task] = {"error": "fewer than 2 classes in train or test",
                        "label_counts_train": dict(Counter(y_train)),
                        "label_counts_test": dict(Counter(y_test))}
            continue
        metrics = _fit_and_eval(X_train, y_train, X_test, y_test, max_iter=max_iter, task=task)
        ctrl = _make_control_labels(y_train=y_train, y_test=y_test, seed=seed + 1337)
        ctrl_metrics = None
        sel = None
        if ctrl is not None:
            cy_train, cy_test = ctrl
            ctrl_metrics = _fit_and_eval(X_train, cy_train, X_test, cy_test, max_iter=max_iter, task=task)
            sel = {
                probe: {m: _metric_diff(metrics[probe].get(m), ctrl_metrics[probe].get(m))
                        for m in ("accuracy", "balanced_accuracy", "roc_auc")}
                for probe in metrics
            }
        out[task] = {
            "num_train": len(train_idx),
            "num_test": len(test_idx),
            "num_train_turns": len(train_turns),
            "num_test_turns": len(test_turns),
            "metrics_by_probe": metrics,
            "control_metrics_by_probe": ctrl_metrics,
            "selectivity_by_probe": sel,
        }
    return out


def _aggregate_across_transcripts(
    per_transcript: dict[str, dict[str, object]], tasks: list[str],
) -> dict[str, dict[str, dict[str, dict[str, float | None]]]]:
    """Aggregate per-transcript (and per-seed, for random_2_3) metrics into mean/std tables."""
    agg: dict[str, dict[str, dict[str, dict[str, float | None]]]] = {}
    for task in tasks:
        probe_names: set[str] = set()
        bag: dict[str, dict[str, list[float | None]]] = {}
        ctrl_bag: dict[str, dict[str, list[float | None]]] = {}
        sel_bag: dict[str, dict[str, list[float | None]]] = {}
        for tid, tblob in per_transcript.items():
            res = tblob.get(task)
            if not isinstance(res, dict) or "metrics_by_probe" not in res:
                continue
            metrics = res["metrics_by_probe"]
            ctrl = res.get("control_metrics_by_probe") or {}
            sel = res.get("selectivity_by_probe") or {}
            for probe, mvals in metrics.items():
                probe_names.add(probe)
                for m, v in mvals.items():
                    bag.setdefault(probe, {}).setdefault(m, []).append(v)
                for m in ("accuracy", "balanced_accuracy", "roc_auc"):
                    ctrl_bag.setdefault(probe, {}).setdefault(m, []).append((ctrl.get(probe) or {}).get(m))
                    sel_bag.setdefault(probe, {}).setdefault(m, []).append((sel.get(probe) or {}).get(m))
        out: dict[str, dict[str, dict[str, float | None]]] = {}
        out_ctrl: dict[str, dict[str, dict[str, float | None]]] = {}
        out_sel: dict[str, dict[str, dict[str, float | None]]] = {}
        for probe in sorted(probe_names):
            out[probe] = {m: _summarize(bag[probe][m]) for m in bag[probe]}
            out_ctrl[probe] = {m: _summarize(ctrl_bag[probe][m]) for m in ctrl_bag[probe]}
            out_sel[probe] = {m: _summarize(sel_bag[probe][m]) for m in sel_bag[probe]}
        agg[task] = {"aggregate": out, "aggregate_control": out_ctrl, "aggregate_selectivity": out_sel}
    return agg


def main() -> None:
    args = parse_args()
    tasks = _parse_tasks(args.tasks)

    rows, emb_meta = _load_embeddings(args.embeddings)
    if args.variants != "both":
        before = len(rows)
        rows = [r for r in rows if str(r.get("variant", "")) == args.variants]
        print(f"Filtered to variant='{args.variants}': {before} -> {len(rows)} rows")
        if not rows:
            raise ValueError(f"No rows left after filtering to variant='{args.variants}'.")
    dialogue_meta = _load_dialogue_meta(args.dialogues)

    model_ids = sorted({str(r.get("model_id", "")) for r in rows if r.get("model_id")})
    if not model_ids:
        raise ValueError("No model_id in embeddings.")

    out_payload: dict[str, object] = {
        "metadata": {
            "script": "mvp_conditional_probe.py",
            "strategy": args.strategy,
            "tasks": tasks,
            "test_size": args.test_size,
            "num_seeds": args.num_seeds,
            "seed": args.seed,
            "variants": args.variants,
            "embeddings_meta": emb_meta,
        },
        "results_by_model": {},
    }

    for model_id in model_ids:
        mrows = [r for r in rows if str(r.get("model_id", "")) == model_id]
        by_transcript: dict[str, list[dict[str, object]]] = {}
        for r in mrows:
            by_transcript.setdefault(str(r.get("transcript_id", "")), []).append(r)

        per_transcript_out: dict[str, dict[str, object]] = {}
        skipped: list[dict[str, str]] = []
        for tid, t_rows in sorted(by_transcript.items()):
            turn_ids = _turn_ids_in_transcript(t_rows)
            if len(turn_ids) < 2:
                skipped.append({"transcript_id": tid, "reason": "too few turns"})
                continue

            if args.strategy == "random_2_3":
                seed_blobs: list[dict[str, object]] = []
                for s_off in range(args.num_seeds):
                    split_seed = args.seed + s_off
                    train, test = _split_random_2_3(turn_ids, args.test_size, split_seed)
                    rec = _run_probe(t_rows, train, test, tasks, split_seed, args.max_iter)
                    if rec is not None:
                        rec["_seed"] = split_seed
                        seed_blobs.append(rec)
                if seed_blobs:
                    # For each task, average over seeds within this transcript.
                    t_blob: dict[str, object] = {}
                    for task in tasks:
                        metric_seeds = [
                            sb[task] for sb in seed_blobs
                            if isinstance(sb.get(task), dict) and "metrics_by_probe" in sb[task]
                        ]
                        if not metric_seeds:
                            continue
                        probe_names = sorted(metric_seeds[0]["metrics_by_probe"].keys())
                        avg_metrics: dict[str, dict[str, float | None]] = {}
                        avg_ctrl: dict[str, dict[str, float | None]] = {}
                        avg_sel: dict[str, dict[str, float | None]] = {}
                        for probe in probe_names:
                            avg_metrics[probe] = {}
                            avg_ctrl[probe] = {}
                            avg_sel[probe] = {}
                            for m in ("accuracy", "balanced_accuracy", "roc_auc"):
                                vals = [mb["metrics_by_probe"][probe][m] for mb in metric_seeds]
                                cvals = [(mb.get("control_metrics_by_probe") or {}).get(probe, {}).get(m)
                                         for mb in metric_seeds]
                                svals = [(mb.get("selectivity_by_probe") or {}).get(probe, {}).get(m)
                                         for mb in metric_seeds]
                                avg_metrics[probe][m] = _summarize(vals)["mean"]
                                avg_ctrl[probe][m] = _summarize(cvals)["mean"]
                                avg_sel[probe][m] = _summarize(svals)["mean"]
                        t_blob[task] = {
                            "metrics_by_probe": avg_metrics,
                            "control_metrics_by_probe": avg_ctrl,
                            "selectivity_by_probe": avg_sel,
                            "per_seed": metric_seeds,
                            "num_train_turns_mean": sum(mb["num_train_turns"] for mb in metric_seeds) / len(metric_seeds),
                            "num_test_turns_mean": sum(mb["num_test_turns"] for mb in metric_seeds) / len(metric_seeds),
                        }
                    if t_blob:
                        per_transcript_out[tid] = t_blob
                continue

            # Deterministic (non-random) strategies: single split per transcript.
            train_turns: set[int] = set()
            test_turns: set[int] = set()
            reason = ""
            if args.strategy == "first_vs_second_half":
                train_turns, test_turns = _split_first_vs_second_half(turn_ids)
            elif args.strategy == "pivot_window":
                win = dialogue_meta.get(tid, {}).get("agreement_window")
                if not isinstance(win, dict):
                    reason = "no agreement_window"
                else:
                    train_turns, test_turns = _split_pivot_window(turn_ids, win)
            elif args.strategy == "pivot_first_vs_second_half":
                win = dialogue_meta.get(tid, {}).get("agreement_window")
                if not isinstance(win, dict):
                    reason = "no agreement_window"
                else:
                    train_turns, test_turns = _split_pivot_window_halves(turn_ids, win)
            elif args.strategy == "quote_noquote_vs_quote":
                train_turns, test_turns = _split_quote(t_rows)
                if not test_turns:
                    reason = "no detected quote turns"
            else:
                raise ValueError(f"Unhandled strategy: {args.strategy}")

            if reason or not train_turns or not test_turns:
                skipped.append({"transcript_id": tid,
                                "reason": reason or "empty train or test split"})
                continue

            rec = _run_probe(t_rows, train_turns, test_turns, tasks,
                             seed=args.seed, max_iter=args.max_iter)
            if rec is not None:
                per_transcript_out[tid] = rec

        aggregate = _aggregate_across_transcripts(per_transcript_out, tasks)
        out_payload["results_by_model"][model_id] = {
            "per_transcript": per_transcript_out,
            "aggregate": aggregate,
            "num_transcripts": len(per_transcript_out),
            "num_skipped": len(skipped),
            "skipped": skipped,
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as fh:
        json.dump(out_payload, fh, indent=2, ensure_ascii=True)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
