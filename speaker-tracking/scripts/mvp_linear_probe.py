#!/usr/bin/env python3
"""Train linear probes on turn embeddings with leakage-safe splits."""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--tasks",
        type=str,
        default="role",
        help="Comma-separated tasks: role,variant,topic",
    )
    parser.add_argument(
        "--split-mode",
        type=str,
        choices=("transcript", "topic"),
        default="transcript",
        help="Group-aware split key to reduce leakage.",
    )
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--num-seeds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--max-iter",
        type=int,
        default=20000,
        help="Max iterations for linear solvers.",
    )
    parser.add_argument(
        "--per-transcript",
        action="store_true",
        help=(
            "If set, also run probes per transcript_id (train/test split within each transcript). "
            "Splitting is done by turn_id groups so base/swapped versions of the same turn do not "
            "leak across splits."
        ),
    )
    parser.add_argument(
        "--variants",
        type=str,
        default="base",
        choices=("base", "speaker_swapped", "both"),
        help=(
            "Which variants to include. Default 'base' avoids the pathological case where base "
            "and speaker_swapped share identical token sequences but opposite speaker labels "
            "(which caps accuracy at 50%%). Use 'both' only if you know what you're doing."
        ),
    )
    parser.add_argument(
        "--save-turn-preds",
        action="store_true",
        help=(
            "If set, include per-test-turn predictions under each seed record "
            "(turn_id/transcript_id metadata, true/pred labels, correctness, and "
            "logreg probabilities where available)."
        ),
    )
    return parser.parse_args()


def _load_rows(path: Path) -> list[dict[str, object]]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    rows = payload.get("turn_embeddings", [])
    if not isinstance(rows, list) or not rows:
        raise ValueError("No rows found under 'turn_embeddings'.")
    return rows


def _parse_tasks(raw: str) -> list[str]:
    allowed = {"role", "variant", "topic"}
    tasks = [x.strip().lower() for x in raw.split(",") if x.strip()]
    if not tasks:
        raise ValueError("Pass at least one task via --tasks.")
    unknown = [x for x in tasks if x not in allowed]
    if unknown:
        raise ValueError(f"Unknown tasks: {unknown}. Allowed: {sorted(allowed)}")
    return tasks


def _label_for_task(row: dict[str, object], task: str) -> str:
    if task == "role":
        return str(row.get("speaker", ""))
    if task == "variant":
        return str(row.get("variant", ""))
    if task == "topic":
        return str(row.get("topic", ""))
    raise ValueError(f"Unsupported task '{task}'.")


def _group_for_split(row: dict[str, object], split_mode: str) -> str:
    if split_mode == "transcript":
        return str(row.get("transcript_id", ""))
    if split_mode == "topic":
        return str(row.get("topic", ""))
    raise ValueError(f"Unsupported split mode '{split_mode}'.")


def _subset_rows(rows: list[dict[str, object]], model_id: str) -> list[dict[str, object]]:
    return [row for row in rows if str(row.get("model_id", "")) == model_id]


def _split_turn_ids(
    turn_ids: list[int], test_size: float, seed: int
) -> tuple[set[int], set[int]]:
    """Split unique turn_id values into train/test."""
    train, test = _split_groups([str(t) for t in turn_ids], test_size=test_size, seed=seed)
    return {int(x) for x in train}, {int(x) for x in test}


def _split_groups(
    groups: list[str], test_size: float, seed: int
) -> tuple[set[str], set[str]]:
    rng = random.Random(seed)
    unique = sorted(set(groups))
    if len(unique) < 2:
        raise ValueError("Need at least 2 split groups.")
    rng.shuffle(unique)
    test_n = max(1, int(math.ceil(len(unique) * test_size)))
    test = set(unique[:test_n])
    train = set(unique[test_n:])
    if not train:
        # Keep at least one train group.
        last = unique[-1]
        test.remove(last)
        train.add(last)
    return train, test


def _compute_binary_auc(y_true: list[int], scores: list[float]) -> float | None:
    try:
        from sklearn.metrics import roc_auc_score
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Install scikit-learn for probes.") from exc
    if len(set(y_true)) < 2:
        return None
    return float(roc_auc_score(y_true, scores))


def _summarize(values: list[float | None]) -> dict[str, float | None]:
    valid = [float(v) for v in values if v is not None]
    if not valid:
        return {"mean": None, "std": None}
    mean = sum(valid) / float(len(valid))
    var = sum((x - mean) ** 2 for x in valid) / float(len(valid))
    return {"mean": mean, "std": math.sqrt(var)}


def _metric_diff(a: float | None, b: float | None) -> float | None:
    if a is None or b is None:
        return None
    return float(a) - float(b)


def _make_control_labels(
    *,
    y_train: list[str],
    y_test: list[str],
    seed: int,
    max_tries: int = 25,
) -> tuple[list[str], list[str]] | None:
    """Create a Hewitt-style control task by randomizing labels.

    We:
    - shuffle training labels (preserves the *exact* training label multiset)
    - sample test labels i.i.d. from the training-label empirical distribution

    This keeps class balance comparable while breaking any relationship to X.
    """
    if len(set(y_train)) < 2:
        return None
    rng = random.Random(seed)
    control_y_train = list(y_train)
    rng.shuffle(control_y_train)

    pool = list(y_train)
    if not pool:
        return None

    control_y_test: list[str] = []
    for _ in range(max_tries):
        control_y_test = [rng.choice(pool) for _ in range(len(y_test))]
        if len(set(control_y_test)) >= 2:
            break
    if len(set(control_y_test)) < 2:
        return None
    return control_y_train, control_y_test


def _fit_and_eval(
    X_train,
    y_train,
    X_test,
    y_test,
    max_iter: int,
    task: str,
    probes: list[str] | None = None,
    collect_predictions: bool = False,
    test_rows: list[dict[str, object]] | None = None,
) -> tuple[dict[str, dict[str, float | None]], dict[str, list[dict[str, object]]] | None]:
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score, recall_score
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.svm import LinearSVC
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Install scikit-learn for probes.") from exc

    selected = set(probes) if probes else {"logreg_l2", "linear_svm"}
    results: dict[str, dict[str, float | None]] = {}
    prediction_rows: dict[str, list[dict[str, object]]] | None = {} if collect_predictions else None

    if "logreg_l2" not in selected and "linear_svm" not in selected:
        raise ValueError(f"probes must include at least one of logreg_l2/linear_svm; got {probes}")

    if "logreg_l2" not in selected:
        # Still need the class set for any downstream labels computation.
        pass

    logreg = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=max_iter,
            class_weight="balanced",
            solver="lbfgs",
        ),
    )
    logreg.fit(X_train, y_train)
    pred = logreg.predict(X_test)
    # Define "balanced accuracy" as macro-average recall over train classes.
    # This avoids sklearn warnings when y_test is missing some classes (common
    # for topic prediction under transcript-held-out splits).
    all_labels = sorted(set(y_train))
    model_out = {
        "accuracy": float(accuracy_score(y_test, pred)),
        "balanced_accuracy": float(
            recall_score(y_test, pred, labels=all_labels, average="macro", zero_division=0)
        ),
        "roc_auc": None,
    }

    # For binary tasks we report AUC from positive class probability.
    if len(set(y_train)) == 2 and len(set(y_test)) == 2:
        clf = logreg.named_steps["logisticregression"]
        scaler = logreg.named_steps["standardscaler"]
        Xs = scaler.transform(X_test)
        probs = clf.predict_proba(Xs)
        # Class index mapping from estimator classes_
        class_order = list(clf.classes_)
        pos_idx = 1 if len(class_order) == 2 else 0
        y_bin = [1 if y == class_order[pos_idx] else 0 for y in y_test]
        model_out["roc_auc"] = _compute_binary_auc(y_bin, probs[:, pos_idx].tolist())
    elif task == "topic":
        # Multiclass AUC for topic is optional and often unstable on small sets.
        model_out["roc_auc"] = None
    results["logreg_l2"] = model_out
    if collect_predictions and prediction_rows is not None:
        clf = logreg.named_steps["logisticregression"]
        scaler = logreg.named_steps["standardscaler"]
        Xs = scaler.transform(X_test)
        probs = clf.predict_proba(Xs)
        class_order = [str(c) for c in clf.classes_]
        class_to_idx = {c: i for i, c in enumerate(class_order)}
        rows_out: list[dict[str, object]] = []
        for i, (yt, yp) in enumerate(zip(y_test, pred)):
            true_label = str(yt)
            pred_label = str(yp)
            true_idx = class_to_idx.get(true_label)
            pred_idx = class_to_idx.get(pred_label)
            row_meta = test_rows[i] if test_rows is not None and i < len(test_rows) else {}
            rows_out.append(
                {
                    "turn_id": int(row_meta.get("turn_id", -1))
                    if str(row_meta.get("turn_id", "")).strip()
                    else None,
                    "transcript_id": str(row_meta.get("transcript_id", "")),
                    "speaker": str(row_meta.get("speaker", "")),
                    "variant": str(row_meta.get("variant", "")),
                    "topic": str(row_meta.get("topic", "")),
                    "true_label": true_label,
                    "pred_label": pred_label,
                    "correct": bool(pred_label == true_label),
                    "true_label_prob": float(probs[i, true_idx]) if true_idx is not None else None,
                    "pred_label_prob": float(probs[i, pred_idx]) if pred_idx is not None else None,
                }
            )
        prediction_rows["logreg_l2"] = rows_out

    linsvc = make_pipeline(
        StandardScaler(),
        LinearSVC(max_iter=max_iter, class_weight="balanced", dual="auto"),
    )
    linsvc.fit(X_train, y_train)
    pred = linsvc.predict(X_test)
    results["linear_svm"] = {
        "accuracy": float(accuracy_score(y_test, pred)),
        "balanced_accuracy": float(
            recall_score(y_test, pred, labels=all_labels, average="macro", zero_division=0)
        ),
        "roc_auc": None,
    }
    if collect_predictions and prediction_rows is not None:
        rows_out = []
        for i, (yt, yp) in enumerate(zip(y_test, pred)):
            true_label = str(yt)
            pred_label = str(yp)
            row_meta = test_rows[i] if test_rows is not None and i < len(test_rows) else {}
            rows_out.append(
                {
                    "turn_id": int(row_meta.get("turn_id", -1))
                    if str(row_meta.get("turn_id", "")).strip()
                    else None,
                    "transcript_id": str(row_meta.get("transcript_id", "")),
                    "speaker": str(row_meta.get("speaker", "")),
                    "variant": str(row_meta.get("variant", "")),
                    "topic": str(row_meta.get("topic", "")),
                    "true_label": true_label,
                    "pred_label": pred_label,
                    "correct": bool(pred_label == true_label),
                    "true_label_prob": None,
                    "pred_label_prob": None,
                }
            )
        prediction_rows["linear_svm"] = rows_out
    return results, prediction_rows


def main() -> None:
    args = parse_args()
    if args.num_seeds <= 0:
        raise ValueError("--num-seeds must be positive.")
    if not (0.0 < args.test_size < 1.0):
        raise ValueError("--test-size must be in (0,1).")

    rows = _load_rows(args.embeddings)
    if args.variants != "both":
        before = len(rows)
        rows = [r for r in rows if str(r.get("variant", "")) == args.variants]
        print(f"Filtered to variant='{args.variants}': {before} -> {len(rows)} rows")
        if not rows:
            raise ValueError(f"No rows left after filtering to variant='{args.variants}'.")
    tasks = _parse_tasks(args.tasks)
    model_ids = sorted({str(row.get("model_id", "")) for row in rows if row.get("model_id")})
    if not model_ids:
        raise ValueError("No model_id values in embeddings.")

    payload: dict[str, object] = {
        "metadata": {
            "script": "mvp_linear_probe.py",
            "tasks": tasks,
            "split_mode": args.split_mode,
            "test_size": args.test_size,
            "num_seeds": args.num_seeds,
            "seed": args.seed,
            "max_iter": args.max_iter,
            "variants": args.variants,
            "save_turn_preds": bool(args.save_turn_preds),
            "control_task": "shuffle_train_and_sample_test_from_train_dist",
        },
        "results_by_model": {},
    }

    for model_id in model_ids:
        model_rows = _subset_rows(rows, model_id)
        per_task: dict[str, object] = {}
        for task in tasks:
            labels = [_label_for_task(row, task) for row in model_rows]
            vectors = [row["vector"] for row in model_rows]
            groups = [_group_for_split(row, args.split_mode) for row in model_rows]

            label_counts = Counter(labels)
            if len(label_counts) < 2:
                per_task[task] = {
                    "error": f"Task '{task}' has fewer than 2 classes for model '{model_id}'.",
                    "label_counts": dict(label_counts),
                }
                continue

            per_seed: list[dict[str, object]] = []
            for seed_offset in range(args.num_seeds):
                split_seed = args.seed + seed_offset
                train_groups, test_groups = _split_groups(
                    groups=groups, test_size=args.test_size, seed=split_seed
                )
                train_idx = [i for i, g in enumerate(groups) if g in train_groups]
                test_idx = [i for i, g in enumerate(groups) if g in test_groups]
                if not train_idx or not test_idx:
                    continue

                X_train = [vectors[i] for i in train_idx]
                y_train = [labels[i] for i in train_idx]
                X_test = [vectors[i] for i in test_idx]
                y_test = [labels[i] for i in test_idx]
                test_rows = [model_rows[i] for i in test_idx]
                if len(set(y_train)) < 2 or len(set(y_test)) < 2:
                    continue

                metrics_by_model, turn_preds_by_probe = _fit_and_eval(
                    X_train=X_train,
                    y_train=y_train,
                    X_test=X_test,
                    y_test=y_test,
                    max_iter=args.max_iter,
                    task=task,
                    collect_predictions=bool(args.save_turn_preds),
                    test_rows=test_rows,
                )
                control = _make_control_labels(y_train=y_train, y_test=y_test, seed=split_seed + 1337)
                control_metrics_by_model = None
                selectivity_by_model = None
                if control is not None:
                    control_y_train, control_y_test = control
                    control_metrics_by_model, _ = _fit_and_eval(
                        X_train=X_train,
                        y_train=control_y_train,
                        X_test=X_test,
                        y_test=control_y_test,
                        max_iter=args.max_iter,
                        task=task,
                        collect_predictions=False,
                    )
                    selectivity_by_model = {}
                    for probe_name, true_metrics in metrics_by_model.items():
                        ctrl_metrics = control_metrics_by_model.get(probe_name, {})
                        selectivity_by_model[probe_name] = {
                            metric: _metric_diff(true_metrics.get(metric), ctrl_metrics.get(metric))
                            for metric in ("accuracy", "balanced_accuracy", "roc_auc")
                        }
                per_seed.append(
                    {
                        "seed": split_seed,
                        "num_train": len(train_idx),
                        "num_test": len(test_idx),
                        "num_train_groups": len(train_groups),
                        "num_test_groups": len(test_groups),
                        "metrics_by_probe": metrics_by_model,
                        "control_metrics_by_probe": control_metrics_by_model,
                        "selectivity_by_probe": selectivity_by_model,
                        "turn_predictions_by_probe": turn_preds_by_probe if args.save_turn_preds else None,
                    }
                )

            if not per_seed:
                per_task[task] = {
                    "error": f"No valid train/test splits for task '{task}'.",
                    "label_counts": dict(label_counts),
                }
                continue

            # Aggregate over seeds.
            aggregate: dict[str, dict[str, dict[str, float | None]]] = {}
            aggregate_control: dict[str, dict[str, dict[str, float | None]]] = {}
            aggregate_selectivity: dict[str, dict[str, dict[str, float | None]]] = {}
            probe_names = sorted(per_seed[0]["metrics_by_probe"].keys())
            for probe_name in probe_names:
                aggregate[probe_name] = {}
                aggregate_control[probe_name] = {}
                aggregate_selectivity[probe_name] = {}
                for metric in ("accuracy", "balanced_accuracy", "roc_auc"):
                    vals = [
                        seed_row["metrics_by_probe"][probe_name][metric]
                        for seed_row in per_seed
                    ]
                    aggregate[probe_name][metric] = _summarize(vals)
                    ctrl_vals = [
                        (seed_row.get("control_metrics_by_probe") or {}).get(probe_name, {}).get(metric)
                        for seed_row in per_seed
                    ]
                    aggregate_control[probe_name][metric] = _summarize(ctrl_vals)
                    sel_vals = [
                        (seed_row.get("selectivity_by_probe") or {}).get(probe_name, {}).get(metric)
                        for seed_row in per_seed
                    ]
                    aggregate_selectivity[probe_name][metric] = _summarize(sel_vals)

            per_task[task] = {
                "label_counts": dict(label_counts),
                "num_rows": len(model_rows),
                "num_groups": len(set(groups)),
                "aggregate": aggregate,
                "aggregate_control": aggregate_control,
                "aggregate_selectivity": aggregate_selectivity,
                "per_seed": per_seed,
            }
        payload["results_by_model"][model_id] = per_task

        if args.per_transcript:
            # Per-transcript breakdown. Note: topic is typically constant within a transcript
            # and will produce a <2 class error (expected).
            by_transcript: dict[str, list[dict[str, object]]] = {}
            for r in model_rows:
                tid = str(r.get("transcript_id", ""))
                by_transcript.setdefault(tid, []).append(r)

            per_transcript_out: dict[str, dict[str, object]] = {}
            for transcript_id, t_rows in sorted(by_transcript.items()):
                t_blob: dict[str, object] = {}
                # Split by turn_id groups to avoid base/swapped leakage.
                uniq_turns = sorted(
                    {int(r.get("turn_id", -1)) for r in t_rows if int(r.get("turn_id", -1)) >= 0}
                )
                if len(uniq_turns) < 2:
                    continue

                for task in tasks:
                    labels = [_label_for_task(r, task) for r in t_rows]
                    vectors = [r["vector"] for r in t_rows]
                    label_counts = Counter(labels)
                    if len(label_counts) < 2:
                        t_blob[task] = {
                            "error": f"Task '{task}' has fewer than 2 classes for transcript '{transcript_id}'.",
                            "label_counts": dict(label_counts),
                        }
                        continue

                    per_seed: list[dict[str, object]] = []
                    for seed_offset in range(args.num_seeds):
                        split_seed = args.seed + seed_offset
                        try:
                            train_turns, test_turns = _split_turn_ids(
                                uniq_turns, test_size=args.test_size, seed=split_seed
                            )
                        except Exception:
                            continue

                        train_idx = [
                            i
                            for i, r in enumerate(t_rows)
                            if int(r.get("turn_id", -1)) in train_turns
                        ]
                        test_idx = [
                            i
                            for i, r in enumerate(t_rows)
                            if int(r.get("turn_id", -1)) in test_turns
                        ]
                        if not train_idx or not test_idx:
                            continue

                        X_train = [vectors[i] for i in train_idx]
                        y_train = [labels[i] for i in train_idx]
                        X_test = [vectors[i] for i in test_idx]
                        y_test = [labels[i] for i in test_idx]
                        test_rows = [t_rows[i] for i in test_idx]
                        if len(set(y_train)) < 2 or len(set(y_test)) < 2:
                            continue

                        metrics_by_probe, turn_preds_by_probe = _fit_and_eval(
                            X_train=X_train,
                            y_train=y_train,
                            X_test=X_test,
                            y_test=y_test,
                            max_iter=args.max_iter,
                            task=task,
                            collect_predictions=bool(args.save_turn_preds),
                            test_rows=test_rows,
                        )
                        control = _make_control_labels(
                            y_train=y_train, y_test=y_test, seed=split_seed + 1337
                        )
                        control_metrics_by_probe = None
                        selectivity_by_probe = None
                        if control is not None:
                            control_y_train, control_y_test = control
                            control_metrics_by_probe, _ = _fit_and_eval(
                                X_train=X_train,
                                y_train=control_y_train,
                                X_test=X_test,
                                y_test=control_y_test,
                                max_iter=args.max_iter,
                                task=task,
                                collect_predictions=False,
                            )
                            selectivity_by_probe = {}
                            for probe_name, true_metrics in metrics_by_probe.items():
                                ctrl_metrics = control_metrics_by_probe.get(probe_name, {})
                                selectivity_by_probe[probe_name] = {
                                    metric: _metric_diff(
                                        true_metrics.get(metric), ctrl_metrics.get(metric)
                                    )
                                    for metric in ("accuracy", "balanced_accuracy", "roc_auc")
                                }
                        per_seed.append(
                            {
                                "seed": split_seed,
                                "num_train": len(train_idx),
                                "num_test": len(test_idx),
                                "num_train_turns": len(train_turns),
                                "num_test_turns": len(test_turns),
                                "metrics_by_probe": metrics_by_probe,
                                "control_metrics_by_probe": control_metrics_by_probe,
                                "selectivity_by_probe": selectivity_by_probe,
                                "turn_predictions_by_probe": turn_preds_by_probe if args.save_turn_preds else None,
                            }
                        )

                    if not per_seed:
                        t_blob[task] = {
                            "error": f"No valid within-transcript splits for task '{task}'.",
                            "label_counts": dict(label_counts),
                        }
                        continue

                    aggregate: dict[str, dict[str, dict[str, float | None]]] = {}
                    aggregate_control: dict[str, dict[str, dict[str, float | None]]] = {}
                    aggregate_selectivity: dict[str, dict[str, dict[str, float | None]]] = {}
                    probe_names = sorted(per_seed[0]["metrics_by_probe"].keys())
                    for probe_name in probe_names:
                        aggregate[probe_name] = {}
                        aggregate_control[probe_name] = {}
                        aggregate_selectivity[probe_name] = {}
                        for metric in ("accuracy", "balanced_accuracy", "roc_auc"):
                            vals = [
                                seed_row["metrics_by_probe"][probe_name][metric]
                                for seed_row in per_seed
                            ]
                            aggregate[probe_name][metric] = _summarize(vals)
                            ctrl_vals = [
                                (seed_row.get("control_metrics_by_probe") or {})
                                .get(probe_name, {})
                                .get(metric)
                                for seed_row in per_seed
                            ]
                            aggregate_control[probe_name][metric] = _summarize(ctrl_vals)
                            sel_vals = [
                                (seed_row.get("selectivity_by_probe") or {}).get(probe_name, {}).get(metric)
                                for seed_row in per_seed
                            ]
                            aggregate_selectivity[probe_name][metric] = _summarize(sel_vals)

                    t_blob[task] = {
                        "label_counts": dict(label_counts),
                        "num_rows": len(t_rows),
                        "num_turns": len(uniq_turns),
                        "aggregate": aggregate,
                        "aggregate_control": aggregate_control,
                        "aggregate_selectivity": aggregate_selectivity,
                        "per_seed": per_seed,
                    }

                if t_blob:
                    per_transcript_out[transcript_id] = t_blob

            payload["results_by_model"][model_id]["per_transcript"] = per_transcript_out

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=True)


if __name__ == "__main__":
    main()
