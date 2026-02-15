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
        default=4000,
        help="Max iterations for linear solvers.",
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


def _fit_and_eval(
    X_train,
    y_train,
    X_test,
    y_test,
    max_iter: int,
    task: str,
) -> dict[str, dict[str, float | None]]:
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score, balanced_accuracy_score
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.svm import LinearSVC
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Install scikit-learn for probes.") from exc

    results: dict[str, dict[str, float | None]] = {}

    logreg = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=max_iter,
            class_weight="balanced",
            multi_class="auto",
            solver="lbfgs",
        ),
    )
    logreg.fit(X_train, y_train)
    pred = logreg.predict(X_test)
    model_out = {
        "accuracy": float(accuracy_score(y_test, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, pred)),
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

    linsvc = make_pipeline(
        StandardScaler(),
        LinearSVC(max_iter=max_iter, class_weight="balanced"),
    )
    linsvc.fit(X_train, y_train)
    pred = linsvc.predict(X_test)
    results["linear_svm"] = {
        "accuracy": float(accuracy_score(y_test, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, pred)),
        "roc_auc": None,
    }
    return results


def main() -> None:
    args = parse_args()
    if args.num_seeds <= 0:
        raise ValueError("--num-seeds must be positive.")
    if not (0.0 < args.test_size < 1.0):
        raise ValueError("--test-size must be in (0,1).")

    rows = _load_rows(args.embeddings)
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
                if len(set(y_train)) < 2 or len(set(y_test)) < 2:
                    continue

                metrics_by_model = _fit_and_eval(
                    X_train=X_train,
                    y_train=y_train,
                    X_test=X_test,
                    y_test=y_test,
                    max_iter=args.max_iter,
                    task=task,
                )
                per_seed.append(
                    {
                        "seed": split_seed,
                        "num_train": len(train_idx),
                        "num_test": len(test_idx),
                        "num_train_groups": len(train_groups),
                        "num_test_groups": len(test_groups),
                        "metrics_by_probe": metrics_by_model,
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
            probe_names = sorted(per_seed[0]["metrics_by_probe"].keys())
            for probe_name in probe_names:
                aggregate[probe_name] = {}
                for metric in ("accuracy", "balanced_accuracy", "roc_auc"):
                    vals = [
                        seed_row["metrics_by_probe"][probe_name][metric]
                        for seed_row in per_seed
                    ]
                    aggregate[probe_name][metric] = _summarize(vals)

            per_task[task] = {
                "label_counts": dict(label_counts),
                "num_rows": len(model_rows),
                "num_groups": len(set(groups)),
                "aggregate": aggregate,
                "per_seed": per_seed,
            }
        payload["results_by_model"][model_id] = per_task

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=True)


if __name__ == "__main__":
    main()
