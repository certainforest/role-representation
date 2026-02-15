#!/usr/bin/env python3
"""Analyze representation geometry for speaker-role structure."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for geometry figures.",
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=4000,
        help="Max points per model for scatter plotting.",
    )
    parser.add_argument(
        "--max-trajectory-transcripts",
        type=int,
        default=6,
        help="How many transcripts to include in trajectory plot.",
    )
    return parser.parse_args()


def _load_rows(path: Path) -> list[dict[str, object]]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    rows = payload.get("turn_embeddings", [])
    if not isinstance(rows, list) or not rows:
        raise ValueError("No rows found under 'turn_embeddings'.")
    return rows


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def _norm(a: list[float]) -> float:
    return math.sqrt(sum(x * x for x in a))


def _sub(a: list[float], b: list[float]) -> list[float]:
    return [x - y for x, y in zip(a, b)]


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


def _cos(a: list[float], b: list[float], eps: float = 1e-8) -> float:
    return _dot(a, b) / max(_norm(a) * _norm(b), eps)


def _safe_model_name(model_id: str) -> str:
    return model_id.replace("/", "__").replace(":", "_")


def _scatter_pca(
    rows: list[dict[str, object]],
    output_path: Path,
    max_points: int,
) -> None:
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        from sklearn.decomposition import PCA
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Install matplotlib, numpy, scikit-learn for geometry plots.") from exc

    sampled = rows[:max_points]
    X = np.array([row["vector"] for row in sampled], dtype=float)
    labels = [str(row.get("speaker", "")) for row in sampled]
    variants = [str(row.get("variant", "")) for row in sampled]

    pca = PCA(n_components=2, random_state=0)
    pts = pca.fit_transform(X)
    evr = pca.explained_variance_ratio_.tolist()

    fig, ax = plt.subplots(figsize=(7.5, 6.0))
    color_map = {"Alice": "#1f77b4", "Bob": "#ff7f0e"}
    marker_map = {"base": "o", "speaker_swapped": "x"}
    for role in sorted(set(labels)):
        for variant in sorted(set(variants)):
            idx = [i for i, (r, v) in enumerate(zip(labels, variants)) if r == role and v == variant]
            if not idx:
                continue
            ax.scatter(
                pts[idx, 0],
                pts[idx, 1],
                s=12,
                alpha=0.55,
                c=color_map.get(role, "#444444"),
                marker=marker_map.get(variant, "."),
                label=f"{role}/{variant}",
            )
    ax.set_xlabel(f"PC1 ({evr[0] * 100.0:.1f}% var)")
    ax.set_ylabel(f"PC2 ({evr[1] * 100.0:.1f}% var)")
    ax.set_title("Role Geometry PCA Scatter")
    handles, labels_out = ax.get_legend_handles_labels()
    uniq = dict(zip(labels_out, handles))
    ax.legend(uniq.values(), uniq.keys(), loc="best", fontsize=8)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _trajectory_plot(rows: list[dict[str, object]], output_path: Path, max_transcripts: int) -> None:
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        from sklearn.decomposition import PCA
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Install matplotlib, numpy, scikit-learn for geometry plots.") from exc

    by_transcript: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        if str(row.get("variant", "")) != "base":
            continue
        by_transcript[str(row.get("transcript_id", ""))].append(row)

    transcript_ids = sorted(by_transcript.keys())[:max_transcripts]
    selected = []
    for t_id in transcript_ids:
        seq = sorted(by_transcript[t_id], key=lambda r: int(r.get("turn_id", 0)))
        selected.extend(seq)
    if not selected:
        return

    X = np.array([row["vector"] for row in selected], dtype=float)
    pts = PCA(n_components=2, random_state=0).fit_transform(X)

    fig, ax = plt.subplots(figsize=(8.0, 6.0))
    start = 0
    for t_id in transcript_ids:
        seq = sorted(by_transcript[t_id], key=lambda r: int(r.get("turn_id", 0)))
        n = len(seq)
        sub = pts[start : start + n]
        start += n
        ax.plot(sub[:, 0], sub[:, 1], marker="o", markersize=3, linewidth=1.0, alpha=0.8, label=t_id)
    ax.set_title("Turn Trajectories in PCA Space (base)")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend(loc="best", fontsize=7)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _direction_heatmap(rows: list[dict[str, object]], output_path: Path) -> list[list[float]]:
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Install matplotlib, numpy for geometry plots.") from exc

    by_transcript: dict[str, dict[str, list[list[float]]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        if str(row.get("variant", "")) != "base":
            continue
        by_transcript[str(row.get("transcript_id", ""))][str(row.get("speaker", ""))].append(row["vector"])

    transcript_ids = []
    directions = []
    for t_id in sorted(by_transcript.keys()):
        alice = by_transcript[t_id].get("Alice", [])
        bob = by_transcript[t_id].get("Bob", [])
        if not alice or not bob:
            continue
        direction = _sub(_mean(alice), _mean(bob))
        directions.append(direction)
        transcript_ids.append(t_id)

    if not directions:
        return []

    n = len(directions)
    mat = [[0.0 for _ in range(n)] for _ in range(n)]
    for i in range(n):
        for j in range(n):
            mat[i][j] = _cos(directions[i], directions[j])

    arr = np.array(mat, dtype=float)
    fig, ax = plt.subplots(figsize=(7.0, 6.0))
    im = ax.imshow(arr, vmin=-1.0, vmax=1.0, cmap="coolwarm")
    ax.set_title("Transcript Role-Direction Cosine")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(transcript_ids, rotation=80, fontsize=6)
    ax.set_yticklabels(transcript_ids, fontsize=6)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return mat


def main() -> None:
    args = parse_args()
    rows = _load_rows(args.embeddings)
    model_ids = sorted({str(row.get("model_id", "")) for row in rows if row.get("model_id")})
    if not model_ids:
        raise ValueError("No model_id values found in embeddings.")

    output_payload: dict[str, object] = {
        "metadata": {
            "script": "mvp_role_geometry.py",
            "max_points": args.max_points,
            "max_trajectory_transcripts": args.max_trajectory_transcripts,
        },
        "results_by_model": {},
    }

    for model_id in model_ids:
        model_rows = [row for row in rows if str(row.get("model_id", "")) == model_id]
        base_rows = [row for row in model_rows if str(row.get("variant", "")) == "base"]
        alice = [row["vector"] for row in base_rows if row.get("speaker") == "Alice"]
        bob = [row["vector"] for row in base_rows if row.get("speaker") == "Bob"]
        if not alice or not bob:
            output_payload["results_by_model"][model_id] = {
                "error": "Missing Alice/Bob rows in base variant."
            }
            continue

        mu_a = _mean(alice)
        mu_b = _mean(bob)
        between_dist = _norm(_sub(mu_a, mu_b))
        within_a = sum(_norm(_sub(vec, mu_a)) for vec in alice) / float(len(alice))
        within_b = sum(_norm(_sub(vec, mu_b)) for vec in bob) / float(len(bob))

        try:
            from sklearn.metrics import silhouette_score
            import numpy as np
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("Install scikit-learn and numpy for geometry metrics.") from exc
        X = np.array([row["vector"] for row in base_rows], dtype=float)
        y = np.array([0 if row.get("speaker") == "Alice" else 1 for row in base_rows], dtype=int)
        silhouette = float(silhouette_score(X, y)) if len(set(y.tolist())) == 2 else None

        safe_name = _safe_model_name(model_id)
        scatter_path = args.output_dir / f"{safe_name}_pca_scatter.png"
        traj_path = args.output_dir / f"{safe_name}_trajectory.png"
        heatmap_path = args.output_dir / f"{safe_name}_direction_heatmap.png"
        _scatter_pca(model_rows, scatter_path, max_points=args.max_points)
        _trajectory_plot(model_rows, traj_path, max_transcripts=args.max_trajectory_transcripts)
        cosine_mat = _direction_heatmap(model_rows, heatmap_path)

        # Swap inversion check in geometry form.
        by_key: dict[tuple[str, int], dict[str, list[float]]] = {}
        for row in model_rows:
            key = (str(row.get("transcript_id", "")), int(row.get("turn_id", -1)))
            by_key.setdefault(key, {})[str(row.get("variant", ""))] = row["vector"]
        role_direction = _sub(mu_a, mu_b)
        inversion = []
        for pair in by_key.values():
            base = pair.get("base")
            swapped = pair.get("speaker_swapped")
            if base is None or swapped is None:
                continue
            inversion.append(1.0 if _dot(base, role_direction) * _dot(swapped, role_direction) < 0 else 0.0)
        swap_inversion_rate = (
            sum(inversion) / float(len(inversion))
            if inversion
            else None
        )

        output_payload["results_by_model"][model_id] = {
            "metrics": {
                "num_rows": len(model_rows),
                "num_base_rows": len(base_rows),
                "between_role_centroid_distance": between_dist,
                "within_role_dispersion_alice": within_a,
                "within_role_dispersion_bob": within_b,
                "distance_over_dispersion": between_dist / max((within_a + within_b) / 2.0, 1e-8),
                "silhouette_role_base_only": silhouette,
                "swap_projection_inversion_rate": swap_inversion_rate,
                "mean_transcript_direction_cosine": (
                    sum(
                        cosine_mat[i][j]
                        for i in range(len(cosine_mat))
                        for j in range(i + 1, len(cosine_mat))
                    )
                    / max((len(cosine_mat) * (len(cosine_mat) - 1)) / 2.0, 1.0)
                    if cosine_mat
                    else None
                ),
            },
            "artifacts": {
                "pca_scatter": str(scatter_path),
                "trajectory_plot": str(traj_path),
                "direction_heatmap": str(heatmap_path),
            },
        }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("w", encoding="utf-8") as handle:
        json.dump(output_payload, handle, indent=2, ensure_ascii=True)


if __name__ == "__main__":
    main()
