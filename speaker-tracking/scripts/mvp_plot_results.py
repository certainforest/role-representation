#!/usr/bin/env python3
"""Plot MVP role-stability comparison across models."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--title",
        type=str,
        default="MVP Role-Binding Signals Across Models",
    )
    return parser.parse_args()


def _short_name(model_id: str) -> str:
    return (
        model_id.replace("meta-llama/Meta-Llama-3.1-8B-Instruct", "Llama-3.1-8B")
        .replace("allenai/OLMo-3-1025-7B", "OLMo-3-1025-7B")
        .replace("google/gemma-2-9b-it", "Gemma-2-9B-IT")
        .replace("google/gemma-3-4b-pt", "Gemma-3-4B-PT")
    )


def main() -> None:
    args = parse_args()
    with args.results.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    results_by_model: dict[str, dict] = payload.get("results_by_model", {})
    if not results_by_model:
        raise ValueError("No results found under 'results_by_model'.")

    model_ids = sorted(results_by_model.keys())
    labels = [_short_name(mid) for mid in model_ids]
    mean_pairwise = [
        float(results_by_model[mid]["metrics"]["mean_pairwise_role_cosine"])
        for mid in model_ids
    ]
    mean_swap = [
        float(results_by_model[mid]["metrics"]["mean_swap_cosine_should_be_negative"])
        for mid in model_ids
    ]

    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Install matplotlib and numpy to plot results.") from exc

    x = np.arange(len(labels))
    width = 0.36
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.bar(x - width / 2.0, mean_pairwise, width, label="Mean Pairwise Role Cosine")
    ax.bar(x + width / 2.0, mean_swap, width, label="Mean Swap Cosine (lower is better)")

    ax.axhline(0.0, color="gray", linewidth=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("Cosine")
    ax.set_title(args.title)
    ax.legend()
    fig.tight_layout()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
