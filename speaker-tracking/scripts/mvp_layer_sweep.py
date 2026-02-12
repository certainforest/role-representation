#!/usr/bin/env python3
"""Run MVP extraction/stability across layers and plot merged results."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dialogues", type=Path, required=True)
    parser.add_argument(
        "--model-ids",
        type=str,
        required=True,
        help="Comma-separated HF model IDs.",
    )
    parser.add_argument(
        "--layers",
        type=str,
        required=True,
        help="Comma-separated integer layers, e.g. '10,15,20'.",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=Path("speaker-tracking/data/layer_sweep"),
        help="Directory to store embeddings, results, and merged outputs.",
    )
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument(
        "--backend",
        type=str,
        choices=("hf", "ndif"),
        default="hf",
        help="Embedding extraction backend passed to mvp_extract_turn_embeddings.py.",
    )
    parser.add_argument("--hf-token", type=str, default="")
    parser.add_argument("--ndif-api-key", type=str, default="")
    parser.add_argument("--include-speaker-prefix", action="store_true")
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip extraction/stability if output files already exist.",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="MVP Role-Binding Layer Sweep",
    )
    return parser.parse_args()


def _parse_layers(raw: str) -> list[int]:
    layers = sorted({int(x.strip()) for x in raw.split(",") if x.strip()})
    if not layers:
        raise ValueError("Pass at least one layer via --layers.")
    return layers


def _parse_model_ids(raw: str) -> list[str]:
    model_ids = [x.strip() for x in raw.split(",") if x.strip()]
    if not model_ids:
        raise ValueError("Pass at least one model via --model-ids.")
    return model_ids


def _safe_model_name(model_id: str) -> str:
    return model_id.replace("/", "__").replace(":", "_")


def _short_name(model_id: str) -> str:
    return (
        model_id.replace("meta-llama/Meta-Llama-3.1-8B-Instruct", "Llama-3.1-8B")
        .replace("allenai/OLMo-3-1025-7B", "OLMo-3-1025-7B")
        .replace("google/gemma-2-9b-it", "Gemma-2-9B-IT")
        .replace("google/gemma-3-4b-pt", "Gemma-3-4B-PT")
    )


def _run(command: list[str]) -> None:
    subprocess.run(command, check=True)


def _plot_layer_sweep(payload: dict[str, object], output_path: Path, title: str) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Install matplotlib to plot layer-sweep results.") from exc

    layers: list[int] = payload["layers"]
    model_ids: list[str] = payload["model_ids"]
    results_by_layer: dict[str, dict[str, dict[str, object]]] = payload["results_by_layer"]

    fig, (ax_pairwise, ax_swap) = plt.subplots(1, 2, figsize=(13.5, 5.5), sharex=True)
    for model_id in model_ids:
        y_pairwise = []
        y_swap = []
        for layer in layers:
            metrics = results_by_layer[str(layer)][model_id]["metrics"]
            y_pairwise.append(float(metrics["mean_pairwise_role_cosine"]))
            y_swap.append(float(metrics["mean_swap_cosine_should_be_negative"]))
        label = _short_name(model_id)
        ax_pairwise.plot(layers, y_pairwise, marker="o", label=label)
        ax_swap.plot(layers, y_swap, marker="o", label=label)

    ax_pairwise.set_title("Mean Pairwise Role Cosine")
    ax_pairwise.set_ylabel("Cosine")
    ax_pairwise.set_xlabel("Layer")
    ax_pairwise.axhline(0.0, color="gray", linewidth=1.0)

    ax_swap.set_title("Mean Swap Cosine (lower is better)")
    ax_swap.set_xlabel("Layer")
    ax_swap.axhline(0.0, color="gray", linewidth=1.0)

    handles, labels = ax_pairwise.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=min(4, max(1, len(labels))))
    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0, 1, 0.92))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    layers = _parse_layers(args.layers)
    model_ids = _parse_model_ids(args.model_ids)

    script_dir = Path(__file__).resolve().parent
    extract_script = script_dir / "mvp_extract_turn_embeddings.py"
    stability_script = script_dir / "mvp_role_stability.py"

    embeddings_dir = args.work_dir / "embeddings"
    results_dir = args.work_dir / "results"
    embeddings_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    results_by_layer: dict[str, dict[str, dict[str, object]]] = {}
    for layer in layers:
        layer_key = str(layer)
        results_by_layer[layer_key] = {}
        for model_id in model_ids:
            safe_name = _safe_model_name(model_id)
            emb_path = embeddings_dir / f"{safe_name}_layer{layer}.json"
            res_path = results_dir / f"{safe_name}_layer{layer}.json"

            if not (args.skip_existing and emb_path.exists()):
                extract_cmd = [
                    sys.executable,
                    str(extract_script),
                    "--dialogues",
                    str(args.dialogues),
                    "--output",
                    str(emb_path),
                    "--model-ids",
                    model_id,
                    "--layer",
                    str(layer),
                    "--backend",
                    args.backend,
                    "--device",
                    args.device,
                ]
                if args.hf_token:
                    extract_cmd.extend(["--hf-token", args.hf_token])
                if args.ndif_api_key:
                    extract_cmd.extend(["--ndif-api-key", args.ndif_api_key])
                if args.include_speaker_prefix:
                    extract_cmd.append("--include-speaker-prefix")
                _run(extract_cmd)

            if not (args.skip_existing and res_path.exists()):
                stability_cmd = [
                    sys.executable,
                    str(stability_script),
                    "--embeddings",
                    str(emb_path),
                    "--output",
                    str(res_path),
                ]
                _run(stability_cmd)

            with res_path.open("r", encoding="utf-8") as handle:
                result_payload = json.load(handle)
            per_model = result_payload.get("results_by_model", {})
            if model_id not in per_model:
                available = ", ".join(sorted(per_model.keys()))
                raise ValueError(
                    f"Model '{model_id}' missing in {res_path}. Available keys: {available}"
                )
            results_by_layer[layer_key][model_id] = per_model[model_id]

    merged_payload = {
        "layers": layers,
        "model_ids": model_ids,
        "results_by_layer": results_by_layer,
    }
    merged_json_path = args.work_dir / "mvp_layer_sweep_results.json"
    with merged_json_path.open("w", encoding="utf-8") as handle:
        json.dump(merged_payload, handle, indent=2, ensure_ascii=True)

    plot_path = args.work_dir / "mvp_layer_sweep.png"
    _plot_layer_sweep(merged_payload, output_path=plot_path, title=args.title)

    print(f"Wrote merged JSON: {merged_json_path}")
    print(f"Wrote layer-sweep plot: {plot_path}")


if __name__ == "__main__":
    main()
