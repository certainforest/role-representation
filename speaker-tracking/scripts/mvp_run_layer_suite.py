#!/usr/bin/env python3
"""Run MVP extraction + linear probe for one model across one or more layers.

This is a thin convenience wrapper around the existing scripts. It standardizes:
- output directory layout: speaker-tracking/data/meanpooled_layer{L}/
- output naming: mvp_<analysis>_<tag>.json

Example:
  python speaker-tracking/scripts/mvp_run_layer_suite.py \
    --dialogues speaker-tracking/data/meanpooled_layer20/mvp_dialogues.json \
    --model-id meta-llama/Meta-Llama-3.1-8B-Instruct \
    --tag llama31_8b \
    --layers 0,5,10,15,20,25,30 \
    --backend ndif \
    --ndif-remote
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path


def _parse_layers(raw: str) -> list[int]:
    vals: list[int] = []
    for chunk in raw.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        vals.append(int(chunk))
    if not vals:
        raise ValueError("Pass at least one layer via --layers (comma-separated ints).")
    return sorted(set(vals))


def _default_tag(model_id: str) -> str:
    # Keep only last path component and normalize to something filename-safe.
    base = model_id.split("/")[-1].strip().lower()
    base = base.replace("meta-llama-", "").replace("meta_llama_", "")
    base = re.sub(r"[^a-z0-9]+", "_", base)
    base = re.sub(r"_+", "_", base).strip("_")
    return base or "model"


def _run(cmd: list[str], *, dry_run: bool) -> None:
    if dry_run:
        print("+ " + " ".join(cmd))
        return
    subprocess.run(cmd, check=True)


def _maybe_set_env(var: str, value: str) -> None:
    if value.strip():
        os.environ[var] = value.strip()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dialogues", type=Path, required=True)
    p.add_argument("--model-id", type=str, required=True)
    p.add_argument("--tag", type=str, default="")
    p.add_argument(
        "--layers",
        type=str,
        required=True,
        help="Comma-separated layer indices, e.g. '0,5,10,15,20'.",
    )
    p.add_argument(
        "--output-root",
        type=Path,
        default=Path("speaker-tracking/data"),
        help="Root directory that will contain meanpooled_layer{L}/ subdirs.",
    )
    p.add_argument(
        "--layer-dir-template",
        type=str,
        default="meanpooled_layer{layer}",
        help="Subdir template under --output-root (must include '{layer}').",
    )
    p.add_argument(
        "--run-id",
        type=str,
        default="",
        help="Optional suffix appended to filenames/dirs (e.g. 'rerun').",
    )
    p.add_argument("--skip-existing", action="store_true", help="Skip steps with existing outputs.")
    p.add_argument("--dry-run", action="store_true", help="Print commands without running.")

    # Extraction flags (forwarded to mvp_extract_turn_embeddings.py).
    p.add_argument("--backend", type=str, choices=("hf", "ndif"), default="hf")
    p.add_argument("--ndif-remote", action="store_true")
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--hf-token", type=str, default="")
    p.add_argument("--ndif-api-key", type=str, default="")
    p.add_argument("--pooling", type=str, choices=("mean", "last"), default="mean")
    p.add_argument("--include-speaker-prefix", action="store_true")
    p.add_argument("--token-level", action="store_true")

    # Probe flags.
    p.add_argument("--split-mode", type=str, choices=("transcript", "topic"), default="transcript")
    p.add_argument("--num-seeds", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)

    return p.parse_args()


def main() -> None:
    args = parse_args()
    layers = _parse_layers(args.layers)
    if "{layer}" not in args.layer_dir_template:
        raise ValueError("--layer-dir-template must include '{layer}'.")

    tag = args.tag.strip() or _default_tag(args.model_id)
    file_tag = f"{tag}_{args.run_id.strip()}" if args.run_id.strip() else tag

    # Set env vars for downstream scripts that read them.
    _maybe_set_env("HF_TOKEN", args.hf_token)
    _maybe_set_env("HUGGINGFACE_TOKEN", args.hf_token)
    _maybe_set_env("NDIF_API_KEY", args.ndif_api_key)

    py = sys.executable
    scripts_dir = Path(__file__).resolve().parent

    backend_label = args.backend
    if args.backend == "ndif" and args.ndif_remote:
        backend_label = "ndif_remote"

    for layer in layers:
        layer_dir = args.output_root / args.layer_dir_template.format(layer=layer)
        if not args.dry_run:
            layer_dir.mkdir(parents=True, exist_ok=True)

        embeddings_name = f"mvp_turn_embeddings_{backend_label}_{file_tag}.json"
        embeddings_path = layer_dir / embeddings_name

        # --- Extract ---
        extract_cmd = [
            py,
            str(scripts_dir / "mvp_extract_turn_embeddings.py"),
            "--dialogues",
            str(args.dialogues),
            "--output",
            str(embeddings_path),
            "--model-ids",
            args.model_id,
            "--layer",
            str(layer),
            "--backend",
            args.backend,
            "--device",
            args.device,
            "--pooling",
            args.pooling,
        ]
        if args.ndif_remote:
            extract_cmd.append("--ndif-remote")
        if args.include_speaker_prefix:
            extract_cmd.append("--include-speaker-prefix")
        if args.token_level:
            extract_cmd.append("--token-level")

        # Only pass tokens if user provided; otherwise rely on environment.
        if args.hf_token.strip():
            extract_cmd += ["--hf-token", args.hf_token.strip()]
        if args.ndif_api_key.strip():
            extract_cmd += ["--ndif-api-key", args.ndif_api_key.strip()]

        if not (args.skip_existing and embeddings_path.exists()):
            _run(extract_cmd, dry_run=args.dry_run)

        # --- Analyses ---
        def out_json(name: str) -> Path:
            return layer_dir / f"{name}_{file_tag}.json"

        linear_probe_out = out_json("mvp_linear_probe")
        if not (args.skip_existing and linear_probe_out.exists()):
            _run(
                [
                    py,
                    str(scripts_dir / "mvp_linear_probe.py"),
                    "--embeddings",
                    str(embeddings_path),
                    "--output",
                    str(linear_probe_out),
                    "--tasks",
                    "role,variant,topic",
                    "--split-mode",
                    args.split_mode,
                    "--num-seeds",
                    str(args.num_seeds),
                    "--seed",
                    str(args.seed),
                    "--per-transcript",
                ],
                dry_run=args.dry_run,
            )


if __name__ == "__main__":
    main()

