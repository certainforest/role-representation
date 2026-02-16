#!/usr/bin/env python3
"""Extract simple turn-level embeddings for MVP role analysis."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dialogues", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--model-ids",
        type=str,
        required=True,
        help="Comma-separated HF model IDs.",
    )
    parser.add_argument("--layer", type=int, default=20)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument(
        "--backend",
        type=str,
        choices=("hf", "ndif"),
        default="hf",
        help="Extraction backend: local HF model load ('hf') or hosted NDIF ('ndif').",
    )
    parser.add_argument(
        "--hf-token",
        type=str,
        default="",
        help="Hugging Face token. Optional if HF_TOKEN/HUGGINGFACE_TOKEN is already set.",
    )
    parser.add_argument(
        "--ndif-api-key",
        type=str,
        default="",
        help="NDIF API key for hosted workflows. Stored in NDIF_API_KEY env at runtime.",
    )
    parser.add_argument(
        "--include-speaker-prefix",
        action="store_true",
        help="If set, format turns as 'Speaker: text'. Default is transcript-style text only.",
    )
    parser.add_argument(
        "--ndif-remote",
        action="store_true",
        help=(
            "If set, run NDIF-hosted remote execution by passing remote=True to nnsight trace. "
            "Requires python==3.12.* on the client per nnsight docs."
        ),
    )
    return parser.parse_args()


def _resolve_credentials(hf_token: str, ndif_api_key: str) -> dict[str, bool]:
    resolved_hf = hf_token.strip() or os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
    resolved_ndif = ndif_api_key.strip() or os.getenv("NDIF_API_KEY")
    if resolved_hf:
        os.environ["HF_TOKEN"] = resolved_hf
        os.environ["HUGGINGFACE_TOKEN"] = resolved_hf
        os.environ["HUGGINGFACEHUB_API_TOKEN"] = resolved_hf
    if resolved_ndif:
        os.environ["NDIF_API_KEY"] = resolved_ndif
    return {
        "has_hf_token": bool(resolved_hf),
        "has_ndif_api_key": bool(resolved_ndif),
    }


def _load_model_hf(model_id: str, device: str):
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Install torch + transformers for extraction.") from exc
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id).to(device)
    model.eval()
    return torch, tokenizer, model, "hf"


def _load_model_ndif(model_id: str, ndif_remote: bool):
    try:
        import torch
        from transformers import AutoTokenizer
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Install torch + transformers for extraction.") from exc
    try:
        from nnsight import LanguageModel
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "NDIF backend requires nnsight. Install with: pip install nnsight"
        ) from exc

    ndif_api_key = os.getenv("NDIF_API_KEY", "").strip()
    if not ndif_api_key:
        raise RuntimeError("Missing NDIF_API_KEY for --backend ndif.")

    # For true NDIF hosted execution, nnsight requires python==3.12.* on the client,
    # and remote=True must be passed to the tracing context.
    if ndif_remote:
        if (sys.version_info.major, sys.version_info.minor) != (3, 12):
            raise RuntimeError(
                "NDIF remote execution requires python==3.12.* on the client "
                f"(current: {sys.version_info.major}.{sys.version_info.minor}). "
                "Create a python 3.12 env and re-run with --ndif-remote."
            )
        from nnsight import CONFIG

        # Prefer explicit config set to avoid relying on env parsing differences.
        CONFIG.set_default_api_key(ndif_api_key)

    # Instantiate LanguageModel without provider/remote kwargs (those can leak into HF constructors
    # depending on nnsight/transformers versions). Remote execution is controlled at trace time.
    model = LanguageModel(model_id)

    tokenizer = getattr(model, "tokenizer", None)
    if tokenizer is None:
        hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN") or None
        tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)
    return torch, tokenizer, model, "ndif"


def _resolve_layer_module(model, layer: int):
    layer_paths = (
        ("model", "layers"),
        ("model", "model", "layers"),
        ("transformer", "h"),
    )
    for path in layer_paths:
        current = model
        ok = True
        for attr in path:
            if not hasattr(current, attr):
                ok = False
                break
            current = getattr(current, attr)
        if ok and hasattr(current, "__getitem__"):
            return current[layer]
    raise RuntimeError("Could not locate decoder layers on NDIF model wrapper.")


def _saved_value(saved_obj):
    return getattr(saved_obj, "value", saved_obj)


def _embed_turn(
    torch_mod,
    tokenizer,
    model,
    text: str,
    layer: int,
    span_start: int,
    span_end: int,
    backend: str,
    ndif_remote: bool,
) -> list[float]:
    encoded = tokenizer(text, return_tensors="pt", return_offsets_mapping=True)
    offsets = encoded["offset_mapping"][0].detach().cpu().tolist()
    selected_token_idxs = [
        i
        for i, (start, end) in enumerate(offsets)
        if end > start and start >= span_start and end <= span_end
    ]

    if backend == "hf":
        encoded = {k: v.to(model.device) for k, v in encoded.items()}
        model_inputs = {k: v for k, v in encoded.items() if k in {"input_ids", "attention_mask"}}
        with torch_mod.no_grad():
            outputs = model(**model_inputs, output_hidden_states=True)
        h = outputs.hidden_states[layer][0]  # [seq, dim]
    elif backend == "ndif":
        input_ids = encoded["input_ids"]
        attention_mask = encoded.get("attention_mask")
        try:
            trace_kwargs = {"attention_mask": attention_mask}
            if ndif_remote:
                trace_kwargs["remote"] = True
            with model.trace(input_ids, **trace_kwargs):
                layer_module = _resolve_layer_module(model, layer)
                h = layer_module.output[0]
                if hasattr(h, "dim") and h.dim() == 3:
                    h = h[0]  # [seq, dim]

                # Pool inside the trace so remote execution only downloads a single vector,
                # not the full [seq, dim] hidden state.
                if selected_token_idxs:
                    pooled = h[selected_token_idxs].mean(dim=0)
                else:
                    pooled = h.mean(dim=0)

                if ndif_remote:
                    saved = pooled.detach().cpu().save()
                else:
                    saved = pooled.save()
            pooled_value = _saved_value(saved)
        except Exception as exc:
            raise RuntimeError(
                "NDIF trace execution failed. "
                "This often indicates an incompatible nnsight/model/runtime combination. "
                f"Original error: {exc}"
            ) from exc
    else:
        raise ValueError(f"Unsupported backend '{backend}'.")

    if backend == "ndif":
        # Already pooled (and for remote, already detached+cpu'd) inside trace.
        return [float(x) for x in pooled_value.detach().cpu().tolist()]

    selected = [h[idx] for idx in selected_token_idxs]
    pooled = torch_mod.stack(selected, dim=0).mean(dim=0) if selected else h.mean(dim=0)
    return [float(x) for x in pooled.detach().cpu().tolist()]


def _render_context(
    turns: list[dict[str, str]],
    current_idx: int,
    include_speaker_prefix: bool,
) -> tuple[str, int, int]:
    lines: list[str] = []
    start_of_current = 0
    for idx, turn in enumerate(turns[: current_idx + 1]):
        if include_speaker_prefix:
            line = f'{turn["speaker"]}: {turn["text"]}'
        else:
            line = turn["text"]
        if idx == current_idx:
            start_of_current = sum(len(existing) + 1 for existing in lines)
        lines.append(line)
    text = "\n".join(lines)
    end_of_current = len(text)
    return text, start_of_current, end_of_current


def main() -> None:
    args = parse_args()
    if args.ndif_remote and args.backend != "ndif":
        raise ValueError("--ndif-remote requires --backend ndif.")
    with args.dialogues.open("r", encoding="utf-8") as handle:
        dialogues = json.load(handle)["dialogues"]

    credential_flags = _resolve_credentials(
        hf_token=args.hf_token,
        ndif_api_key=args.ndif_api_key,
    )
    model_ids = [m.strip() for m in args.model_ids.split(",") if m.strip()]
    if not model_ids:
        raise ValueError("Pass at least one model via --model-ids.")
    rows: list[dict[str, object]] = []
    for model_id in model_ids:
        if args.backend == "ndif":
            torch_mod, tokenizer, model, active_backend = _load_model_ndif(
                model_id, ndif_remote=args.ndif_remote
            )
        else:
            torch_mod, tokenizer, model, active_backend = _load_model_hf(model_id, args.device)
        for item in dialogues:
            transcript_id = item["transcript_id"]
            topic = item.get("topic", "")
            for variant in ("base", "speaker_swapped"):
                turns = item[variant]
                for turn_idx, turn in enumerate(turns):
                    prompt, span_start, span_end = _render_context(
                        turns=turns,
                        current_idx=turn_idx,
                        include_speaker_prefix=args.include_speaker_prefix,
                    )
                    vector = _embed_turn(
                        torch_mod=torch_mod,
                        tokenizer=tokenizer,
                        model=model,
                        text=prompt,
                        layer=args.layer,
                        span_start=span_start,
                        span_end=span_end,
                        backend=active_backend,
                        ndif_remote=args.ndif_remote,
                    )
                    rows.append(
                        {
                            "model_id": model_id,
                            "transcript_id": transcript_id,
                            "topic": topic,
                            "variant": variant,
                            "layer": args.layer,
                            "speaker": turn["speaker"],
                            "turn_id": turn_idx,
                            "text": turn["text"],
                            "vector": vector,
                        }
                    )

    payload = {
        "metadata": {
            "model_ids": model_ids,
            "layer": args.layer,
            "backend": args.backend,
            "include_speaker_prefix": args.include_speaker_prefix,
            "credentials": credential_flags,
        },
        "turn_embeddings": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=True)


if __name__ == "__main__":
    main()
