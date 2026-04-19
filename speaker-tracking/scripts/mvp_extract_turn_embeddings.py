#!/usr/bin/env python3
"""Extract simple turn-level embeddings for MVP role analysis."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
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
        "--pooling",
        type=str,
        choices=("mean", "last"),
        default="last",
        help=(
            "Pooling for turn-level vectors. 'last' (default) uses the last token in the current-turn "
            "span; 'mean' averages token vectors in the current-turn span. Ignored if --token-level is set."
        ),
    )
    parser.add_argument(
        "--token-level",
        action="store_true",
        help=(
            "If set, emit one row per token in the current-turn span (same labels as the turn). "
            "This produces much larger outputs but enables token-level probing."
        ),
    )
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
    parser.add_argument(
        "--max-context-turns",
        type=int,
        default=None,
        help=(
            "If set, limit the rolling context window to the last N turns when building the "
            "prompt for each turn embedding. Useful for longer dialogues that would otherwise "
            "exceed the model's context capacity. Default: no limit (full history)."
        ),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help=(
            "Path to a checkpoint JSONL file for resume support. Each completed row is appended "
            "immediately. On re-run, already-checkpointed (transcript_id, variant, turn_id) keys "
            "are skipped. Defaults to <output>.ckpt.jsonl if not specified."
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
        ("model", "language_model", "layers"),
        ("model", "language_model", "model", "layers"),
        ("language_model", "layers"),
        ("language_model", "model", "layers"),
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


def _selected_token_idxs(
    offsets: list[list[int]],
    span_start: int,
    span_end: int,
) -> list[int]:
    return [
        i
        for i, (start, end) in enumerate(offsets)
        if end > start and start >= span_start and end <= span_end
    ]


def _embed_turn_pooled(
    torch_mod,
    tokenizer,
    model,
    text: str,
    layer: int,
    span_start: int,
    span_end: int,
    backend: str,
    ndif_remote: bool,
    pooling: str,
) -> list[float]:
    encoded = tokenizer(text, return_tensors="pt", return_offsets_mapping=True)
    offsets = encoded["offset_mapping"][0].detach().cpu().tolist()
    selected_token_idxs = _selected_token_idxs(offsets, span_start=span_start, span_end=span_end)

    if backend == "hf":
        encoded = {k: v.to(model.device) for k, v in encoded.items()}
        model_inputs = {k: v for k, v in encoded.items() if k in {"input_ids", "attention_mask"}}
        with torch_mod.no_grad():
            outputs = model(**model_inputs, output_hidden_states=True)
        h = outputs.hidden_states[layer][0]  # [seq, dim]
    elif backend == "ndif":
        input_ids = encoded["input_ids"]
        attention_mask = encoded.get("attention_mask")
        _RETRYABLE = (
            "connection error",
            "namespaces failed",
            "timeout",
            "unavailable",
            "try again later",
            "submitting request",
            "remoteexception",
            "502",
            "503",
            "504",
        )
        max_retries = 5
        for attempt in range(max_retries):
            try:
                # For NDIF remote execution, nnsight's documented API is to pass the prompt
                # string and set remote=True on the trace context (client must be python 3.12).
                if ndif_remote:
                    trace_input = text
                    trace_kwargs = {"remote": True}
                else:
                    trace_input = input_ids
                    trace_kwargs = {"attention_mask": attention_mask}

                with model.trace(trace_input, **trace_kwargs):
                    layer_module = _resolve_layer_module(model, layer)
                    h = layer_module.output[0]
                    if hasattr(h, "dim") and h.dim() == 3:
                        h = h[0]  # [seq, dim]

                    # Pool inside the trace so remote execution only downloads a single vector,
                    # not the full [seq, dim] hidden state.
                    if selected_token_idxs:
                        if pooling == "last":
                            pooled = h[selected_token_idxs[-1]]
                        else:
                            pooled = h[selected_token_idxs].mean(dim=0)
                    else:
                        # Fallback if offsets fail: keep prior behavior.
                        pooled = h[-1] if pooling == "last" else h.mean(dim=0)

                    if ndif_remote:
                        saved = pooled.detach().cpu().save()
                    else:
                        saved = pooled.save()
                pooled_value = _saved_value(saved)
                break  # success
            except Exception as exc:
                err_str = str(exc).lower()
                is_retryable = any(kw in err_str for kw in _RETRYABLE)
                if is_retryable and attempt < max_retries - 1:
                    wait = 5 * (2 ** attempt)  # 5, 10, 20, 40s
                    print(f"  [retry {attempt+1}/{max_retries-1}] transient error, sleeping {wait}s: {exc}", flush=True)
                    time.sleep(wait)
                    continue
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

    if selected_token_idxs:
        if pooling == "last":
            pooled = h[selected_token_idxs[-1]]
        else:
            pooled = h[selected_token_idxs].mean(dim=0)
    else:
        pooled = h[-1] if pooling == "last" else h.mean(dim=0)
    return [float(x) for x in pooled.detach().cpu().tolist()]


def _embed_turn_tokens(
    torch_mod,
    tokenizer,
    model,
    text: str,
    layer: int,
    span_start: int,
    span_end: int,
    backend: str,
    ndif_remote: bool,
) -> tuple[list[dict[str, object]], bool]:
    """Return per-token vectors for the current-turn span.

    Returns (token_rows, used_fallback). Each token_row has token_idx, token_id, token, vector.
    """
    encoded = tokenizer(text, return_tensors="pt", return_offsets_mapping=True)
    offsets = encoded["offset_mapping"][0].detach().cpu().tolist()
    selected_token_idxs = _selected_token_idxs(offsets, span_start=span_start, span_end=span_end)

    input_ids = encoded["input_ids"][0].detach().cpu().tolist()
    token_strs = tokenizer.convert_ids_to_tokens(input_ids)

    used_fallback = False
    if not selected_token_idxs:
        # If offsets fail, fall back to the last token so callers still get *something*.
        used_fallback = True
        selected_token_idxs = [len(input_ids) - 1]

    if backend == "hf":
        encoded = {k: v.to(model.device) for k, v in encoded.items()}
        model_inputs = {k: v for k, v in encoded.items() if k in {"input_ids", "attention_mask"}}
        with torch_mod.no_grad():
            outputs = model(**model_inputs, output_hidden_states=True)
        h = outputs.hidden_states[layer][0]  # [seq, dim]
        selected = h[selected_token_idxs]  # [n, dim]
        selected_value = selected.detach().cpu()
    elif backend == "ndif":
        attention_mask = encoded.get("attention_mask")
        try:
            if ndif_remote:
                trace_input = text
                trace_kwargs = {"remote": True}
            else:
                trace_input = encoded["input_ids"]
                trace_kwargs = {"attention_mask": attention_mask}

            with model.trace(trace_input, **trace_kwargs):
                layer_module = _resolve_layer_module(model, layer)
                h = layer_module.output[0]
                if hasattr(h, "dim") and h.dim() == 3:
                    h = h[0]  # [seq, dim]

                selected = h[selected_token_idxs]  # [n, dim]
                if ndif_remote:
                    saved = selected.detach().cpu().save()
                else:
                    saved = selected.save()
            selected_value = _saved_value(saved).detach().cpu()
        except Exception as exc:
            raise RuntimeError(
                "NDIF trace execution failed for token-level extraction. "
                "This can happen if the remote runtime cannot return large tensors. "
                f"Original error: {exc}"
            ) from exc
    else:
        raise ValueError(f"Unsupported backend '{backend}'.")

    token_rows: list[dict[str, object]] = []
    for local_i, tok_i in enumerate(selected_token_idxs):
        tok_id = int(input_ids[tok_i])
        tok = str(token_strs[tok_i]) if tok_i < len(token_strs) else ""
        vec = selected_value[local_i].tolist()
        token_rows.append(
            {
                "token_idx": int(tok_i),
                "token_id": tok_id,
                "token": tok,
                "vector": [float(x) for x in vec],
            }
        )
    return token_rows, used_fallback


def _render_context(
    turns: list[dict[str, str]],
    current_idx: int,
    include_speaker_prefix: bool,
    max_context_turns: int | None = None,
) -> tuple[str, int, int]:
    if max_context_turns is not None and max_context_turns > 0:
        start_idx = max(0, current_idx + 1 - max_context_turns)
        window = turns[start_idx : current_idx + 1]
    else:
        window = turns[: current_idx + 1]
    lines: list[str] = []
    start_of_current = 0
    for idx, turn in enumerate(window):
        if include_speaker_prefix:
            line = f'{turn["speaker"]}: {turn["text"]}'
        else:
            line = turn["text"]
        if idx == len(window) - 1:  # current turn is always the last element
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

    # Checkpoint: resume support — load already-completed rows keyed by (model_id, transcript_id, variant, turn_id).
    ckpt_path: Path = args.checkpoint or args.output.with_suffix(".ckpt.jsonl")
    done_keys: set[tuple] = set()
    rows: list[dict[str, object]] = []
    if ckpt_path.exists():
        with ckpt_path.open("r", encoding="utf-8") as ckpt_fh:
            for line in ckpt_fh:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                rows.append(r)
                done_keys.add((r["model_id"], r["transcript_id"], r["variant"], r["turn_id"]))
        print(f"Resuming: loaded {len(rows)} rows from checkpoint {ckpt_path}", flush=True)

    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    ckpt_fh = ckpt_path.open("a", encoding="utf-8")

    model_ids = [m.strip() for m in args.model_ids.split(",") if m.strip()]
    if not model_ids:
        raise ValueError("Pass at least one model via --model-ids.")
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
                    if (model_id, transcript_id, variant, turn_idx) in done_keys:
                        continue
                    prompt, span_start, span_end = _render_context(
                        turns=turns,
                        current_idx=turn_idx,
                        include_speaker_prefix=args.include_speaker_prefix,
                        max_context_turns=args.max_context_turns,
                    )
                    base_row = {
                        "model_id": model_id,
                        "transcript_id": transcript_id,
                        "topic": topic,
                        "variant": variant,
                        "layer": args.layer,
                        "speaker": turn["speaker"],
                        "turn_id": turn_idx,
                        "text": turn["text"],
                    }

                    if args.token_level:
                        token_rows, used_fallback = _embed_turn_tokens(
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
                        for tr in token_rows:
                            row = dict(base_row)
                            row.update(tr)
                            if used_fallback:
                                row["span_fallback"] = True
                            rows.append(row)
                    else:
                        _RETRYABLE_OUTER = (
                            "connection error",
                            "namespaces failed",
                            "timeout",
                            "unavailable",
                            "try again later",
                            "submitting request",
                            "remoteexception",
                            "502",
                            "503",
                            "504",
                        )
                        _max_outer = 6
                        for _attempt in range(_max_outer):
                            try:
                                vector = _embed_turn_pooled(
                                    torch_mod=torch_mod,
                                    tokenizer=tokenizer,
                                    model=model,
                                    text=prompt,
                                    layer=args.layer,
                                    span_start=span_start,
                                    span_end=span_end,
                                    backend=active_backend,
                                    ndif_remote=args.ndif_remote,
                                    pooling=args.pooling,
                                )
                                break
                            except Exception as _exc:
                                _estr = str(_exc).lower()
                                if any(kw in _estr for kw in _RETRYABLE_OUTER) and _attempt < _max_outer - 1:
                                    _wait = 5 * (2 ** _attempt)
                                    print(f"  [outer-retry {_attempt+1}/{_max_outer-1}] sleeping {_wait}s: {_exc}", flush=True)
                                    time.sleep(_wait)
                                else:
                                    raise
                        row = dict(base_row)
                        row["vector"] = vector
                        rows.append(row)
                        ckpt_fh.write(json.dumps(row, ensure_ascii=True) + "\n")
                        ckpt_fh.flush()

    ckpt_fh.close()
    payload = {
        "metadata": {
            "model_ids": model_ids,
            "layer": args.layer,
            "backend": args.backend,
            "include_speaker_prefix": args.include_speaker_prefix,
            "credentials": credential_flags,
            "ndif_remote": args.ndif_remote,
            "pooling": args.pooling,
            "token_level": bool(args.token_level),
        },
        "turn_embeddings": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=True)


if __name__ == "__main__":
    main()
