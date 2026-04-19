#!/usr/bin/env bash
# One-shot runner: extract mean-pooled turn embeddings + probes + conditional probes
# for each (model, experiment) pair. Writes into per-model subdirs under
# speaker-tracking/data/experiments/mean_pooling/{embeddings,analysis}/<tag>/.
#
# Usage:
#   bash speaker-tracking/scripts/run_multi_model_mean_pipeline.sh [TAGS] [EXPS]
#
#   TAGS: space-separated tags to run. Default: "llama31_8b gemma2_9b gemma3_4b olmo3_7b".
#   EXPS: space-separated exp names. Default: "agreement_pivot similar_neutral similar_polarize quote_intrusion distinct_names".
#
# Requires: HF_TOKEN and NDIF_API_KEY in env (or .env loaded), python 3.12 env for ndif remote.

set -euo pipefail

# Load .env if present.
if [ -f .env ]; then
  set -a; . ./.env; set +a
fi

: "${HF_TOKEN:?HF_TOKEN must be set (or exported via .env)}"
: "${NDIF_API_KEY:?NDIF_API_KEY must be set (or exported via .env)}"

TAGS="${1:-llama31_8b gemma2_9b gemma3_4b olmo3_7b}"
EXPS="${2:-agreement_pivot similar_neutral similar_polarize quote_intrusion distinct_names}"

# Portable lookups (macOS default bash is 3.2 — no associative arrays).
model_id_for() {
  case "$1" in
    llama31_8b) echo "meta-llama/Meta-Llama-3.1-8B-Instruct" ;;
    gemma2_9b)  echo "google/gemma-2-9b-it" ;;
    gemma3_4b)  echo "google/gemma-3-4b-it" ;;
    olmo3_7b)   echo "allenai/Olmo-3-7B-Instruct" ;;
    qwen25_7b)  echo "Qwen/Qwen2.5-7B-Instruct" ;;
    qwen25_14b) echo "Qwen/Qwen2.5-14B-Instruct" ;;
    *)          echo "" ;;
  esac
}

# Layer picks at ~0.625 * num_layers (matches Llama 3.1 8B layer 20/32).
layer_for() {
  case "$1" in
    llama31_8b) echo 20 ;;  # 32 layers
    gemma2_9b)  echo 26 ;;  # 42 layers
    gemma3_4b)  echo 21 ;;  # 34 layers
    olmo3_7b)   echo 20 ;;  # 32 layers
    qwen25_7b)  echo 18 ;;  # 28 layers
    qwen25_14b) echo 30 ;;  # 48 layers
    *)          echo 20 ;;
  esac
}

MEAN_ROOT="speaker-tracking/data/experiments/mean_pooling"
DLG_ROOT="speaker-tracking/data/experiments/dialogues"

for TAG in $TAGS; do
  MID="$(model_id_for "$TAG")"
  LYR="$(layer_for "$TAG")"
  if [ -z "$MID" ]; then
    echo "skip unknown tag '$TAG' (no model id)"; continue
  fi
  EMB_DIR="$MEAN_ROOT/embeddings/$TAG"
  AN_DIR="$MEAN_ROOT/analysis/$TAG"
  mkdir -p "$EMB_DIR" "$AN_DIR"

  for EXP in $EXPS; do
    DLG="$DLG_ROOT/exp_${EXP}_dialogues.json"
    EMB="$EMB_DIR/exp_${EXP}_mean_embeddings.json"
    PROBE="$AN_DIR/exp_${EXP}_linear_probe.json"

    if [ ! -f "$DLG" ]; then
      echo "[$TAG/$EXP] missing dialogues $DLG — skip"; continue
    fi

    echo "===== [$TAG / $EXP] model=$MID layer=$LYR ====="

    if [ ! -f "$EMB" ]; then
      # Backend: default NDIF remote; override per-model via env, e.g.
      #   BACKEND_GEMMA2_9B=hf  BACKEND_OLMO3_7B=hf  bash run_multi_model_mean_pipeline.sh ...
      TAG_UP="$(echo "$TAG" | tr '[:lower:]' '[:upper:]')"
      BACKEND_VAR="BACKEND_${TAG_UP}"
      BACKEND_VAL="$(eval echo "\${$BACKEND_VAR:-${BACKEND_DEFAULT:-ndif}}")"
      if [ "$BACKEND_VAL" = "hf" ]; then
        python speaker-tracking/scripts/mvp_extract_turn_embeddings.py \
          --dialogues "$DLG" --output "$EMB" \
          --model-ids "$MID" --layer "$LYR" --pooling mean \
          --backend hf --device "${HF_DEVICE:-cuda}"
      else
        python speaker-tracking/scripts/mvp_extract_turn_embeddings.py \
          --dialogues "$DLG" --output "$EMB" \
          --model-ids "$MID" --layer "$LYR" --pooling mean \
          --backend ndif --ndif-remote
      fi
    else
      echo "skip extract (exists): $EMB"
    fi

    # Sanity: verify base/swap aren't identical (else role probe caps at 50%).
    SANITY="$AN_DIR/exp_${EXP}_variant_sanity.json"
    python speaker-tracking/scripts/mvp_variant_sanity.py \
      --embeddings "$EMB" --output "$SANITY"

    # Always overwrite probes since the --variants=base fix supersedes any prior run.
    # Variant is recorded in each JSON's metadata block.
    python speaker-tracking/scripts/mvp_linear_probe.py \
      --embeddings "$EMB" \
      --output "$PROBE" \
      --tasks role,variant \
      --split-mode transcript \
      --num-seeds 5 \
      --per-transcript \
      --variants base

    # Conditional probes — always run (cheap).
    # (a) random 2/3 per-transcript (the main ask).
    python speaker-tracking/scripts/mvp_conditional_probe.py \
      --embeddings "$EMB" --dialogues "$DLG" \
      --output "$AN_DIR/exp_${EXP}_probe_random_2_3.json" \
      --strategy random_2_3 --tasks role --num-seeds 5 \
      --variants base

    # (b) first half vs second half — applies to any experiment.
    python speaker-tracking/scripts/mvp_conditional_probe.py \
      --embeddings "$EMB" --dialogues "$DLG" \
      --output "$AN_DIR/exp_${EXP}_probe_first_vs_second.json" \
      --strategy first_vs_second_half --tasks role \
      --variants base

    # (c) experiment-specific conditional splits.
    case "$EXP" in
      agreement_pivot)
        python speaker-tracking/scripts/mvp_conditional_probe.py \
          --embeddings "$EMB" --dialogues "$DLG" \
          --output "$AN_DIR/exp_${EXP}_probe_pivot_window.json" \
          --strategy pivot_window --tasks role \
          --variants base
        python speaker-tracking/scripts/mvp_conditional_probe.py \
          --embeddings "$EMB" --dialogues "$DLG" \
          --output "$AN_DIR/exp_${EXP}_probe_pivot_halves.json" \
          --strategy pivot_first_vs_second_half --tasks role \
          --variants base
        ;;
      quote_intrusion)
        python speaker-tracking/scripts/mvp_conditional_probe.py \
          --embeddings "$EMB" --dialogues "$DLG" \
          --output "$AN_DIR/exp_${EXP}_probe_quote.json" \
          --strategy quote_noquote_vs_quote --tasks role \
          --variants base
        ;;
    esac
  done
done

echo "DONE."
