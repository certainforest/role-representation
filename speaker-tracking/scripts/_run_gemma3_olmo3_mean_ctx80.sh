#!/usr/bin/env bash
# Temporary helper: resume mean-embedding extraction for gemma3_4b + olmo3_7b
# using NDIF remote with --max-context-turns 80 and fail-fast/continue behavior.
set -euo pipefail

cd "$(dirname "$0")/../.."
source .venv-mvp/bin/activate
if [ -f .env ]; then set -a; . ./.env; set +a; fi
: "${HF_TOKEN:?HF_TOKEN missing}"
: "${NDIF_API_KEY:?NDIF_API_KEY missing}"

EXPS="agreement_pivot similar_neutral similar_polarize quote_intrusion distinct_names"

for TAG in gemma3_4b olmo3_7b; do
  case "$TAG" in
    gemma3_4b) MID="google/gemma-3-4b-it";       LYR=21 ;;
    olmo3_7b)  MID="allenai/Olmo-3-7B-Instruct"; LYR=20 ;;
  esac
  EMB_DIR="speaker-tracking/data/experiments/mean_pooling/embeddings/$TAG"
  mkdir -p "$EMB_DIR"
  for EXP in $EXPS; do
    DLG="speaker-tracking/data/experiments/dialogues/exp_${EXP}_dialogues.json"
    OUT="$EMB_DIR/exp_${EXP}_mean_embeddings.json"
    CKPT="$EMB_DIR/exp_${EXP}_mean_embeddings.ckpt.jsonl"
    if [ -f "$OUT" ]; then
      echo "[$TAG/$EXP] skip (exists)" >&2
      continue
    fi
    echo "===== [$TAG/$EXP] model=$MID layer=$LYR max_ctx=80 =====" >&2
    python -u speaker-tracking/scripts/mvp_extract_turn_embeddings.py \
      --dialogues "$DLG" \
      --output "$OUT" \
      --model-ids "$MID" \
      --layer "$LYR" \
      --pooling mean \
      --backend ndif --ndif-remote \
      --checkpoint "$CKPT" \
      --max-context-turns 80 \
      || echo "[$TAG/$EXP] FAILED (continuing)" >&2
  done
done

echo "DONE embeddings (max_ctx=80) for gemma3_4b + olmo3_7b"
