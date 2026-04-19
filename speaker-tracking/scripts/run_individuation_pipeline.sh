#!/usr/bin/env bash
# Run full individuation pipeline for one or more experiment conditions.
# Usage: bash speaker-tracking/scripts/run_individuation_pipeline.sh <exp1> [exp2] ...
# Example: bash speaker-tracking/scripts/run_individuation_pipeline.sh distinct_names name_collision
set -e

if [ $# -eq 0 ]; then
  echo "Usage: $0 <experiment_name> [experiment_name ...]"
  echo "Conditions: distinct_names name_collision quote_intrusion cue_corrupted"
  exit 1
fi

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

if [ -f .env ]; then
  set -a; source .env; set +a
  echo "Loaded .env"
fi

DATA="speaker-tracking/data/experiments"
mkdir -p "$DATA/dialogues" "$DATA/last_pooling/embeddings" "$DATA/last_pooling/analysis"

for EXP in "$@"; do
  DIALOGUES="$DATA/dialogues/exp_${EXP}_dialogues.json"
  EMBEDDINGS="$DATA/last_pooling/embeddings/exp_${EXP}_embeddings.json"
  PROBE="$DATA/last_pooling/analysis/exp_${EXP}_linear_probe.json"

  echo ""
  echo "================================================================"
  echo "  PIPELINE: $EXP  ($(date))"
  echo "================================================================"

  # Step 1: Generate dialogues (role-rep env, Python 3.10)
  NEED_DIALOGUES=true
  if [ -f "$DIALOGUES" ]; then
    COUNT=$(python3 -c "import json; print(len(json.load(open('$DIALOGUES')).get('dialogues',[])))" 2>/dev/null || echo 0)
    if [ "$COUNT" -ge 20 ]; then
      echo "[1/3] SKIP dialogues — $DIALOGUES has $COUNT dialogues"
      NEED_DIALOGUES=false
    else
      echo "[1/3] Resuming dialogues for $EXP ($COUNT/20 exist) ..."
    fi
  else
    echo "[1/3] Generating dialogues for $EXP ..."
  fi
  if [ "$NEED_DIALOGUES" = true ]; then
    conda run -n role-rep python speaker-tracking/scripts/mvp_generate_experiment_dialogues.py \
      --experiment "$EXP" \
      --output "$DIALOGUES" \
      --num-dialogues 20 \
      --num-turns 120 \
      --min-words-per-turn 10 \
      --max-words-per-turn 80 \
      --min-words-tolerance 8 \
      --chunk-turns 20
    echo "[1/3] DONE dialogues ($DIALOGUES)"
  fi

  # Step 2: Extract embeddings (tdyn312 env, Python 3.12 required for NDIF remote)
  if [ -f "$EMBEDDINGS" ]; then
    echo "[2/3] SKIP embeddings — $EMBEDDINGS exists"
  else
    echo "[2/3] Extracting embeddings for $EXP ..."
    conda run -n tdyn312 python speaker-tracking/scripts/mvp_extract_turn_embeddings.py \
      --dialogues "$DIALOGUES" \
      --output "$EMBEDDINGS" \
      --model-ids "meta-llama/Meta-Llama-3.1-8B-Instruct" \
      --layer 20 \
      --pooling last \
      --backend ndif --ndif-remote
    echo "[2/3] DONE embeddings ($EMBEDDINGS)"
  fi

  # Step 3: Linear probe
  if [ -f "$PROBE" ]; then
    echo "[3/3] SKIP probe — $PROBE exists"
  else
    echo "[3/3] Running linear probe for $EXP ..."
    conda run -n role-rep python speaker-tracking/scripts/mvp_linear_probe.py \
      --embeddings "$EMBEDDINGS" \
      --output "$PROBE" \
      --tasks role,variant \
      --split-mode transcript \
      --num-seeds 5 \
      --per-transcript
    echo "[3/3] DONE probe ($PROBE)"
  fi

  echo "=== COMPLETE: $EXP  ($(date)) ==="
done

echo ""
echo "All requested conditions finished."
