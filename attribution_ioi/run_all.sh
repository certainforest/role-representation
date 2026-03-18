#!/bin/bash
# Pipeline:
#   1. Run --all-types (Swap+Null+HiSwap+HiNull) for qwen + llama in parallel  →  wait
#   2. Run --all-types --reverse for qwen + llama in parallel  →  wait
#   3. Posthoc voting: all types, then reverse-only
#
# GPUs: qwen on GPU 1, llama on GPU 3 (adjust as needed)

set -e

cd "$(dirname "$0")"

# Kill all child processes on Ctrl+C or error
trap 'echo "Interrupted — killing all child processes..."; kill 0; exit 1' INT TERM EXIT

run_parallel() {
    local label="$1"; shift
    echo "======================================================================"
    echo "Launching: $label"
    echo "======================================================================"

    python run_ioi_style.py --model qwen  --gpu 1 "$@" &
    local PID_QWEN=$!
    python run_ioi_style.py --model llama --gpu 3 "$@" &
    local PID_LLAMA=$!

    echo "qwen PID=$PID_QWEN  llama PID=$PID_LLAMA"
    wait $PID_QWEN  || { echo "ERROR: qwen failed";  exit 1; }
    wait $PID_LLAMA || { echo "ERROR: llama failed"; exit 1; }
    echo "Done: $label"
}

# ── Step 1: forward patching ──────────────────────────────────────────────────
run_parallel "all-types (forward)" --all-types

# ── Step 2: reverse patching ──────────────────────────────────────────────────
run_parallel "all-types --reverse" --all-types --reverse

# ── Step 3: posthoc voting ────────────────────────────────────────────────────
echo "======================================================================"
echo "Posthoc voting..."
echo "======================================================================"

for MODEL in qwen llama; do
    JSON="results/$MODEL/F_raw_top_heads.json"
    echo "--- $MODEL: all types ---"
    python run_ioi_posthoc.py "$JSON"

    echo "--- $MODEL: reverse only ---"
    python run_ioi_posthoc.py "$JSON" --type reverse_
done

echo "======================================================================"
echo "All done."
echo "======================================================================"

trap - INT TERM EXIT  # clear trap on clean exit
