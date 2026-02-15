# Speaker Tracking MVP Scripts

This is a minimal starting point to validate whether role/filler binding is worth pursuing.

## Scripts

- `mvp_make_dialogues.py`: samples real transcripts from MeetingBank and builds base + speaker-swapped variants.
- `mvp_generate_llm_dialogues.py`: generates long synthetic Alice/Bob transcripts directly from an LLM.
- `mvp_extract_turn_embeddings.py`: extracts turn-level embeddings from one or many models at one layer.
- `mvp_role_stability.py`: computes role-vector stability and swap sign-flip checks.
- `mvp_plot_results.py`: generates a model-comparison figure from `mvp_results.json`.
- `mvp_linear_probe.py`: runs leakage-aware linear probes (role/variant/topic) with transcript/topic grouped splits.
- `mvp_role_direction_eval.py`: evaluates held-out role-direction ("function vector"-style) behavior with shuffled/random controls.
- `mvp_role_geometry.py`: computes representation-geometry metrics and saves PCA/trajectory/heatmap figures.

## Quickstart

```bash
# One-time dependency for dataset loading
pip install datasets
# Optional for hosted NDIF backend
pip install nnsight

python speaker-tracking/scripts/mvp_make_dialogues.py \
  --output speaker-tracking/data/mvp_dialogues.json \
  --dataset-id lytang/MeetingBank-transcript \
  --split train \
  --num-dialogues 20 \
  --speaker-text-mode omit \
  --seed 42

# Alternative: generate long Alice/Bob transcripts from an LLM
export OPENAI_API_KEY="your_key_here"
python speaker-tracking/scripts/mvp_generate_llm_dialogues.py \
  --output speaker-tracking/data/mvp_dialogues.json \
  --model gpt-4o-mini \
  --num-dialogues 20 \
  --num-turns 120 \
  --min-words-per-turn 30 \
  --max-words-per-turn 80

python speaker-tracking/scripts/mvp_extract_turn_embeddings.py \
  --dialogues speaker-tracking/data/mvp_dialogues.json \
  --output speaker-tracking/data/mvp_turn_embeddings.json \
  --model-ids "allenai/OLMo-3-1025-7B,google/gemma-2-9b-it,google/gemma-3-4b-pt,meta-llama/Meta-Llama-3.1-8B-Instruct" \
  --layer 20 \
  --backend hf \
  --hf-token "$HF_TOKEN" \
  --ndif-api-key "$NDIF_API_KEY"

python speaker-tracking/scripts/mvp_role_stability.py \
  --embeddings speaker-tracking/data/mvp_turn_embeddings.json \
  --output speaker-tracking/data/mvp_results.json

python speaker-tracking/scripts/mvp_plot_results.py \
  --results speaker-tracking/data/mvp_results.json \
  --output speaker-tracking/data/mvp_results.png

python speaker-tracking/scripts/mvp_linear_probe.py \
  --embeddings speaker-tracking/data/mvp_turn_embeddings.json \
  --output speaker-tracking/data/mvp_linear_probe.json \
  --tasks role,variant,topic \
  --split-mode transcript \
  --num-seeds 5

python speaker-tracking/scripts/mvp_role_direction_eval.py \
  --embeddings speaker-tracking/data/mvp_turn_embeddings.json \
  --output speaker-tracking/data/mvp_role_direction_eval.json \
  --split-mode transcript \
  --num-seeds 5 \
  --num-random-directions 20

python speaker-tracking/scripts/mvp_role_geometry.py \
  --embeddings speaker-tracking/data/mvp_turn_embeddings.json \
  --output-json speaker-tracking/data/mvp_role_geometry.json \
  --output-dir speaker-tracking/data/geometry_plots
```

## Output

`mvp_results.json` includes:

- per-model mean pairwise role-vector cosine across dialogues
- per-model per-transcript role-vector norms
- per-model swap sign-flip consistency score

New analysis outputs include:

- `mvp_linear_probe.json`: per-model probe metrics (accuracy, balanced accuracy, optional AUC) with per-seed breakdown.
- `mvp_role_direction_eval.json`: held-out direction accuracy, swap-flip rates, label-shuffle and random-direction controls.
- `mvp_role_geometry.json` + `geometry_plots/`: centroid/separability geometry metrics and diagnostic plots.

## Transcript-style Input

By default, extraction uses transcript-style utterances only (for example `"hi bob"`), not explicit `"Alice: hi bob"` speaker tags.

- Keep default behavior for natural transcription-style prompts.
- Use `--include-speaker-prefix` only if you want the explicit tag format for ablations.

## MeetingBank Notes

`mvp_make_dialogues.py` uses `source` text from `lytang/MeetingBank-transcript`, then:

- parses speaker-attributed turns from transcript text
- keeps the two most frequent speakers per sample
- remaps them to `Alice`/`Bob` so downstream MVP scripts continue to work
- emits a `speaker_aliases` field so you can recover original names
- supports `--speaker-text-mode` for ablations:
  - `omit`: text excludes speaker IDs (default)
  - `keep`: text prepends original IDs (`Name: utterance`)
  - `anonymize`: text prepends stable `SPEAKER_1`/`SPEAKER_2`

## Credentials

`mvp_extract_turn_embeddings.py` accepts:

- `--hf-token` for gated Hugging Face model access
- `--ndif-api-key` for hosted NDIF workflows
- `--backend` to select extraction backend:
  - `hf` (default): local transformers model load/inference
  - `ndif`: hosted NDIF inference (requires `nnsight` + `NDIF_API_KEY`)

Both are optional when environment variables are already set:

- `HF_TOKEN` or `HUGGINGFACE_TOKEN`
- `NDIF_API_KEY`
