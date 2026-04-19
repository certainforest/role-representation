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
- `mvp_tpr_norm_eval.py`: TPR-inspired test: compare L1 vs soft-min composition using probe-score distances.
- `mvp_tpr_interaction_eval.py`: TPR-inspired interaction check via matched (speaker, topic) quadruples.
- `mvp_rank_diagnostics.py`: SVD-based effective/stable-rank growth; re-run after projecting out role direction.
- `mvp_dense_role_test.py`: role-as-dense-direction diagnostics; compare projections to random directions.
- `mvp_generate_experiment_dialogues.py`: generates experiment-specific dialogues for three paradigms (agreement pivot, similar-profile neutral, similar-profile echo-chamber polarization).

## Experiments

### Experiment 1: Agreement Pivot (`agreement_pivot`)

Same argumentative setup as the standard LLM-generated dialogues, but Alice and Bob converge to **agreement** around a configurable turn window (default turns 40–50). Three phases:

1. **Disagreement** (turns 1 to `--agreement-turn-start - 1`): firm, opposing arguments.
2. **Transition** (`--agreement-turn-start` to `--agreement-turn-end`): one speaker acknowledges a strong point, common ground emerges.
3. **Agreement** (`--agreement-turn-end + 1` onward): collaborative, building on shared conclusions.

**Question**: Does the role-direction geometry collapse, rotate, or persist after the speakers agree?

### Experiment 2: Similar Profiles, Neutral Dialogue (`similar_neutral`)

Alice and Bob share similar demographic and ideological profiles (e.g., both progressive Democrats in LA). They have a **neutral, non-argumentative** conversation about everyday topics (food, hobbies, neighborhood changes).

**Question**: When speakers are similar and the conversation is non-contentious, does a role axis still emerge? How does across-transcript similarity compare to the debate condition?

### Experiment 3: Similar Profiles, Echo-Chamber Polarization (`similar_polarize`)

Same similar profiles, but the topic is politically charged and both speakers **agree from the start**. Over the conversation they reinforce and escalate each other's views (echo-chamber dynamic).

**Question**: When speakers move in the same ideological direction, does the model still separate them by role, or does the shared stance collapse the role axis?

## Quickstart

### Recommended output layout (by layer)

Most downstream MVP scripts only need an `mvp_turn_embeddings*.json` file, but those embeddings are layer-specific. A convenient layout is:

- `speaker-tracking/data/meanpooled_layer{L}/mvp_turn_embeddings_<backend>_<tag>.json`
- `speaker-tracking/data/meanpooled_layer{L}/mvp_*_<tag>.json` for all analysis outputs

This keeps multi-layer sweeps (same model, different layers) clean and avoids overwriting outputs.

### One-command runner (layer sweeps)

`mvp_run_layer_suite.py` runs extraction + all analysis scripts and writes into `meanpooled_layer{L}/` for each requested layer:

```bash
python speaker-tracking/scripts/mvp_run_layer_suite.py \
  --dialogues speaker-tracking/data/meanpooled_layer20/mvp_dialogues.json \
  --model-id meta-llama/Meta-Llama-3.1-8B-Instruct \
  --tag llama31_8b \
  --layers 0,5,10,15,20,25,30 \
  --backend ndif \
  --ndif-remote \
  --skip-existing
```

Use `--dry-run` to print the commands that would be executed.

### Manual run (single layer)

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
  --num-seeds 5 \
  --per-transcript

python speaker-tracking/scripts/mvp_role_direction_eval.py \
  --embeddings speaker-tracking/data/mvp_turn_embeddings.json \
  --output speaker-tracking/data/mvp_role_direction_eval.json \
  --split-mode transcript \
  --num-seeds 5 \
  --num-random-directions 20 \
  --per-transcript

python speaker-tracking/scripts/mvp_role_geometry.py \
  --embeddings speaker-tracking/data/mvp_turn_embeddings.json \
  --output-json speaker-tracking/data/mvp_role_geometry.json \
  --output-dir speaker-tracking/data/geometry_plots

# TPR-inspired tests (all run on the same mvp_turn_embeddings.json)
python speaker-tracking/scripts/mvp_tpr_norm_eval.py \
  --embeddings speaker-tracking/data/mvp_turn_embeddings.json \
  --output speaker-tracking/data/mvp_tpr_norm_eval.json \
  --split-mode transcript \
  --num-seeds 5

python speaker-tracking/scripts/mvp_tpr_interaction_eval.py \
  --embeddings speaker-tracking/data/mvp_turn_embeddings.json \
  --output speaker-tracking/data/mvp_tpr_interaction_eval.json \
  --split-mode transcript \
  --num-seeds 5

python speaker-tracking/scripts/mvp_rank_diagnostics.py \
  --embeddings speaker-tracking/data/mvp_turn_embeddings.json \
  --output speaker-tracking/data/mvp_rank_diagnostics.json \
  --center

python speaker-tracking/scripts/mvp_dense_role_test.py \
  --embeddings speaker-tracking/data/mvp_turn_embeddings.json \
  --output speaker-tracking/data/mvp_dense_role_test.json \
  --num-random-directions 50
```

### Experiment dialogues

```bash
# Experiment 1: Agreement pivot (debate → agreement around turns 40-50)
python speaker-tracking/scripts/mvp_generate_experiment_dialogues.py \
  --experiment agreement_pivot \
  --output speaker-tracking/data/exp_agreement_pivot_dialogues.json \
  --num-dialogues 20 \
  --num-turns 120 \
  --agreement-turn-start 40 \
  --agreement-turn-end 50

# Experiment 2: Similar profiles, neutral conversation
python speaker-tracking/scripts/mvp_generate_experiment_dialogues.py \
  --experiment similar_neutral \
  --output speaker-tracking/data/exp_similar_neutral_dialogues.json \
  --num-dialogues 20 \
  --num-turns 120

# Experiment 3: Similar profiles, echo-chamber polarization
python speaker-tracking/scripts/mvp_generate_experiment_dialogues.py \
  --experiment similar_polarize \
  --output speaker-tracking/data/exp_similar_polarize_dialogues.json \
  --num-dialogues 20 \
  --num-turns 120

# Then run the standard pipeline on each experiment's dialogues:
for EXP in agreement_pivot similar_neutral similar_polarize; do
  python speaker-tracking/scripts/mvp_extract_turn_embeddings.py \
    --dialogues "speaker-tracking/data/exp_${EXP}_dialogues.json" \
    --output "speaker-tracking/data/exp_${EXP}_embeddings.json" \
    --model-ids "meta-llama/Meta-Llama-3.1-8B-Instruct" \
    --layer 20 \
    --backend ndif --ndif-remote

  python speaker-tracking/scripts/mvp_role_geometry.py \
    --embeddings "speaker-tracking/data/exp_${EXP}_embeddings.json" \
    --output-json "speaker-tracking/data/exp_${EXP}_geometry.json" \
    --output-dir "speaker-tracking/data/exp_${EXP}_geometry_plots" \
    --per-transcript-plots

  python speaker-tracking/scripts/mvp_linear_probe.py \
    --embeddings "speaker-tracking/data/exp_${EXP}_embeddings.json" \
    --output "speaker-tracking/data/exp_${EXP}_linear_probe.json" \
    --tasks role,variant \
    --split-mode transcript \
    --num-seeds 5 \
    --per-transcript
done
```

## Output

`mvp_results.json` includes:

- per-model mean pairwise role-vector cosine across dialogues
- per-model per-transcript role-vector norms
- per-model swap sign-flip consistency score

New analysis outputs include:

- `mvp_linear_probe.json`: per-model probe metrics + Hewitt-style control-probe metrics and selectivity (`metric_true - metric_control`), plus (optional) per-transcript breakdown (use `--per-transcript`).
- `mvp_role_direction_eval.json`: held-out direction accuracy + (optional) per-transcript breakdown (use `--per-transcript`).
- `mvp_role_geometry.json` + `geometry_plots/`: centroid/separability geometry metrics and diagnostic plots; also includes per-transcript scalar metrics.
- `mvp_tpr_norm_eval.json`: L1 vs soft-min norm-family comparison for predicting `same_joint` from probe-score distances.
- `mvp_tpr_interaction_eval.json`: fit error comparing distance-composition families vs joint-probe distance on matched quadruples.
- `mvp_rank_diagnostics.json`: rank-proxy growth over transcript prefixes, plus residual ranks after projecting out role direction.
- `mvp_dense_role_test.json`: projection stats vs random directions + cross-transcript role-direction stability.

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
