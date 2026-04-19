# Speaker Tracking — Role Probe Pipeline

Scripts for reproducing the role-probe results in §3 and Appendix C of the paper
(*In-Context Individuation*). The pipeline extracts mean-pooled turn embeddings
from open-weight LLMs and fits linear probes to decode speaker identity across
five controlled dialogue conditions.

## Scripts

### Dialogue generation

- `mvp_generate_experiment_dialogues.py`: generates the five synthetic Alice/Bob
  experiment conditions (`agreement_pivot`, `similar_neutral`, `similar_polarize`,
  `quote_intrusion`, `distinct_names`) via `gpt-4o-mini` with schema validation.
- `mvp_generate_llm_dialogues.py`: generic long-form Alice/Bob generator used
  upstream of the experiment-specific script.
- `mvp_make_dialogues.py`: samples real transcripts from the MeetingBank corpus
  and builds base + speaker-swapped variants.
- `mvp_make_noisy_dialogues.py`: perturbation utilities for the MeetingBank
  variants (ablation helpers).

### Embedding extraction

- `mvp_extract_turn_embeddings.py`: extracts turn-level residual-stream
  activations from one or many models at one layer, mean-pooled over the current
  turn's token span. Backends: local HuggingFace (`hf`) or hosted NDIF (`ndif`).

### Probing

- `mvp_linear_probe.py`: standard L2-regularized logistic-regression probe with
  transcript-grouped splits, Hewitt-style shuffled-label control, and
  per-transcript breakdowns. Produces Tables 2, 3, 4.
- `mvp_conditional_probe.py`: conditional-transfer probes that train and test on
  disjoint subsets of a single transcript. Strategies:
  `random_2_3`, `first_vs_second_half`, `pivot_window`, `pivot_first_vs_second_half`,
  `quote_noquote_vs_quote`. Produces Figure 2 (quote intrusion) and Figure 9
  (conditional transfer).

### Orchestrators

- `mvp_run_layer_suite.py`: runs extraction + linear probe for one model across
  one or more layers; standardizes output directory layout under
  `speaker-tracking/data/meanpooled_layer{L}/`.
- `run_multi_model_mean_pipeline.sh`: one-shot runner for (model × experiment)
  pairs; writes into `experiments/mean_pooling/{embeddings,analysis}/<tag>/`.
- `run_individuation_pipeline.sh`: per-experiment runner for the last-pooling
  variant of the pipeline.
- `_run_gemma3_olmo3_mean_ctx80.sh`: model-specific wrapper that pins ctx-length
  80 and backend configuration for Gemma-3-4B and OLMo-3-7B.

### Plotting

- `mvp_plot_results.py`: quick model-comparison figure from a legacy
  `mvp_results.json`. Paper figures themselves are produced in
  `speaker-tracking/notebooks/260419_paper_figures.ipynb`.

## Experiments

Each experiment is a set of 20 synthetic two-speaker dialogues (Alice/Bob),
120 turns each, generated with gpt-4o-mini. Within-transcript speakers alternate
strictly; topics are sampled per transcript and balanced across conditions.

### `distinct_names`
Clearly distinct personas (baseline). Sharp demographic/stylistic asymmetry
between speakers.

### `similar_neutral`
Demographically and ideologically similar speakers in a neutral cooperative
exchange. Minimizes content asymmetry between roles.

### `similar_polarize`
Similar profiles that begin aligned and mutually intensify a shared stance over
the conversation. Content diverges within each transcript while role assignment
stays fixed (echo-chamber dynamic).

### `agreement_pivot`
Initially adversarial conversation that converges to agreement within a
configurable pivot window (default turns 40–50):

1. **Disagreement** (turns 1 to `--agreement-turn-start − 1`): firm, opposing arguments.
2. **Transition** (`--agreement-turn-start` to `--agreement-turn-end`): one speaker acknowledges a strong point; common ground emerges.
3. **Agreement** (`--agreement-turn-end + 1` onward): collaborative, building on shared conclusions.

### `quote_intrusion`
Ordinary conversation with a subset of turns where one speaker quotes the other
verbatim inside their own turn (single-quoted excerpt ≥ 10 chars containing a
space). The quote creates a content-vs-authorship dissociation.

## Quickstart

Paper-figure reproduction flow: generate dialogues → extract embeddings →
run probes → render figures in `260419_paper_figures.ipynb`.

### One-shot (multi-model × multi-experiment)

```bash
bash speaker-tracking/scripts/run_multi_model_mean_pipeline.sh
```

Writes per-model analysis under
`speaker-tracking/data/experiments/mean_pooling/analysis/<tag>/` and embeddings
under `.../embeddings/<tag>/`.

### Manual (single experiment, single model)

```bash
# One-time setup
pip install datasets nnsight  # nnsight only needed for ndif backend

# 1. Generate experiment dialogues (repeat per experiment)
for EXP in agreement_pivot similar_neutral similar_polarize quote_intrusion distinct_names; do
  python speaker-tracking/scripts/mvp_generate_experiment_dialogues.py \
    --experiment "$EXP" \
    --output "speaker-tracking/data/experiments/dialogues/exp_${EXP}_dialogues.json" \
    --num-dialogues 20 \
    --num-turns 120
done

# 2. Extract mean-pooled turn embeddings at layer 20
TAG=llama31_8b
for EXP in agreement_pivot similar_neutral similar_polarize quote_intrusion distinct_names; do
  python speaker-tracking/scripts/mvp_extract_turn_embeddings.py \
    --dialogues "speaker-tracking/data/experiments/dialogues/exp_${EXP}_dialogues.json" \
    --output "speaker-tracking/data/experiments/mean_pooling/embeddings/${TAG}/exp_${EXP}_mean_embeddings.json" \
    --model-ids "meta-llama/Meta-Llama-3.1-8B-Instruct" \
    --layer 20 \
    --pooling mean \
    --backend ndif --ndif-remote
done

# 3. Standard role probe (Table 2)
for EXP in agreement_pivot similar_neutral similar_polarize quote_intrusion distinct_names; do
  python speaker-tracking/scripts/mvp_linear_probe.py \
    --embeddings "speaker-tracking/data/experiments/mean_pooling/embeddings/${TAG}/exp_${EXP}_mean_embeddings.json" \
    --output "speaker-tracking/data/experiments/mean_pooling/analysis/${TAG}/exp_${EXP}_linear_probe.json" \
    --tasks role,variant \
    --split-mode transcript \
    --num-seeds 5 \
    --variants base
done

# 4. Conditional probes (Tables 3, 4 + Figures 2, 9)
for EXP in agreement_pivot similar_neutral similar_polarize quote_intrusion distinct_names; do
  EMB="speaker-tracking/data/experiments/mean_pooling/embeddings/${TAG}/exp_${EXP}_mean_embeddings.json"
  ANLY="speaker-tracking/data/experiments/mean_pooling/analysis/${TAG}"
  python speaker-tracking/scripts/mvp_conditional_probe.py --embeddings "$EMB" \
    --strategy random_2_3          --output "$ANLY/exp_${EXP}_probe_random_2_3.json"
  python speaker-tracking/scripts/mvp_conditional_probe.py --embeddings "$EMB" \
    --strategy first_vs_second_half --output "$ANLY/exp_${EXP}_probe_first_vs_second.json"
done

# Experiment-specific conditional probes
python speaker-tracking/scripts/mvp_conditional_probe.py \
  --embeddings "speaker-tracking/data/experiments/mean_pooling/embeddings/${TAG}/exp_agreement_pivot_mean_embeddings.json" \
  --strategy pivot_window \
  --output "speaker-tracking/data/experiments/mean_pooling/analysis/${TAG}/exp_agreement_pivot_probe_pivot_window.json"

python speaker-tracking/scripts/mvp_conditional_probe.py \
  --embeddings "speaker-tracking/data/experiments/mean_pooling/embeddings/${TAG}/exp_agreement_pivot_mean_embeddings.json" \
  --strategy pivot_first_vs_second_half \
  --output "speaker-tracking/data/experiments/mean_pooling/analysis/${TAG}/exp_agreement_pivot_probe_pivot_halves.json"

python speaker-tracking/scripts/mvp_conditional_probe.py \
  --embeddings "speaker-tracking/data/experiments/mean_pooling/embeddings/${TAG}/exp_quote_intrusion_mean_embeddings.json" \
  --strategy quote_noquote_vs_quote \
  --output "speaker-tracking/data/experiments/mean_pooling/analysis/${TAG}/exp_quote_intrusion_probe_quote.json"

# 5. Paper figures
jupyter nbconvert --to notebook --execute \
  speaker-tracking/notebooks/260419_paper_figures.ipynb
```

### Layer sweep (single model, many layers)

`mvp_run_layer_suite.py` is a thin wrapper around the extraction + linear-probe
steps that writes into `speaker-tracking/data/meanpooled_layer{L}/` per layer:

```bash
python speaker-tracking/scripts/mvp_run_layer_suite.py \
  --dialogues speaker-tracking/data/experiments/dialogues/exp_quote_intrusion_dialogues.json \
  --model-id meta-llama/Meta-Llama-3.1-8B-Instruct \
  --tag llama31_8b \
  --layers 0,5,10,15,20,25,30 \
  --backend ndif --ndif-remote \
  --skip-existing
```

Use `--dry-run` to print the commands that would be executed.

## Output layout

```
speaker-tracking/data/experiments/
├── dialogues/
│   └── exp_<exp>_dialogues.json                # generated dialogues (base + speaker_swapped)
└── mean_pooling/
    ├── embeddings/<tag>/
    │   └── exp_<exp>_mean_embeddings.{json,ckpt.jsonl}
    └── analysis/<tag>/
        ├── exp_<exp>_linear_probe.json          # Table 2 (standard probe)
        ├── exp_<exp>_probe_random_2_3.json      # Table 3 (within-transcript)
        ├── exp_<exp>_probe_first_vs_second.json # Table 4 (temporal stability)
        ├── exp_agreement_pivot_probe_pivot_window.json
        ├── exp_agreement_pivot_probe_pivot_halves.json
        └── exp_quote_intrusion_probe_quote.json # Figure 2 (quote intrusion)
```

Each analysis JSON carries both the true metric and a Hewitt-style shuffled-label
control under `aggregate` / `aggregate_control` (standard probe) or
`metrics_by_probe` / `control_metrics_by_probe` (conditional probes). Selectivity
is `accuracy − control_accuracy`.

## Transcript-style input

Extraction uses transcript-style utterances only (for example `"hi bob"`),
not explicit `"Alice: hi bob"` speaker tags. All paper results use
`--include-speaker-prefix=false` (the default).

`base` and `speaker_swapped` variants share an identical token stream (speaker
labels are metadata only); probes are therefore trained on `--variants base`
so that paired invariance can be checked separately rather than folded into
training.

## MeetingBank input

`mvp_make_dialogues.py` uses `source` text from `lytang/MeetingBank-transcript`
and:

- parses speaker-attributed turns from transcript text
- keeps the two most frequent speakers per sample
- remaps them to `Alice`/`Bob` so downstream scripts continue to work
- emits a `speaker_aliases` field so original names can be recovered
- supports `--speaker-text-mode`:
  - `omit` (default): text excludes speaker IDs
  - `keep`: text prepends original IDs (`Name: utterance`)
  - `anonymize`: text prepends stable `SPEAKER_1`/`SPEAKER_2`

## Credentials

`mvp_extract_turn_embeddings.py` accepts:

- `--hf-token` for gated Hugging Face model access
- `--ndif-api-key` for hosted NDIF workflows
- `--backend {hf,ndif}`:
  - `hf` (default): local transformers model load/inference
  - `ndif`: hosted NDIF inference (requires `nnsight` + `NDIF_API_KEY`)

Either flag can be omitted when the corresponding environment variable is set:

- `HF_TOKEN` or `HUGGINGFACE_TOKEN`
- `NDIF_API_KEY`
