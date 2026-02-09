# Speaker Tracking MVP Scripts

This is a minimal starting point to validate whether role/filler binding is worth pursuing.

## Scripts

- `mvp_make_dialogues.py`: builds small synthetic transcript set with base + speaker-swapped variants.
- `mvp_extract_turn_embeddings.py`: extracts turn-level embeddings from one or many models at one layer.
- `mvp_role_stability.py`: computes role-vector stability and swap sign-flip checks.

## Quickstart

```bash
python speaker-tracking/scripts/mvp_make_dialogues.py \
  --output speaker-tracking/scripts/data/mvp_dialogues.json \
  --num-dialogues 20

python speaker-tracking/scripts/mvp_extract_turn_embeddings.py \
  --dialogues speaker-tracking/scripts/data/mvp_dialogues.json \
  --output speaker-tracking/scripts/data/mvp_turn_embeddings.json \
  --model-ids "allenai/OLMo-3-1025-7B,google/gemma-2-9b-it,google/gemma-3-4b-pt,meta-llama/Meta-Llama-3.1-8B-Instruct" \
  --layer 20

python speaker-tracking/scripts/mvp_role_stability.py \
  --embeddings speaker-tracking/scripts/data/mvp_turn_embeddings.json \
  --output speaker-tracking/scripts/data/mvp_results.json
```

## Output

`mvp_results.json` includes:

- per-model mean pairwise role-vector cosine across dialogues
- per-model per-transcript role-vector norms
- per-model swap sign-flip consistency score

## Transcript-style Input

By default, extraction uses transcript-style utterances only (for example `"hi bob"`), not explicit `"Alice: hi bob"` speaker tags.

- Keep default behavior for natural transcription-style prompts.
- Use `--include-speaker-prefix` only if you want the explicit tag format for ablations.
