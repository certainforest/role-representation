# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This is a course repository for **CS7180** (likely a neural networks/interpretability course) containing lab assignments and research materials focused on language model interpretability and mechanistic analysis.

## Repository Structure

- **`labs/`**: Lab assignments and experiments
  - **`week1/`**: Logit lens implementation using nnsight and NDIF
  - **`transcripts/`**: Text files containing interview transcripts (numbered 1.txt, 2.txt, 3.txt)
  - **`benchmark/`**: Empty directory, likely for future benchmark results
  - **`steering.py`**: Empty placeholder file for steering experiments

## Key Technologies

### nnsight and NDIF
The primary codebase uses **nnsight** (a library for language model interpretability) with the **NDIF (National Deep Inference Fabric)** remote inference API. This allows running large models (e.g., Llama 3.1 70B) without local GPU resources.

**Authentication Requirements:**
- `NDIF_API_KEY`: Required for remote model execution via NDIF (get from https://ndif.us/)
- `HF_TOKEN`: Hugging Face token for model access

When working with notebooks that use NDIF, these should be set as environment variables or Colab secrets.

### Core Dependencies
Install with:
```bash
pip install nnsight
pip install git+https://github.com/davidbau/logitlenskit.git#subdirectory=python
```

## Working with Jupyter Notebooks

The lab notebooks are designed to run in **Google Colab** with remote execution on NDIF. Each notebook includes a Colab badge at the top.

**Key Pattern in Notebooks:**
```python
# Always use remote=True for NDIF execution
model = LanguageModel("meta-llama/Llama-3.1-70B-Instruct", device_map="auto")

# All model interactions should use remote=True
with model.trace(prompt, remote=REMOTE):
    # Intercept model internals here
    pass
```

## Interpretability Techniques Covered

### Logit Lens
The week1 notebook demonstrates the logit lens technique, which projects hidden states at each layer to vocabulary space to visualize what the model "thinks" at intermediate layers.

**Architecture-Specific Components:**
- For Llama models: Use `model.model.norm` (RMSNorm) and `model.lm_head`
- Access hidden states via `model.model.layers[i].output[0]`
- Apply final layer normalization before projecting to vocabulary

### Key Analysis Areas
1. **Multilingual Concepts**: Models use English as internal "concept language" even when processing other languages
2. **Pun Recognition**: Tracking when dual meanings emerge across layers
3. **Representation Hijacking**: In-context examples can shift word semantics across layers (Doublespeak technique)

## Important Patterns

### Two-Pass Data Collection
When tracking specific tokens through layers, use high k values (k=50) to ensure target tokens appear in trajectories:
```python
data = collect_logit_lens(prompt, model, k=50, remote=True)
```

### Getting Values from Saved Tensors
Handle both local and remote execution:
```python
def get_value(saved):
    try:
        return saved.value
    except AttributeError:
        return saved
```

## Research Context

The labs reference several key papers:
- Wendler et al. (2024): "Do Llamas Work in English?" - Multilingual concept representations
- Yona et al. (2024): "In-Context Representation Hijacking" - Doublespeak attacks
- Belrose et al. (2023): "Tuned Lens"
- nostalgebraist's Logit Lens post

## Transcript Files

The `labs/transcripts/` directory contains interview transcripts (likely for NLP analysis tasks). These are plain text files with speaker labels and timestamps, appearing to be interviews about sports medicine research.

---

## Speaker Role Binding Research Implementation

**Status**: In Progress (Phase 1: Data Infrastructure)
**Started**: 2026-02-02
**Research Goal**: Understand how language models build and update internal representations that bind speakers to their attributes, beliefs, and statements across conversational context.

### Collaboration Setup

**Repository**: https://github.com/certainforest/role-representation.git
**Branch**: `yuchen` (ALWAYS work on this branch)
**Remote**: `collaborate` (tracking collaborate/yuchen)

**Git Commands**:
```bash
# Check current branch (should always be yuchen)
git branch

# Pull latest changes from collaborate
git pull collaborate yuchen

# Push changes to yuchen branch
git push collaborate yuchen
```

**IMPORTANT**: Never commit to main or other branches. All work stays on `yuchen` branch.

### Implementation Progress

#### Phase 1: Data Infrastructure ✓ IN PROGRESS

**1.1 Speaker-Token Mapping System** ✓ COMPLETED
- **File**: `labs/data_processing/speaker_token_mapper.py`
- **Status**: Implemented (2026-02-02)
- **Features**:
  - Token-level mapping with precise character offsets
  - Speaker span tracking across turns
  - Turn boundary detection
  - Support for labeled, unlabeled, and corrupted versions
- **Dependencies**: transformers (needs installation)

**1.2 Three Label Conditions Generator** ✓ COMPLETED
- **File**: `labs/data_processing/condition_generator.py`
- **Status**: Implemented (2026-02-02)
- **Conditions**:
  1. Labeled - Normal transcript with "Speaker: text" format
  2. Unlabeled - Pure dialogue without speaker markers
  3. Corrupted - 30% random speaker label swaps
- **Output**: Saves all three conditions + metadata to benchmark/conditions/

**1.3 Transcript 3 Manual Formatting** ⏳ PENDING
- **Action Required**: Format `labs/transcripts/3.txt` (currently single-line) into proper speaker/timestamp structure
- **Next Step**: Manual editing or auto-formatting script

#### Phase 2: Activation Collection Infrastructure ⏳ PENDING

**2.1 Activation Collector** - NOT STARTED
- **File**: `labs/experiments/activation_collector.py`
- **Target Model**: OLMo-3-7B (local) or Llama-3.1-8B (NDIF remote)
- **Storage**: HDF5 format for efficient activation storage

**2.2 Batch Collection Script** - NOT STARTED
- **File**: `labs/experiments/collect_all_activations.py`
- **Purpose**: Process all transcripts × conditions × models

#### Phase 3: Core Experiments ⏳ PENDING

All experiments pending Phase 2 completion:
- **Exp1**: PCA visualization of speaker clustering
- **Exp2**: Logit lens oscillations (entity binding)
- **Exp3**: Probing classifiers for speaker identity
- **Exp4**: Activation patching (causal intervention)
- **Exp5**: Steering vectors for speaker perspective
- **Exp6**: Turn-taking pragmatics

#### Phase 4: Analysis & Visualization ⏳ PENDING

- Statistical analysis suite
- Visualization dashboard
- Benchmark leaderboard

### Directory Structure (Current)

```
labs/
├── data_processing/          ✓ NEW
│   ├── speaker_token_mapper.py       ✓ IMPLEMENTED
│   └── condition_generator.py        ✓ IMPLEMENTED
│
├── experiments/              ⏳ CREATED (empty)
│
├── analysis/                 ⏳ CREATED (empty)
│
├── benchmark/                ⏳ CREATED
│   ├── conditions/           ✓ CREATED (empty)
│   ├── activations/          ✓ CREATED (empty)
│   └── results/              ✓ CREATED (empty)
│
├── steering.py               ⏳ PLACEHOLDER
├── transcript_to_jsonl.py    ✓ EXISTING
└── transcripts/              ✓ EXISTING (1.txt, 2.txt, 3.txt)
```

### Next Steps

1. **Install Dependencies**:
   ```bash
   pip install transformers nnsight
   pip install git+https://github.com/davidbau/logitlenskit.git#subdirectory=python
   pip install h5py scikit-learn plotly pandas
   ```

2. **Test Data Pipeline**:
   ```bash
   # Test speaker-token mapper
   python labs/data_processing/speaker_token_mapper.py \
       --input labs/transcripts/jsonl/1.jsonl \
       --model meta-llama/Llama-3.1-8B \
       --visualize

   # Generate all conditions for transcript 1
   python labs/data_processing/condition_generator.py \
       --input labs/transcripts/jsonl/1.jsonl \
       --output labs/benchmark/conditions/transcript_1/
   ```

3. **Format Transcript 3**: Manually structure single-line file

4. **Implement Activation Collector**: Begin Phase 2

### Research Questions

1. **RQ1: Representation Formation** - Do models form distinct speaker representations in activation space?
2. **RQ2: Binding Mechanisms** - How do models bind entities to their attributes?
3. **RQ3: Pragmatic Inference** - Can models use conversational structure to infer speakers?
4. **RQ4: Controllability** - Can we steer model perspective to specific speakers?

### Models to Test

- **Primary**: OLMo-3-7B (local on Mac Mini - 8GB GPU RAM)
- **Secondary**: Llama-3.1-8B/70B (NDIF remote)
- **Baseline**: GPT-2 (known to fail binding tests)
- **Additional**: Gemma, Qwen (if compute available)

### Key Insights from Planning

- Late-layer oscillations (layers 20-28) may reveal binding computation
- Speaker slots likely form in mid-layers (15-20)
- Three conditions (labeled/unlabeled/corrupted) test model's binding robustness
- Activation patching can identify causally important layers

### Timeline Estimate

- **Week 1**: Data pipeline ✓ IN PROGRESS (15-20 hrs)
- **Week 2**: PCA + Probes (20-25 hrs)
- **Week 3**: Advanced experiments (20-25 hrs)
- **Week 4**: Analysis + writeup (15-20 hrs)

**Total**: ~70-90 hours over 4 weeks
