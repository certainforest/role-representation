# Speaker Role Binding Research Implementation

**Status**: In Progress (Phase 1: Data Infrastructure)
**Started**: 2026-02-02
**Research Goal**: Understand how language models build and update internal representations that bind speakers to their attributes, beliefs, and statements across conversational context.

## Collaboration Setup

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

## Implementation Progress

### Phase 1: Data Infrastructure ✓ IN PROGRESS

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

### Phase 2: Activation Collection Infrastructure ✓ COMPLETED

**2.1 Activation Collector** ✓ COMPLETED
- **File**: `labs/experiments/activation_collector.py`
- **Status**: Implemented (2026-02-02)
- **Target Model**: OLMo-3-7B (local) or Llama-3.1-8B (NDIF remote)
- **Storage**: HDF5 format for efficient activation storage

**2.2 Batch Collection Script** ✓ COMPLETED
- **File**: `labs/experiments/collect_all_activations.py`
- **Status**: Implemented (2026-02-02)
- **Purpose**: Process all transcripts × conditions × models

### Phase 3: Core Experiments ⏳ IN PROGRESS

**Exp1: PCA visualization of speaker clustering** ✓ COMPLETED
- **File**: `labs/experiments/exp1_pca_speaker_clustering.py`
- **Status**: Implemented (2026-02-02)

**Exp2: Logit lens oscillations (entity binding)** ⏳ PENDING
**Exp3: Probing classifiers for speaker identity** ⏳ PENDING
**Exp4: Activation patching (causal intervention)** ⏳ PENDING
**Exp5: Steering vectors for speaker perspective** ⏳ PENDING
**Exp6: Turn-taking pragmatics** ⏳ PENDING

### Phase 4: Analysis & Visualization ⏳ PENDING

- Statistical analysis suite
- Visualization dashboard
- Benchmark leaderboard

## Directory Structure (Current)

```
labs/
├── data_processing/          ✓ NEW
│   ├── speaker_token_mapper.py       ✓ IMPLEMENTED
│   └── condition_generator.py        ✓ IMPLEMENTED
│
├── experiments/              ✓ NEW
│   ├── activation_collector.py       ✓ IMPLEMENTED
│   ├── collect_all_activations.py    ✓ IMPLEMENTED
│   └── exp1_pca_speaker_clustering.py ✓ IMPLEMENTED
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

## Next Steps

1. **Install Dependencies**:
   ```bash
   pip install transformers nnsight
   pip install git+https://github.com/davidbau/logitlenskit.git#subdirectory=python
   pip install h5py scikit-learn plotly pandas matplotlib seaborn
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

4. **Implement Remaining Experiments**: Exp2-6

## Research Questions

1. **RQ1: Representation Formation** - Do models form distinct speaker representations in activation space?
2. **RQ2: Binding Mechanisms** - How do models bind entities to their attributes?
3. **RQ3: Pragmatic Inference** - Can models use conversational structure to infer speakers?
4. **RQ4: Controllability** - Can we steer model perspective to specific speakers?

## Models to Test

- **Primary**: OLMo-3-7B (local on Mac Mini - 8GB GPU RAM)
- **Secondary**: Llama-3.1-8B/70B (NDIF remote)
- **Baseline**: GPT-2 (known to fail binding tests)
- **Additional**: Gemma, Qwen (if compute available)

## Key Insights from Planning

- Late-layer oscillations (layers 20-28) may reveal binding computation
- Speaker slots likely form in mid-layers (15-20)
- Three conditions (labeled/unlabeled/corrupted) test model's binding robustness
- Activation patching can identify causally important layers

## Timeline Estimate

- **Week 1**: Data pipeline ✓ IN PROGRESS (15-20 hrs)
- **Week 2**: PCA + Probes (20-25 hrs)
- **Week 3**: Advanced experiments (20-25 hrs)
- **Week 4**: Analysis + writeup (15-20 hrs)

**Total**: ~70-90 hours over 4 weeks
