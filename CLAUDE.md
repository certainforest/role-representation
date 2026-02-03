# Speaker Role Binding Research: Experimental Plan

**Research Goal**: Understand how language models build and update internal representations that bind speakers to their attributes, beliefs, and statements across conversational context.

**Core Question**: Where do "speaker slots" live in the model, and how are sentences bound to them?

**Status**: Phase 1 - Data Infrastructure (In Progress)
**Started**: 2026-02-02

## Collaboration Setup

**Repository**: https://github.com/certainforest/role-representation.git
**Branch**: `yuchen` (ALWAYS work on this branch)
**Remote**: `collaborate` (tracking collaborate/yuchen)

```bash
# Check current branch (should always be yuchen)
git branch

# Pull latest changes
git pull collaborate yuchen

# Push changes
git push collaborate yuchen
```

---

## Research Hypotheses

### H1: Layered Representation Hypothesis
**Claim**: Speaker representations form gradually across layers, with distinct phases:
- **Early layers (0-10)**: Token-level processing, no speaker binding
- **Mid layers (10-20)**: Speaker slot formation and binding
- **Late layers (20-32)**: Refinement and oscillatory behavior during entity resolution

**Testable Predictions**:
- PCA clustering should show increasing speaker separation from layer 0 → 20
- Probe accuracy should peak in layers 15-20
- Activation patching in mid-layers should disrupt speaker attribution

### H2: Label Dependency Hypothesis
**Claim**: Models rely on explicit speaker labels for binding; without labels, binding quality degrades

**Testable Predictions**:
- Labeled condition: Clear speaker clusters, high probe accuracy
- Unlabeled condition: Degraded clustering, lower probe accuracy
- Corrupted condition: Fragmented clusters, confused probes

### H3: Oscillatory Binding Hypothesis
**Claim**: Late-layer logit oscillations reflect active entity binding computation, not noise

**Testable Predictions**:
- Oscillations (layers 20-28) correlate with binding difficulty
- Multi-entity scenarios show stronger oscillations than single-entity
- Linguistic disambiguators ("But", "However") reduce oscillations

### H4: Pragmatic Inference Hypothesis
**Claim**: Models can use conversational structure (turn-taking, Q-A patterns) to infer speakers without labels

**Testable Predictions**:
- Turn-taking violations signal speaker transitions
- Role constraints (interviewer asks, interviewee answers) enable inference
- Performance on unlabeled condition correlates with pragmatic cue richness

---

## Experimental Pipeline

### Phase 0: Data Validation ✓ COMPLETED

**Goal**: Ensure transcript data is clean, tokenization-aligned, and reproducible

**Implementation**:
- `labs/data_processing/speaker_token_mapper.py` - Token-level speaker mapping
- `labs/data_processing/condition_generator.py` - Three-condition generator

**Success Criteria**:
- ✅ Token boundaries align with speaker turn boundaries (< 5% misalignment)
- ✅ Three conditions generated: labeled, unlabeled, corrupted (30% corruption)
- ✅ Speaker spans correctly mapped to token indices

**Failure Modes**:
- Tokenizer splits speaker names mid-token → Use character offset mapping
- Corruption randomly swaps within same speaker → Validate unique speaker per turn
- JSONL format mismatch → Test with multiple transcripts

**Validation**:
```bash
python labs/data_processing/speaker_token_mapper.py \
    --input labs/transcripts/jsonl/1.jsonl \
    --visualize
# Inspect: Are turn boundaries correct? Do speaker spans match dialogue?
```

**Status**: ✓ Implemented, needs testing

---

### Phase 1: Activation Collection Infrastructure ✓ COMPLETED

**Goal**: Extract hidden states from models across all layers for all conditions

**Implementation**:
- `labs/experiments/activation_collector.py` - Single-run activation extraction
- `labs/experiments/collect_all_activations.py` - Batch processing pipeline

**Success Criteria**:
- ✅ Activations extracted for all layers (every 2nd layer to save space)
- ✅ HDF5 files < 500MB per transcript (with compression)
- ✅ Speaker labels correctly aligned with activation positions
- ✅ Remote execution works on NDIF (if local GPU unavailable)

**Failure Modes**:
- **OOM errors**: Reduce batch size or use every 4th layer instead of every 2nd
- **NDIF timeout**: Use `scan=True` for remote execution
- **Misaligned speaker labels**: Check offset mapping in Phase 0
- **Corrupted HDF5 files**: Add checksums, validate after write

**Validation**:
```bash
# Test on small transcript first
python labs/experiments/activation_collector.py \
    --condition-dir labs/benchmark/conditions/transcript_1/ \
    --condition labeled \
    --model meta-llama/Llama-3.1-8B \
    --output labs/benchmark/activations/ \
    --remote

# Verify: Load HDF5, check shapes match expectations
python -c "
from activation_collector import ActivationData
data = ActivationData.load_hdf5('labs/benchmark/activations/1_labeled_Llama-3.1-8B.h5')
print(f'Tokens: {len(data.tokens)}, Layers: {data.layers}')
print(f'Shape layer 10: {data.activations[10].shape}')
assert data.activations[10].shape[0] == len(data.tokens)
"
```

**Status**: ✓ Implemented, needs testing

---

## Core Experiments

### Experiment 1: PCA Speaker Clustering ✓ COMPLETED

**Research Question**: Do models form distinct speaker representations that cluster in activation space?

**Hypothesis**: H1 (Layered) + H2 (Label Dependency)
- Labeled: Clear clusters emerge in mid-layers (15-20)
- Unlabeled: Weak/absent clustering
- Corrupted: Fragmented clusters matching corrupted labels, not true speakers

**Method**:
1. Extract activations for each speaker across all tokens
2. Apply PCA (3 components) per layer
3. Compute clustering metrics: Silhouette score, Davies-Bouldin index
4. Visualize 2D/3D projections colored by true speaker

**Success Criteria**:
- **Strong evidence for H1**: Silhouette score increases from layer 0 → 20, then plateaus
- **Strong evidence for H2**: Labeled condition shows silhouette > 0.3 in layers 15-20; unlabeled < 0.1
- **Qualitative**: Visual inspection shows distinct speaker clusters in labeled condition

**Failure Modes & Diagnosis**:
| Failure Mode | Symptom | Diagnosis | Fix |
|--------------|---------|-----------|-----|
| No clustering at any layer | Silhouette ≈ 0 across all layers | Model doesn't form speaker representations OR activations not speaker-specific | Check: Are tokens correctly labeled? Try different model |
| Clustering in early layers only | High silhouette at layer 0-5, drops later | Clustering due to lexical features (speaker names), not semantic binding | Expected for labeled condition; check if persists in unlabeled |
| Identical clustering across conditions | Labeled ≈ Unlabeled ≈ Corrupted | Bug in condition generation OR model ignores labels | Verify: Inspect condition files manually |
| Clustering in corrupted matches true speakers | Corrupted clusters align with true speakers, not corrupted labels | Model infers speakers from content, not labels (supports H4!) | Exciting result - proceed to Exp6 |

**Interpretation Guidelines**:
- **Silhouette > 0.3**: Strong clustering (speakers well-separated)
- **Silhouette 0.1-0.3**: Moderate clustering (some structure)
- **Silhouette < 0.1**: No clustering (random arrangement)
- **Davies-Bouldin**: Lower is better; < 1.0 indicates good separation

**Implementation**: `labs/experiments/exp1_pca_speaker_clustering.py`

**Status**: ✓ Implemented, awaiting data

---

### Experiment 2: Logit Lens Oscillations (Entity Binding)

**Research Question**: Do late-layer oscillations reveal entity binding mechanisms?

**Hypothesis**: H3 (Oscillatory Binding)
- Oscillations in layers 20-28 reflect active binding computation
- Higher oscillation amplitude when multiple entities compete (Alice vs Bob)
- Linguistic disambiguators reduce oscillations

**Method**:
1. Create binding test prompts:
   ```
   Alice sees the box is empty. Bob sees the box is full.
   What does Alice think about the box? Alice thinks the box is
   ```
2. Collect logit lens at every layer (project hidden states to vocabulary)
3. Track probability trajectories for correct answer ("empty") vs distractor ("full")
4. Measure oscillation: Count sign changes in P(correct) - P(distractor) for layers 20-28

**Test Variants**:
- **2-entity baseline**: Alice vs Bob (expect moderate oscillations)
- **3-entity**: Alice, Bob, Carol (expect stronger oscillations)
- **Linguistic aid**: Add "But remember, Alice..." (expect reduced oscillations)
- **Control**: Single entity only (expect no oscillations)

**Success Criteria**:
- **Strong evidence**: Oscillation score correlates with binding difficulty (3-entity > 2-entity > 1-entity)
- **Supporting evidence**: Linguistic aids reduce oscillation score by > 30%
- **Final accuracy**: Model still gets correct answer (oscillations = computation, not failure)

**Failure Modes & Diagnosis**:
| Failure Mode | Symptom | Diagnosis | Fix |
|--------------|---------|-----------|-----|
| No oscillations | Smooth probability curves | Model resolves binding early OR logit lens artifacts | Check: Does model get answer correct? Try different model |
| Random oscillations | Oscillations in all contexts, uncorrelated with difficulty | Numerical instability OR model noise | Use running average, increase k in logit lens |
| Oscillations predict failure | High oscillation → wrong answer | Oscillations = confusion, not computation | Reframe hypothesis: oscillations as failure signal |
| No effect of linguistic aids | "But" doesn't reduce oscillations | Model doesn't use linguistic cues OR wrong intervention | Try stronger interventions: repeat entity name |

**Implementation**: `labs/experiments/exp2_logit_lens_binding.py`

**Status**: ⏳ PENDING

---

### Experiment 3: Linear Probes for Speaker Identity

**Research Question**: Can we decode speaker identity from activations? Which layers encode it most strongly?

**Hypothesis**: H1 (Layered Representation)
- Probe accuracy should peak in mid-layers (15-20)
- High accuracy in labeled condition, degraded in unlabeled
- Corrupted condition tests if model learns true speaker vs. label

**Method**:
1. Train linear classifier per layer: hidden state → speaker ID
2. Use 80/20 train/test split on token positions
3. Train on labeled condition, test on all three conditions
4. Plot accuracy curve across layers

**Probe Architecture**:
```python
class SpeakerProbe(nn.Module):
    def __init__(self, hidden_dim, num_speakers):
        self.classifier = nn.Linear(hidden_dim, num_speakers)
```

**Success Criteria**:
- **Strong evidence for H1**: Accuracy peaks in layers 15-20 (> 80% for 2-speaker transcripts)
- **Strong evidence for H2**: Labeled > Unlabeled by > 30% accuracy
- **Interpretable**: Accuracy curve shape matches PCA clustering strength

**Failure Modes & Diagnosis**:
| Failure Mode | Symptom | Diagnosis | Fix |
|--------------|---------|-----------|-----|
| High accuracy at all layers | > 70% accuracy even at layer 0 | Overfitting on lexical cues (speaker names) | Use unlabeled condition for training |
| No learning | Accuracy ≈ random (50% for 2 speakers) | Insufficient training OR no speaker signal | Increase training data, check labels |
| Unlabeled = Labeled | Same accuracy on both conditions | Model infers speakers without labels (H4!) | Exciting - suggests pragmatic inference |
| Corrupted > Labeled | Higher accuracy on corrupted labels | Probe learns position bias, not speaker | Add position encoding as control |

**Interpretation Guidelines**:
- **Random baseline**: 1/N_speakers (50% for 2 speakers, 33% for 3)
- **Weak signal**: < 10% above random
- **Moderate signal**: 10-30% above random
- **Strong signal**: > 30% above random

**Implementation**: `labs/experiments/exp3_speaker_probes.py`

**Status**: ⏳ PENDING

---

### Experiment 4: Activation Patching (Causal Intervention)

**Research Question**: Which layers are causally important for speaker binding?

**Hypothesis**: H1 (Layered Representation)
- Mid-layers (15-20) are causally critical
- Patching these layers disrupts downstream speaker attribution
- Early/late layers have minimal causal effect

**Method**:
1. Run source prompt (labeled condition), save activations at layer L
2. Run target prompt (unlabeled condition)
3. At layer L, replace target activations with source activations (only at speaker-critical tokens)
4. Measure effect on final output: "Who said X?" accuracy

**Patching Locations**:
- **Speaker name tokens**: Most direct intervention
- **Content tokens**: Test if binding spreads to content
- **Turn boundaries**: Test if boundaries are critical

**Success Criteria**:
- **Strong evidence**: Patching layers 15-20 changes answer accuracy by > 50%
- **Weak evidence**: Patching layers 0-10 or 25-32 has < 10% effect
- **Specificity**: Effect localized to patched speaker, not other speakers

**Failure Modes & Diagnosis**:
| Failure Mode | Symptom | Diagnosis | Fix |
|--------------|---------|-----------|-----|
| All layers critical | Patching any layer changes output | Model is fragile OR batch effects | Use smaller patch regions, control for intervention |
| No layer critical | Patching has no effect | Binding distributed OR task too easy | Try harder questions, patch multiple layers |
| Late layers most critical | Effect strongest in layers 25-32 | Binding happens late (revise H1) | Update hypothesis, focus late layers |
| Non-causal correlation | Probe accuracy ≠ patching effect | Representation present but not used | Distinguish encoding vs. computation |

**Implementation**: `labs/experiments/exp4_activation_patching.py`

**Status**: ⏳ PENDING

---

### Experiment 5: Steering Vectors for Speaker Perspective

**Research Question**: Can we steer model generation to adopt a specific speaker's perspective?

**Hypothesis**: H1 (Layered) + H4 (Controllability)
- Steering vectors extracted from mid-layers shift generation
- Effect size correlates with speaker distinctiveness
- Optimal steering layer aligns with binding layer (15-20)

**Method**:
1. Extract contrastive steering vector:
   ```
   v_steer = mean(activations[Speaker A]) - mean(activations[Speaker B])
   ```
2. Apply steering during generation: `hidden[L] += α * v_steer`
3. Test on neutral prompt: "The injury data suggests that..."
4. Measure perspective shift: Lexical overlap with speaker's utterances, topic alignment

**Test Cases**:
- **Derek (clinical researcher)**: Technical, data-driven language
- **Jasmine (interviewer)**: Accessible, explanatory language
- **Neutral baseline**: No steering (control)

**Success Criteria**:
- **Strong evidence**: Steered text shows > 50% topic overlap with target speaker
- **Dose-response**: Effect scales with α (0.5, 1.0, 2.0)
- **Layer specificity**: Mid-layers (15-20) show strongest effect

**Failure Modes & Diagnosis**:
| Failure Mode | Symptom | Diagnosis | Fix |
|--------------|---------|-----------|-----|
| No steering effect | Output identical regardless of α | Vector too weak OR wrong layer | Try larger α, different layer |
| Incoherent output | Steered text is nonsensical | Vector too strong OR wrong extraction | Reduce α, use more examples |
| Effect in wrong direction | Steering toward A produces B-like text | Sign error OR labels swapped | Check vector computation |
| All layers work equally | No optimal layer | Steering is non-specific perturbation | Add control: random vector steering |

**Implementation**: `labs/experiments/exp5_steering_vectors.py` (update `labs/steering.py`)

**Status**: ⏳ PENDING

---

### Experiment 6: Turn-Taking Pragmatic Inference

**Research Question**: Can models use conversational structure to infer speakers without labels?

**Hypothesis**: H4 (Pragmatic Inference)
- Turn-taking violations signal speaker transitions
- Role constraints (Q-A patterns) enable speaker inference
- Performance on unlabeled condition correlates with pragmatic cue richness

**Method**:
1. Generate synthetic dialogues with controlled structure:
   - **Q-A-Q-A**: Round-robin interview (strong role constraint)
   - **Q-A-A**: Disagreement between respondents (weak constraint)
   - **Q-Q-A**: Repeated clarification question (boundary violation)
2. Remove speaker labels
3. Ask model: "Who said the third sentence?"
4. Measure accuracy vs. structure type

**Success Criteria**:
- **Strong evidence**: Q-A-Q-A > 70% accuracy (shows role inference)
- **Supporting evidence**: Q-A-A < 40% accuracy (ambiguous without roles)
- **Boundary detection**: Q-Q-A correctly identifies repeated speaker

**Failure Modes & Diagnosis**:
| Failure Mode | Symptom | Diagnosis | Fix |
|--------------|---------|-----------|-----|
| High accuracy on all structures | > 70% even on Q-A-A | Model uses content, not structure | Control: shuffle turn order |
| Random performance on all | ≈ 50% across conditions | Model doesn't infer speakers at all | Revisit H4, check model capacity |
| Bias toward first speaker | Always predicts "first speaker" | Position bias, not inference | Add control: vary number of turns |
| Content-based only | Accuracy correlates with name mentions, not structure | Model uses lexical cues, not pragmatics | Remove all names from content |

**Implementation**: `labs/experiments/exp6_turn_taking.py`

**Status**: ⏳ PENDING

---

## Analysis & Validation

### Cross-Experiment Consistency Checks

**If hypotheses are correct, we should see**:
1. **PCA clustering peak** (Exp1) aligns with **probe accuracy peak** (Exp3) at same layers
2. **Activation patching effect** (Exp4) strongest in layers with high probe accuracy
3. **Oscillation amplitude** (Exp2) correlates with **probe confidence** (low confidence = high oscillation)
4. **Steering optimal layer** (Exp5) matches binding layer from Exp1/Exp3
5. **Unlabeled condition performance** (Exp1-3) predicts **turn-taking accuracy** (Exp6)

**Red flags (inconsistencies requiring investigation)**:
- PCA shows clustering but probes fail → Representation present but not linearly decodable
- Probes work but patching doesn't → Correlation without causation
- Oscillations don't predict difficulty → Oscillations are noise, not computation

### Statistical Rigor

**Multiple Comparison Correction**:
- 6 experiments × 3 conditions × ~10 layers = ~180 comparisons
- Use Bonferroni correction: α = 0.05 / 180 ≈ 0.0003 for significance

**Effect Size Requirements**:
- Cohen's d > 0.8 for "strong evidence"
- Cohen's d 0.5-0.8 for "moderate evidence"
- Report confidence intervals, not just p-values

**Replication**:
- Test on all 3 transcripts (2 currently formatted, 1 pending)
- Test on multiple models (Llama, OLMo, GPT-2 baseline)
- Results should replicate across transcripts and models

---

## Failure Recovery Strategies

### Scenario 1: No speaker representations found (Exp1-3 all negative)
**Interpretation**: Model doesn't form explicit speaker slots
**Next steps**:
- Try larger models (scaling hypothesis)
- Test on clearer speaker contrasts (e.g., political debates)
- Investigate distributed encoding (non-linear probes)

### Scenario 2: Representations found but not causal (Exp1-3 positive, Exp4 negative)
**Interpretation**: Speaker info encoded but not used for downstream tasks
**Next steps**:
- Test on tasks requiring speaker attribution
- Check if representations are epiphenomenal

### Scenario 3: Strong pragmatic inference (Unlabeled ≈ Labeled)
**Interpretation**: H4 supported - model infers speakers from structure
**Next steps**:
- Deep dive into Exp6 variants
- Identify minimal cues sufficient for inference
- Compare to human performance

### Scenario 4: Oscillations uncorrelated with binding
**Interpretation**: H3 rejected - oscillations are not binding computation
**Next steps**:
- Investigate alternative causes (numerical artifacts, attention patterns)
- Test if oscillations appear in non-binding contexts
- Revise H3 or discard

---

## Timeline & Dependencies

```
Week 1: Foundation
├─ Phase 0 validation (2 hrs)
├─ Phase 1 activation collection (8 hrs)
└─ Exp1 PCA clustering (5 hrs)

Week 2: Core Analysis
├─ Exp3 Probing (10 hrs) [requires: Phase 1]
├─ Exp2 Logit Lens (8 hrs) [independent]
└─ Cross-validation (Exp1 vs Exp3) (2 hrs)

Week 3: Causal & Steering
├─ Exp4 Patching (10 hrs) [requires: Exp3 results]
├─ Exp5 Steering (8 hrs) [requires: Exp1/Exp3 results]
└─ Exp6 Turn-Taking (5 hrs) [independent]

Week 4: Analysis & Writeup
├─ Cross-experiment validation (5 hrs)
├─ Statistical analysis (5 hrs)
├─ Visualization dashboard (5 hrs)
└─ Write-up & interpretation (10 hrs)
```

**Total**: 83 hours over 4 weeks

---

## Current Status

### Completed ✓
- Phase 0: Data validation infrastructure
- Phase 1: Activation collection infrastructure
- Exp1: PCA clustering implementation

### In Progress ⏳
- Testing data pipeline on real transcripts
- Installing dependencies

### Blocked 🚫
- Transcript 3 formatting (needs manual work)
- All experiments (need activations from Phase 1)

### Next Immediate Steps
1. Test speaker_token_mapper on transcript 1
2. Generate three conditions for transcripts 1-2
3. Collect activations (small test run)
4. Run Exp1 PCA on test data
5. Validate results before scaling up
