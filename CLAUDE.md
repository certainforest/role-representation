# Speaker Role Binding Research: Experimental Plan

**Research Goal**: Understand how language models build and update internal representations that bind speakers (roles) to their utterances across conversational context.

**Core Question**: Where do "speaker slots" live in the model, and how are tokens bound to speaker roles?

**Experimental Approach**:
- Treat entire transcript as one context (single forward pass)
- **Remove speaker names from input text** (no lexical cues)
- **Turn-pooled analysis** (primary): Mean-pool activations within each turn
- Use PCA clustering and linear probes to analyze layer-wise role representations

**Key Design Decisions**:

1. **No Speaker Names in Model Input**
   - Names used **only for labeling**, never shown to model
   - Tests whether models form role representations from **semantic content and context**
   - Avoids lexical confound (clustering on "Derek" vs "Jasmine")

2. **Turn-Pooled Representations** (Standard in Research)
   - Pool activations across all tokens within a turn: `turn_repr = mean(activations[turn_start:turn_end])`
   - One representation per speaker turn
   - Cleaner signal, follows dialogue research best practices
   - Can optionally drill down to token-level later if needed

**Status**: Phase 1 - Data Infrastructure (In Progress)
**Started**: 2026-02-02
**Updated**: 2026-02-04 (clarified name removal)

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

### H1: Layered Role Representation Hypothesis
**Claim**: Speaker role representations form gradually across layers:
- **Early layers (0-10)**: Token-level processing, minimal role binding
- **Mid layers (10-20)**: Role slot formation and binding emerges
- **Late layers (20-32)**: Refined role representations

**Testable Predictions**:
- PCA clustering: Increasing role separation from layer 0 → 20
- Linear probe accuracy: Peaks in mid-layers (15-20)
- Silhouette score: Low in early layers, high in mid-layers

### H2: Content-Based Role Binding Hypothesis
**Claim**: Models bind roles based on semantic content and conversational context, not lexical labels

**Testable Predictions** (with names removed from input, turn-pooled analysis):
- Clustering/probe accuracy > 30% above random without speaker names
- Role representations emerge from content patterns (vocabulary, style, topics)
- Performance correlates with speaker distinctiveness (clearer contrasts → better clustering)

**Alternative Outcomes**:
- **If no clustering/learning**: Model requires explicit labels for role binding (H2 rejected)
- **If weak clustering (0.1-0.2 silhouette)**: Model has weak role representations (partial H2)
- **If strong clustering (>0.3 silhouette)**: Model forms robust role representations from content alone (H2 strongly supported)

**Note**: Turn-pooling is standard practice and doesn't test token-level binding. If results are interesting, can optionally explore token-level analysis later.

---

## Experimental Pipeline

### Phase 0: Data Preparation ✓ COMPLETED

**Goal**: Prepare transcript data with token-level role labels, **without speaker names in the input text**

**Key Concept**:
- Treat entire transcript as ONE context
- **Remove speaker names from input** (e.g., "Derek:", "Jasmine:")
- Use names **only for labeling** - model never sees them
- Test if model forms role representations from **content and context alone**

**Implementation**:
- `labs/data_processing/speaker_token_mapper.py` - Maps every content token to its speaker role
- Input: Raw transcript with speaker turns (e.g., "Derek: I think... Jasmine: That's interesting...")
- Processing: Extract speaker names for labels, remove from text
- Output: Clean content with role labels

**Data Format**:
```python
{
  # Model input: content only, NO speaker names
  "text": "I think the data shows... That's interesting because...",
  "tokens": ["I", "think", "the", "data", "shows", "...", "That's", "interesting", "because", "..."],

  # Turn structure (PRIMARY)
  "turns": [
    {"role_id": 0, "role_name": "Derek", "start": 0, "end": 5},   # "I think the data shows..."
    {"role_id": 1, "role_name": "Jasmine", "start": 5, "end": 10}, # "That's interesting because..."
    {"role_id": 0, "role_name": "Derek", "start": 10, "end": 15},  # Next turn...
  ],

  # Metadata
  "role_names": ["Derek", "Jasmine"],
  "n_turns": 25,
  "n_tokens": 512
}
```

**Processing Pipeline**:
```python
# 1. Tokenize entire transcript (no speaker names)
tokens = tokenizer(transcript_without_names)

# 2. Extract activations for all tokens, all layers
activations = model(tokens, output_hidden_states=True)  # [n_layers, n_tokens, hidden_dim]

# 3. Pool activations by turn (PRIMARY ANALYSIS)
turn_representations = []
for turn in turns:
    turn_acts = activations[:, turn['start']:turn['end'], :]  # [n_layers, turn_len, hidden_dim]
    turn_repr = turn_acts.mean(dim=1)  # [n_layers, hidden_dim] - mean pool across tokens
    turn_representations.append({
        'role_id': turn['role_id'],
        'representations': turn_repr  # One repr per layer
    })
```

**Success Criteria**:
- ✅ Every content token has exactly one role label
- ✅ **Speaker names completely removed** from input text (validate: no "Derek:", "Jasmine:" in tokens)
- ✅ Turn boundaries correctly identified (role changes align with actual speaker transitions)
- ✅ Role IDs are consistent across entire transcript
- ✅ Whole transcript processed as single context (no turn segmentation)

**Failure Modes**:
- Speaker names leak into input → Strict validation: assert no role_names in tokens
- Role labels misaligned with tokens → Validate character-to-token mapping after name removal
- Ambiguous turn boundaries → Use explicit turn markers in source data
- Memory issues with long transcripts → Process in chunks but maintain role labels

**Why This Design**:
- **Avoids lexical confound**: Model can't just cluster on "Derek" vs "Jasmine" tokens
- **Tests semantic binding**: Forces model to use context, content, and conversational structure
- **Cleaner hypothesis test**: Any role clustering must come from deeper understanding, not surface patterns

**Validation**:
```bash
python labs/data_processing/speaker_token_mapper.py \
    --input labs/transcripts/jsonl/1.jsonl \
    --output labs/benchmark/token_role_data/1.json \
    --remove-names \
    --visualize

# Critical checks:
# 1. No speaker names in tokens
python -c "
import json
data = json.load(open('labs/benchmark/token_role_data/1.json'))
tokens_str = ' '.join(data['tokens']).lower()
for name in data['role_names']:
    assert name.lower() not in tokens_str, f'Speaker name {name} found in tokens!'
print('✓ No speaker names in input')
"

# 2. Role transitions are clean
# 3. Every token has a role
# Inspect visualization: Are role transitions correct? Does content make sense?
```

**Status**: ✓ Implemented, needs testing

---

### Phase 1: Activation Collection Infrastructure ✓ COMPLETED

**Goal**: Extract hidden states for entire transcript context across all layers

**Key Concept**: Process entire transcript as ONE forward pass through the model, extracting activations for every token at every layer.

**Implementation**:
- `labs/experiments/activation_collector.py` - Single-transcript activation extraction
- `labs/experiments/collect_all_activations.py` - Batch processing for multiple transcripts

**Input**: Transcript with turn boundaries from Phase 0
**Output**: HDF5 file with turn-pooled representations:
```python
{
  # Raw activations (for potential token-level analysis later)
  'tokens': [N],                    # All tokens
  'layer_0': [N, hidden_dim],       # Full activations
  'layer_2': [N, hidden_dim],
  ...

  # Turn-pooled representations (PRIMARY)
  'turn_boundaries': [(0, 5), (5, 12), ...],  # Turn spans
  'turn_roles': [0, 1, 0, 1, ...],            # Role per turn
  'turn_repr_layer_0': [n_turns, hidden_dim], # Mean-pooled per turn
  'turn_repr_layer_2': [n_turns, hidden_dim],
  ...
  'turn_repr_layer_32': [n_turns, hidden_dim]
}
```

**Success Criteria**:
- ✅ Activations extracted for all layers (every 2nd layer: 0, 2, 4, ..., 32)
- ✅ HDF5 files < 500MB per transcript (with compression)
- ✅ Role labels perfectly aligned: `role_labels[i]` matches `layer_L[i]`
- ✅ Entire transcript processed as single context (preserves cross-turn dependencies)
- ✅ Remote execution works on NDIF (if local GPU unavailable)

**Failure Modes**:
- **OOM errors**: Transcript too long for single forward pass → Chunk transcript but overlap chunks
- **NDIF timeout**: Use `scan=True` for remote execution with longer timeout
- **Misaligned role labels**: Verify token count matches activation shape at every layer
- **Context truncation**: Model context window < transcript length → Note truncation point

**Validation**:
```bash
# Test on small transcript first
python labs/experiments/activation_collector.py \
    --input labs/benchmark/token_role_data/1.json \
    --model meta-llama/Llama-3.1-8B \
    --output labs/benchmark/activations/1_Llama-3.1-8B.h5 \
    --layers 0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32 \
    --remote

# Verify: Load and check alignment
python -c "
import h5py
with h5py.File('labs/benchmark/activations/1_Llama-3.1-8B.h5', 'r') as f:
    n_tokens = len(f['tokens'])
    n_roles = len(f['role_labels'])
    layer_shape = f['layer_10'].shape
    print(f'Tokens: {n_tokens}, Roles: {n_roles}, Layer 10: {layer_shape}')
    assert n_tokens == n_roles == layer_shape[0]
"
```

**Status**: ✓ Implemented, needs testing

---

## Core Experiments

### Experiment 1: PCA Role Clustering Analysis ✓ COMPLETED

**Research Question**: Do models form distinct role representations that cluster in activation space across the entire transcript context?

**Hypothesis**: H1 (Layered Role Representation) + H2 (Token-Level Binding)
- Role clusters should emerge in mid-layers (10-20)
- Clustering should include all tokens, not just speaker names
- Clustering quality increases from early to mid layers, then stabilizes

**Method** (Turn-Pooled Analysis):
1. Load turn-pooled representations from Phase 1
   ```python
   # For each layer L and each turn i:
   turn_repr[L][i] = mean(activations[L][turn_start:turn_end])
   # Result: [n_layers, n_turns, hidden_dim]
   ```

2. For each layer L:
   ```python
   # Extract turn representations at layer L
   X = turn_representations[L]  # [n_turns, hidden_dim]
   y = turn_roles  # [n_turns] - role ID for each turn

   # Apply PCA
   pca = PCA(n_components=3)
   pca_result = pca.fit_transform(X)  # [n_turns, 3]

   # Compute clustering metrics
   silhouette = silhouette_score(pca_result, y)
   db_score = davies_bouldin_score(pca_result, y)

   # Visualize (color by role)
   plt.scatter(pca_result[:, 0], pca_result[:, 1], c=y)
   ```

3. Plot metrics across layers to identify optimal role-binding layer

**Dataset Size**: For a transcript with 30 turns, you have 30 samples per layer (vs 500+ tokens in token-level)

**Visualizations**:
- **2D PCA scatter**: Colored by role (one plot per layer, show layers 0, 10, 16, 20, 28)
- **3D PCA scatter**: Interactive plot for mid-layers (16-20)
- **Layer-wise metrics**: Line plots showing silhouette/DB score vs layer

**Success Criteria**:
- **Strong evidence for H1**: Silhouette score increases from layer 0 → 16-20, then plateaus/decreases
- **Strong evidence for H2**: Clustering persists when analyzing only content tokens (exclude speaker names)
- **Qualitative**: Visual separation of role clusters in 2D/3D PCA plots for mid-layers

**Failure Modes & Diagnosis**:
| Failure Mode | Symptom | Diagnosis | Fix |
|--------------|---------|-----------|-----|
| No clustering at any layer | Silhouette ≈ 0 across all layers | Model doesn't form role representations from content | Try different model, clearer role contrasts, longer transcript |
| Clustering in early layers only | Peak at layer 0-5, drops later | Lexical/stylistic clustering only (vocabulary differences) | Check vocabulary overlap between speakers |
| No layer progression | Flat silhouette (≈0.2) across all layers | Representation doesn't evolve OR weak throughout | Try larger model, check if speakers are distinctive |
| Very small sample size | Only 10-15 turns in transcript | Clustering metrics unreliable | Need longer transcript (30+ turns recommended) |

**Note**: Since speaker names are removed from input, any clustering must come from:
- **Content differences**: What speakers talk about (topics, domains)
- **Stylistic patterns**: How speakers express ideas (vocabulary, syntax)
- **Contextual understanding**: Model tracking "who is speaking" based on conversation flow

This is a **stronger test** of role binding than including names!

**Interpretation Guidelines**:

*Clustering strength*:
- **Silhouette > 0.3**: Strong clustering (roles well-separated) - H1 supported
- **Silhouette 0.1-0.3**: Moderate clustering (some structure) - Weak H1 support
- **Silhouette < 0.1**: No clustering (random arrangement) - H1 rejected
- **Davies-Bouldin**: Lower is better; < 1.0 indicates good separation

*Layer progression* (testing H1 - Layered Representation):
- **Expected pattern**: Low → High → Plateau (early layers: 0.1, mid layers: 0.4, late layers: 0.35)
- **Peak in layers 15-20**: Strong support for H1 (role binding in mid-layers)
- **Flat across layers**: Roles not hierarchically processed (H1 rejected)
- **Peak in early layers**: Lexical/stylistic clustering only (not semantic binding)

*Cross-transcript consistency*:
- Peak layer should be consistent (±2 layers) across different transcripts
- Effect size (silhouette) should correlate across transcripts (r > 0.7)

**Analysis Outputs**:
- `labs/results/exp1/pca_scatter_layer_{L}.png` - 2D PCA for key layers (0, 10, 16, 20, 28)
- `labs/results/exp1/pca_3d_interactive.html` - Interactive 3D plot for peak layer
- `labs/results/exp1/metrics_across_layers.png` - Silhouette and DB score curves
- `labs/results/exp1/layer_progression.gif` - Animated PCA across layers
- `labs/results/exp1/summary.json` - Peak layer, scores, interpretation

**Implementation**: `labs/experiments/exp1_pca_role_clustering.py`

**Status**: ✓ Implemented, awaiting data

---

### Experiment 2: Linear Probes for Role Classification

**Research Question**: Can we linearly decode role identity from turn representations? Which layers encode role most strongly?

**Hypothesis**: H1 (Layered Role Representation)
- Probe accuracy should peak in mid-layers (15-20)
- Accuracy curve aligns with PCA clustering strength from Exp1
- High accuracy (>30% above random) indicates linearly separable role representations

**Method** (Turn-Pooled Analysis):
1. For each layer L:
   ```python
   # Load turn-pooled representations
   X = turn_representations[L]  # [n_turns, hidden_dim]
   y = turn_roles  # [n_turns] - role ID for each turn

   # Train/test split (80/20)
   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

   # Train linear probe
   probe = nn.Linear(hidden_dim, n_roles)
   optimizer = Adam(probe.parameters(), lr=1e-3, weight_decay=1e-4)

   for epoch in range(50):
       logits = probe(X_train)
       loss = CrossEntropyLoss()(logits, y_train)
       loss.backward()
       optimizer.step()

   # Evaluate
   with torch.no_grad():
       logits_test = probe(X_test)
       accuracy = (logits_test.argmax(dim=1) == y_test).float().mean()
   ```

2. Plot accuracy curve across all layers
3. Compute confidence intervals (bootstrap with 1000 iterations)
4. Compare to random baseline (1 / n_roles)

**Probe Architecture**:
```python
class RoleProbe(nn.Module):
    def __init__(self, hidden_dim, num_roles):
        self.classifier = nn.Linear(hidden_dim, num_roles)

    def forward(self, hidden_states):
        return self.classifier(hidden_states)  # [batch, num_roles]
```

**Training Details**:
- Loss: Cross-entropy
- Optimizer: Adam, lr=1e-3
- Epochs: 50 (early stopping on validation loss)
- Batch size: 256 tokens
- Regularization: L2 weight decay (1e-4)

**Success Criteria**:
- **Strong evidence for H1**: Accuracy peaks in layers 15-20 (> 85% for 2-role transcripts)
- **Interpretable**: Peak layer matches PCA silhouette peak from Exp1 (within ±2 layers)
- **Statistical significance**: Accuracy > 30% above random (with 95% CI not overlapping baseline)
- **Cross-validation**: Results replicate across multiple train/test splits (std < 5%)

**Failure Modes & Diagnosis**:
| Failure Mode | Symptom | Diagnosis | Fix |
|--------------|---------|-----------|-----|
| High accuracy at all layers | > 75% accuracy even at layer 0 | Speakers have very distinct vocabularies | Check vocabulary overlap, try more subtle role contrasts |
| No learning | Accuracy ≈ random (50% for 2 roles) | Insufficient training data OR no role signal | Need longer transcript (30+ turns), check role distinctiveness |
| Accuracy doesn't match PCA | High probe accuracy but low silhouette | Non-linear separability | Try non-linear probe (MLP with 1 hidden layer) |
| Overfitting | Train accuracy >> Test accuracy | Too few samples (small n_turns) | Use stronger regularization, cross-validation |
| Inconsistent peak layers | Exp1 peaks at L16, Exp2 peaks at L24 | Representation present but not used consistently | Investigate what differs between PCA and classification task |

**Note**: Since speaker names are removed, probe must learn from:
- **Semantic content**: What speakers talk about (topics, domains)
- **Stylistic patterns**: How speakers express ideas (vocabulary, syntax)
- **Contextual understanding**: Where in conversation flow

High accuracy is **strong evidence** for learned role representations!

**Interpretation Guidelines**:
- **Random baseline**: 1/n_roles (50% for 2 roles, 33% for 3 roles)
- **Weak signal**: < 10% above random
- **Moderate signal**: 10-30% above random
- **Strong signal**: > 30% above random

**Analysis Outputs**:
- `labs/results/exp2/probe_accuracy_by_layer.png` - Accuracy curve with 95% confidence intervals
- `labs/results/exp2/confusion_matrix_layer_{L}.png` - Confusion matrices for key layers
- `labs/results/exp2/consistency_with_exp1.png` - Overlay PCA silhouette with probe accuracy
- `labs/results/exp2/weight_analysis_layer_{L}.png` - Probe weight visualization (which dimensions matter)
- `labs/results/exp2/summary.json` - Peak layer, accuracies, interpretation

**Implementation**: `labs/experiments/exp2_role_probes.py`

**Status**: ⏳ PENDING

---

## Core Analysis Workflow

**Step 1**: Run Experiment 1 (PCA) → Identify optimal role-binding layers
**Step 2**: Run Experiment 2 (Linear Probes) → Validate findings from Exp1
**Step 3**: Cross-validate → Ensure PCA peak aligns with probe accuracy peak
**Step 4**: Analyze token-level patterns → Verify H2 (role binding across all tokens)

**Deliverables**:
- Layer-wise role representation quality metrics
- Identification of optimal role-binding layers (likely 15-20)
- Evidence for/against token-level role binding hypothesis
- Visualizations for paper/presentation

---

## Optional Extensions (Future Work)

### Experiment 3: Activation Patching (Causal Intervention)

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

**Implementation**: `labs/experiments/exp3_activation_patching.py`

**Status**: ⏳ PENDING

---

### Experiment 4: Steering Vectors for Speaker Perspective

**Research Question**: Can we steer model generation to adopt a specific speaker's perspective?

**Hypothesis**: H1 (Layered) + H3 (Controllability)
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

**Implementation**: `labs/experiments/exp4_steering_vectors.py` (update `labs/steering.py`)

**Status**: ⏳ PENDING

---

### Experiment 5: Turn-Taking Pragmatic Inference

**Research Question**: Can models use conversational structure to infer speakers without labels?

**Hypothesis**: H3 (Pragmatic Inference)
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
| Random performance on all | ≈ 50% across conditions | Model doesn't infer speakers at all | Revisit H3, check model capacity |
| Bias toward first speaker | Always predicts "first speaker" | Position bias, not inference | Add control: vary number of turns |
| Content-based only | Accuracy correlates with name mentions, not structure | Model uses lexical cues, not pragmatics | Remove all names from content |

**Implementation**: `labs/experiments/exp5_turn_taking.py`

**Status**: ⏳ PENDING

---

## Analysis & Validation

### Cross-Experiment Consistency Checks

**If hypotheses are correct, we should see**:
1. **PCA clustering peak** (Exp1) aligns with **probe accuracy peak** (Exp2) at same layers
   - Both should peak around layers 15-20
   - Silhouette score curve shape should mirror accuracy curve
2. **Token-level consistency**:
   - High probe accuracy on content tokens (not just speaker names)
   - PCA clustering persists when filtering to content words only
3. **Cross-transcript replication**:
   - Optimal layer should be consistent across transcripts (±2 layers)
   - Effect sizes should replicate (Cohen's d > 0.5)

**Red flags (inconsistencies requiring investigation)**:
- **PCA shows clustering but probes fail** → Representation present but not linearly decodable (try non-linear probes)
- **High accuracy only at speaker names** → Lexical not semantic binding (H2 unsupported)
- **Different optimal layers per transcript** → Role binding not consistent (model-dependent or transcript-dependent)
- **No layer progression** → Flat metrics across layers (role representations not layered)

### Statistical Rigor

**Multiple Comparison Correction**:
- 2 experiments × ~17 layers = ~34 primary comparisons
- Use Bonferroni correction: α = 0.05 / 34 ≈ 0.0015 for significance
- Report adjusted p-values

**Effect Size Requirements**:
- **Strong evidence**: Cohen's d > 0.8 OR silhouette > 0.3 OR accuracy > 30% above random
- **Moderate evidence**: Cohen's d 0.5-0.8 OR silhouette 0.1-0.3 OR accuracy 10-30% above random
- **Weak/no evidence**: Cohen's d < 0.5 OR silhouette < 0.1 OR accuracy < 10% above random
- Always report confidence intervals (bootstrap 95% CI)

**Replication Plan**:
- **Test on all transcripts**: Currently have 2 formatted, 1 pending
- **Test on multiple models**: Llama-3.1-8B (primary), Llama-3.1-70B (if resources allow), OLMo-7B (baseline)
- **Consistency criteria**:
  - Optimal layer within ±2 layers across transcripts/models
  - Effect size direction consistent (e.g., all show mid-layer peak)
  - Magnitude correlation > 0.7 across transcripts

---

## Failure Recovery Strategies

### Scenario 1: No role representations found (PCA and Probes both negative)
**Symptoms**: Silhouette ≈ 0 at all layers, probe accuracy ≈ random
**Interpretation**: Model doesn't form explicit role representations in hidden states
**Next steps**:
1. Check data quality: Are role labels correct? Visualize token-role mapping
2. Try larger models: Test Llama-70B or GPT-3 scale models (capacity hypothesis)
3. Test on clearer role contrasts: Use transcripts with very distinct speakers (e.g., expert vs novice, different domains)
4. Investigate non-linear encoding: Train MLP probes instead of linear
5. Check attention patterns: Analyze attention weights for cross-turn dependencies

### Scenario 2: Representations found but only at speaker names (H2 unsupported)
**Symptoms**: High clustering/probe accuracy at name tokens, drops at content tokens
**Interpretation**: Lexical binding only, no distributed role representation
**Next steps**:
1. Analyze content words only: Filter dataset to exclude first 3 tokens of each turn
2. Test on role-anonymous transcripts: Replace names with "Speaker A", "Speaker B"
3. Investigate context window effects: Do later tokens in long utterances lose role info?
4. Compare to baselines: Static embeddings (Word2Vec) - do they also cluster by speaker?

### Scenario 3: Flat metrics across layers (No layer progression)
**Symptoms**: Similar silhouette/accuracy at layer 0 and layer 20
**Interpretation**: Role representations don't follow hierarchical processing
**Next steps**:
1. Check if representation is present but stable: Look at absolute values, not progression
2. Test smaller models: GPT-2 small might show clearer progression
3. Investigate residual stream: Model may copy role info from early layers
4. Compare to position embeddings: Is role info just position-based?

### Scenario 4: Inconsistent results across transcripts
**Symptoms**: Optimal layer varies by transcript (e.g., transcript 1 peaks at layer 12, transcript 2 at layer 22)
**Interpretation**: Role binding is content-dependent or transcript-structure-dependent
**Next steps**:
1. Analyze transcript properties: Length, number of turns, role distinctiveness
2. Correlate optimal layer with transcript features: Does complexity predict binding layer?
3. Test on synthetic dialogues: Controlled structure to isolate variables
4. Consider individual differences: Different role pairs may have different binding dynamics

---

## Simplified Timeline

**Week 1: Data Pipeline** (10 hrs)
- Day 1-2: Phase 0 - Token-role mapping validation (2 hrs)
- Day 3-5: Phase 1 - Activation collection for all transcripts (8 hrs)

**Week 2: Core Experiments** (15 hrs)
- Day 1-3: Exp1 - PCA clustering analysis (8 hrs)
- Day 4-5: Exp2 - Linear probe training and evaluation (7 hrs)

**Week 3: Analysis & Validation** (10 hrs)
- Day 1-2: Cross-validation (Exp1 vs Exp2 consistency) (3 hrs)
- Day 3-4: Token-level analysis (name vs content tokens) (4 hrs)
- Day 5: Statistical testing and effect sizes (3 hrs)

**Week 4: Writeup** (10 hrs)
- Day 1-3: Results visualization and interpretation (5 hrs)
- Day 4-5: Paper/report draft (5 hrs)

**Total**: 45 hours over 4 weeks

---

## Current Status (Updated: 2026-02-04)

### Completed ✓
- Research plan updated to focus on PCA + Linear Probes
- Phase 0: Token-role mapping infrastructure
- Phase 1: Activation collection infrastructure
- Exp1: PCA clustering implementation

### In Progress ⏳
- Testing data pipeline on real transcripts
- Validating token-role alignment

### Blocked 🚫
- Transcript 3 formatting (needs manual work)
- Experiments require activations from Phase 1

### Immediate Next Steps
1. **Test Phase 0**: Run `speaker_token_mapper.py` on transcript 1
   - Verify every token gets a role label
   - Check alignment quality
2. **Run Phase 1**: Collect activations for transcript 1
   - Single forward pass for entire transcript
   - Extract all layers (0, 2, 4, ..., 32)
3. **Test Exp1**: Run PCA on collected activations
   - Generate layer-wise visualizations
   - Compute silhouette scores
4. **Test Exp2**: Train probes on one transcript
   - Verify training pipeline works
   - Check accuracy curves make sense
5. **Scale up**: Once validated, process all transcripts

### Key Decisions
- ✅ Treat entire transcript as one context (not turn-by-turn)
- ✅ Focus on PCA and Linear Probes only (defer causal interventions)
- ✅ Every token gets a role label (not just boundaries)
- ✅ Test on one transcript first before scaling
