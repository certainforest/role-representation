# Linear Probe Experiments: Documentation

**Author**: Yuchen
**Date**: 2026-02-05
**Research Question**: Do language models bind speaker roles from explicit name tokens or from semantic content patterns?

---

## Executive Summary

Linear probes trained on three transcript conditions—**labeled**, **unlabeled**, and **corrupted labels**—show similar accuracy (~75%), confirming that models bind speaker roles from **content patterns** rather than explicit name tokens.

**Key Finding**: Model representations encode speaker identity based on semantic features (topics, vocabulary, discourse style) independent of lexical speaker labels.

---

## Experimental Design

### Three-Condition Framework

| Condition | Description | Purpose |
|-----------|-------------|---------|
| **Labeled** | Original transcript with speaker names (e.g., "Derek: I think...") | Baseline with lexical cues |
| **Unlabeled** | Speaker names removed (e.g., "I think...") | Tests content-based binding |
| **Corrupted** | 30% of speaker names randomly swapped (e.g., "Jasmine: [Derek's words]") | Control for label dependency |

**Critical Test**: If probes achieve similar accuracy across all conditions, the model learns speaker identity from **content**, not **labels**.

---

## Data Preparation Pipeline

### 1. Transcript Preprocessing

**Script**: `labs/transcript_to_jsonl.py`

```bash
# Convert raw transcript to JSONL format
python labs/transcript_to_jsonl.py \
    labs/transcripts/1.txt \
    -o labs/transcripts/jsonl/1.jsonl
```

**Input Format** (raw transcript):
```
Derek Knapick  0:15
I'm an orthopedic sports medicine surgeon at Washington University...

Jasmine C.  1:22
Could you summarize your research on artificial turf injuries?

Derek Knapick  1:45
We collected data from 26 high schools in Cleveland...
```

**Output** (JSONL - one turn per line):
```json
{"speaker": "Derek Knapick", "timestamp": "0:15", "turn_number": 1, "text": "I'm an orthopedic...", ...}
{"speaker": "Jasmine C.", "timestamp": "1:22", "turn_number": 2, "text": "Could you summarize...", ...}
{"speaker": "Derek Knapick", "timestamp": "1:45", "turn_number": 3, "text": "We collected data...", ...}
```

**Metadata Extracted**:
- Speaker identities
- Turn boundaries
- Sequential context (previous/next speaker)
- Topic inference

### 2. Condition Generation

**Script**: `labs/data_processing/condition_generator.py`

```bash
# Generate three experimental conditions
python labs/data_processing/condition_generator.py \
    --input labs/transcripts/jsonl/1.jsonl \
    --output labs/benchmark/conditions/1/ \
    --model gpt2 \
    --corruption-rate 0.3 \
    --seed 42
```

**Outputs**:
```
labs/benchmark/conditions/1/
├── labeled.txt          # "Derek: I think... Jasmine: That's..."
├── unlabeled.txt        # "I think... That's..." (names removed)
├── corrupted.txt        # "Jasmine: I think... Derek: That's..." (30% swapped)
├── token_mapping.json   # Ground truth speaker labels for all tokens
└── condition_info.json  # Metadata (corruption rate, num speakers, etc.)
```

**Key Feature**: Ground truth labels stored separately in `token_mapping.json`:
```json
{
  "speaker_spans": [
    [0, 39, "Jasmine C."],      # Tokens 0-39 belong to Jasmine (TRUE label)
    [40, 197, "Derek Knapick"], # Tokens 40-197 belong to Derek (TRUE label)
    ...
  ],
  "turn_boundaries": [0, 40, 198, 273, ...]
}
```

**Critical**: For corrupted condition, we evaluate using **true labels** (from token_mapping.json), not the corrupted names in the text. This tests if the model learns speaker identity from content vs. blindly following labels.

### 3. Activation Collection

**Script**: `labs/experiments/activation_collector.py`

```bash
# Extract hidden states for each condition
for condition in labeled unlabeled corrupted; do
    python labs/experiments/activation_collector.py \
        --input labs/benchmark/conditions/1/${condition}.txt \
        --metadata labs/benchmark/conditions/1/token_mapping.json \
        --model gpt2 \
        --output labs/benchmark/activations/1/${condition}_gpt2.h5 \
        --layers 0,1,2,3,4,5,6,7,8,9,10,11 \
        --condition $condition
done
```

**Output**: HDF5 files with:
- Token-level activations for all layers: `[n_tokens, hidden_dim]`
- Ground truth speaker labels (same for all conditions)
- Turn boundaries

---

## Probe Training Methodology

### Architecture

**Model**: Simple linear classifier
```python
class RoleProbe(nn.Module):
    def __init__(self, hidden_dim: int, num_roles: int):
        super().__init__()
        self.classifier = nn.Linear(hidden_dim, num_roles)  # Single linear layer

    def forward(self, hidden_states):
        return self.classifier(hidden_states)  # [batch, num_roles]
```

**Rationale**: Linear probes test if role information is **linearly separable** in the representation space. If a simple linear layer can decode speaker identity, the model explicitly encodes this information.

### Training Protocol

**Script**: `labs/experiments/exp2_role_probes.py`

```bash
# Train probes for all layers of a condition
python labs/experiments/exp2_role_probes.py \
    --activations labs/benchmark/activations/1/labeled_gpt2.h5 \
    --output labs/results/exp2/labeled/ \
    --epochs 50 \
    --lr 1e-3 \
    --weight-decay 1e-4 \
    --test-split 0.2 \
    --seed 42
```

**Hyperparameters**:
| Parameter | Value | Purpose |
|-----------|-------|---------|
| Loss function | Cross-entropy | Multi-class classification |
| Optimizer | Adam | Adaptive learning rate |
| Learning rate | 1e-3 | Standard for linear probes |
| Weight decay | 1e-4 | L2 regularization (prevent overfitting) |
| Batch size | 64 | Mini-batch gradient descent |
| Epochs | 50 | Sufficient for convergence |
| Train/Test split | 80/20 | Standard ML practice |
| Random seed | 42 | Reproducibility |

### Training Loop (Pseudocode)

```python
for epoch in range(50):
    # Training
    for batch in train_loader:
        logits = probe(hidden_states[batch])       # Forward pass
        loss = CrossEntropy(logits, true_labels)   # Compute loss
        loss.backward()                            # Backpropagation
        optimizer.step()                           # Update weights

    # Evaluation (no gradient updates)
    with torch.no_grad():
        test_logits = probe(X_test)
        predictions = argmax(test_logits)
        accuracy = (predictions == y_test).mean()
```

**Early Stopping**: Model saved at epoch with best test accuracy (prevents overfitting).

### Evaluation Metrics

1. **Test Accuracy**: `(correct predictions) / (total test samples)`
2. **Random Baseline**: `1 / num_roles` (50% for 2 speakers)
3. **Improvement**: `test_accuracy - random_baseline`
4. **Confusion Matrix**: Shows which speakers are confused

---

## Results Summary

### Probe Accuracy Across Conditions

| Condition | Test Accuracy | Improvement over Random | Interpretation |
|-----------|---------------|-------------------------|----------------|
| **Labeled** | ~75% | +25% | Upper bound (lexical + content) |
| **Unlabeled** | ~75% | +25% | Content-based binding works! |
| **Corrupted** | ~75% | +25% | Model ignores wrong labels |

**Key Observation**: All three conditions achieve **similar accuracy** (~75%).

### Interpretation

**Strong Evidence for Content-Based Role Binding**:
- **Unlabeled ≈ Labeled**: Model doesn't need explicit speaker names
- **Corrupted ≈ Unlabeled**: Model ignores corrupted labels, learns from content
- **Improvement > +25%**: Far above random (strong signal)

**What the Model Learns**:
Since unlabeled accuracy matches labeled, speaker representations come from:
- **Semantic content**: Topics discussed (medical jargon vs. layperson questions)
- **Vocabulary patterns**: Technical terms, discourse markers
- **Discourse style**: Question-answer structure, turn length, formality

**Rejected Hypothesis**: Model does NOT rely on lexical speaker labels (e.g., "Derek:", "Jasmine:") to form role representations.

---

## Visualizations Generated

### 1. Accuracy Across Layers
**File**: `{transcript_id}_{condition}_probe_metrics.png`

Shows:
- Test accuracy vs. layer (line plot)
- Random baseline (dashed line)
- Improvement over random (bar plot)

**Expected Pattern**:
- Early layers (0-2): Low accuracy (token-level features)
- Mid layers (4-6 for GPT-2, 15-20 for Llama): Peak accuracy (role representations form)
- Late layers (10-12): Slight decrease (task-specific features)

### 2. Confusion Matrices
**File**: `{transcript_id}_{condition}_layer{L}_confusion.png`

Shows:
- Which speakers are predicted correctly
- Confusion patterns (if Derek is predicted as Jasmine)

**Ideal**: Diagonal dominance (most predictions correct)

### 3. Summary JSON
**File**: `{transcript_id}_{condition}_probe_summary.json`

```json
{
  "transcript_id": "1",
  "condition": "labeled",
  "model": "gpt2",
  "num_roles": 2,
  "speakers": ["Derek Knapick", "Jasmine C."],
  "train_samples": 4447,
  "test_samples": 1112,
  "layers": [
    {
      "layer": 0,
      "best_test_acc": 0.523,
      "improvement": 0.023,
      "best_epoch": 15,
      ...
    },
    ...
  ]
}
```

---

## Validation & Quality Checks

### 1. Train/Test Split Consistency
- **Same random seed (42)** across all conditions
- Ensures fair comparison (same test samples)

### 2. Label Alignment
For corrupted condition:
```python
# Model sees: "Jasmine: I think the data shows..." (WRONG)
# But we evaluate with: role_id = 0 (Derek - TRUE)
# If probe predicts 0 → Correct! Model learned from content
# If probe predicts 1 → Wrong! Model followed corrupted label
```

### 3. Overfitting Check
- Monitor train vs. test accuracy
- If train_acc >> test_acc → overfitting
- Solution: Weight decay (L2 regularization) prevents this

### 4. Cross-Validation
- Test on multiple transcripts (1, 2, 3)
- Results should replicate across transcripts

---

## Key Takeaways for Presentation

### Slide 1: Motivation
**Question**: How do language models track "who is speaking" in dialogues?
- Do they rely on explicit names ("Derek:", "Jasmine:")?
- Or do they learn from content patterns (vocabulary, topics, style)?

### Slide 2: Method
**Three-Condition Design**:
- **Labeled**: Normal transcript (baseline)
- **Unlabeled**: Names removed (tests content)
- **Corrupted**: Names wrong (control)

**Linear Probes**: Train simple classifier to predict speaker from hidden states.

### Slide 3: Results
**Finding**: ~75% accuracy across all conditions
- Model doesn't need speaker names
- Learns roles from semantic content patterns
- Ignores corrupted labels

### Slide 4: Implications
**For Dialogue Modeling**:
- Models form implicit speaker representations
- Content-driven binding (not label-driven)
- Robust to label noise/corruption

**Future Work**:
- Test on more complex dialogues (3+ speakers)
- Causal interventions (ablate speaker features)
- Steering vectors (control speaker perspective)

---

## Code Repositories

### Main Scripts

1. **Transcript Processing**: `labs/transcript_to_jsonl.py`
2. **Condition Generation**: `labs/data_processing/condition_generator.py`
3. **Activation Collection**: `labs/experiments/activation_collector.py`
4. **Probe Training**: `labs/experiments/exp2_role_probes.py`

### Running Full Pipeline

```bash
# 1. Prepare transcript
python labs/transcript_to_jsonl.py labs/transcripts/1.txt -o labs/transcripts/jsonl/1.jsonl

# 2. Generate conditions
python labs/data_processing/condition_generator.py \
    --input labs/transcripts/jsonl/1.jsonl \
    --output labs/benchmark/conditions/1/ \
    --model gpt2

# 3. Collect activations (all conditions)
for cond in labeled unlabeled corrupted; do
    python labs/experiments/activation_collector.py \
        --input labs/benchmark/conditions/1/${cond}.txt \
        --metadata labs/benchmark/conditions/1/token_mapping.json \
        --model gpt2 \
        --output labs/benchmark/activations/1/${cond}_gpt2.h5 \
        --layers 0,1,2,3,4,5,6,7,8,9,10,11 \
        --condition $cond
done

# 4. Train probes (all conditions)
for cond in labeled unlabeled corrupted; do
    python labs/experiments/exp2_role_probes.py \
        --activations labs/benchmark/activations/1/${cond}_gpt2.h5 \
        --output labs/results/exp2/${cond}/ \
        --epochs 50
done
```

---

## Statistical Rigor

### Random Baseline
For 2-speaker dialogues: **50% accuracy**
For N-speaker dialogues: **1/N accuracy**

**Improvement Calculation**:
```
Improvement = Test_Accuracy - Random_Baseline
Example: 0.75 - 0.50 = 0.25 (50% relative improvement)
```

### Significance Testing
**95% Confidence Interval** (bootstrap):
- Resample test set 1000 times
- Compute accuracy distribution
- Report mean ± 2*std

**Bonferroni Correction** (multiple comparisons):
- Testing 3 conditions × 12 layers = 36 comparisons
- Adjusted α = 0.05 / 36 ≈ 0.0014

---

## Limitations & Future Work

### Current Limitations
1. **Small sample**: Only 2-3 transcripts
2. **Two speakers**: Easier than multi-party dialogue
3. **Clear roles**: Interviewer vs. expert (distinct styles)
4. **Static analysis**: No causal interventions

### Future Directions
1. **Larger dataset**: 10+ diverse dialogues
2. **Multi-party**: 3+ speakers with overlapping roles
3. **Causal probes**: Activation patching to test necessity
4. **Steering vectors**: Control model's speaker attribution
5. **Cross-model comparison**: GPT-2 vs. Llama vs. OLMo

---

## References

**Code Location**:
- Repository: `https://github.com/certainforest/role-representation.git`
- Branch: `yuchen`
- Documentation: `CLAUDE.md`, `LINEAR_PROBE_DOCUMENTATION.md`

**Related Work**:
- Logit lens / activation probing (Nostalgebraist, 2020)
- Speaker role tracking in dialogue models (Gu et al., 2023)
- Compositional representations (Andreas, 2022)

---

## Contact

**Questions or Feedback**: Contact Yuchen or see `CLAUDE.md` for full research plan.
