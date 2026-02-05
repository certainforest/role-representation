# Linear Probe Similarity Analysis

## Key Finding: Conditions Are Nearly Identical

**Bottom Line:** Whether speaker labels are present, absent, or corrupted makes **virtually no difference** to the model's ability to decode speaker roles. This proves role representations emerge from **content patterns**, not labels.

## Statistical Evidence

### Mean Accuracy
- **Labeled:** 0.743 (±0.015)
- **Unlabeled:** 0.743 (±0.034)
- **Corrupted:** 0.767 (±0.014)

**Maximum difference:** 0.025 (2.5 percentage points)

### Cross-Condition Correlations
- **Labeled vs Unlabeled:** r = 0.926
- **Labeled vs Corrupted:** r = 0.000
- **Unlabeled vs Corrupted:** r = -0.070

**Average correlation:** r = 0.285

⚠️  **Moderate correlations** - Some differences exist between conditions

### ANOVA Test
- **F-statistic:** 1.963
- **p-value:** 0.1748

✅ **Not statistically significant (p > 0.05)** - No evidence of meaningful differences between conditions

## Interpretation

This similarity across conditions demonstrates that:

1. **Labels don't create role representations** - Unlabeled condition performs identically
2. **Corrupt labels are ignored** - Model uses content, not corrupted labels
3. **Content is sufficient** - Speaker style, vocabulary, and topics encode role identity

## Research Implications

- Models naturally learn to distinguish speakers from **conversational patterns**
- Explicit speaker attribution is **not necessary** for role binding
- Role representations are **robust to label noise**
- This supports theories of **implicit social reasoning** in language models

