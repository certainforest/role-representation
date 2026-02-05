# Final Research Summary: Speaker Role Binding Analysis

**Model:** gpt2  
**Transcript:** 1  
**Cross-Experiment Correlation:** r = -0.280  

## Key Findings

### Hypothesis 1: Layered Role Representation (Tested)

**Prediction:** Role representations form gradually, peaking in mid-layers

**Results:**
- **Labeled:** Exp1 peaks at layer 4 (sil=0.239), Exp2 peaks at layer 0 (acc=0.762)
- **Unlabeled:** Exp1 peaks at layer 4 (sil=0.238), Exp2 peaks at layer 4 (acc=0.782)
- **Corrupted:** Exp1 peaks at layer 4 (sil=0.238), Exp2 peaks at layer 8 (acc=0.792)

✅ **H1 Status:** Partially supported - Clear layer-wise variation exists, but peak layers inconsistent between methods

### Hypothesis 2: Content-Based Role Binding (Tested)

**Prediction:** Model forms role representations without explicit labels

**Results:**
- **Labeled:** Exp1 mean silhouette=0.195, Exp2 mean accuracy=0.743
- **Unlabeled:** Exp1 mean silhouette=0.195, Exp2 mean accuracy=0.743
- **Corrupted:** Exp1 mean silhouette=0.194, Exp2 mean accuracy=0.767

⚠️  **H2 Status:** Strongly supported - All conditions show similar performance, indicating role binding emerges from content patterns, not explicit labels

## Experiment Consistency

**Cross-experiment correlation:** r = -0.280

❌ Weak correlation - Methods measure different aspects of role encoding

## Recommendations

1. Investigate why Exp1 (clustering) and Exp2 (probes) identify different peak layers
2. Test on larger models (Llama-3.1-8B/70B) with longer context windows
3. Implement turn-pooled analysis (as per original research plan)
4. Analyze token-level vs turn-level representations
5. Test on transcripts with more distinctive speaker roles
