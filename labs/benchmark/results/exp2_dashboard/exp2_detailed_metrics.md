# Linear Probe Analysis - Detailed Results

**Model:** gpt2  
**Transcript:** 1  
**Num Roles:** 2  
**Train Samples:** 406  
**Test Samples:** 101  

## Layer-wise Accuracy Comparison

| Layer | Labeled<br>Accuracy | Unlabeled<br>Accuracy | Corrupted<br>Accuracy | Best Condition | Improvement vs Random |
|------:|--------------------:|----------------------:|----------------------:|:---------------|----------------------:|
|  0    | 0.762 (76.2%)   | 0.762 (76.2%)    | 0.762 (76.2%)    | **Labeled** | +0.262 (52.5%) |
|  2    | 0.752 (75.2%)   | 0.772 (77.2%)    | 0.772 (77.2%)    | **Unlabeled** | +0.272 (54.5%) |
|  4    | 0.752 (75.2%)   | 0.782 (78.2%)    | 0.752 (75.2%)    | **Unlabeled** | +0.282 (56.4%) |
|  6    | 0.723 (72.3%)   | 0.693 (69.3%)    | 0.772 (77.2%)    | **Corrupted** | +0.272 (54.5%) |
|  8    | 0.743 (74.3%)   | 0.743 (74.3%)    | 0.792 (79.2%)    | **Corrupted** | +0.292 (58.4%) |
| 10    | 0.723 (72.3%)   | 0.703 (70.3%)    | 0.752 (75.2%)    | **Corrupted** | +0.252 (50.5%) |

## Summary Statistics

| Condition | Peak Accuracy | Mean Accuracy | Peak Layer |
|:----------|:--------------|:--------------|:-----------|
| **Labeled** | 0.762 (76.2%) | 0.743 (74.3%) | Layer 0 |
| **Unlabeled** | 0.782 (78.2%) | 0.743 (74.3%) | Layer 4 |
| **Corrupted** | 0.792 (79.2%) | 0.767 (76.7%) | Layer 8 |
