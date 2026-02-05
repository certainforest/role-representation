# Speaker Role Binding Analysis - Detailed Results

**Model:** gpt2  
**Transcript:** 1  

## Layer-wise Metrics Comparison

| Layer | Labeled<br>Silhouette | Unlabeled<br>Silhouette | Corrupted<br>Silhouette | Best Condition |
|------:|----------------------:|------------------------:|------------------------:|:---------------|
|  0    | 0.121                | 0.125                   | 0.121                   | **Unlabeled** |
|  2    | 0.141                | 0.146                   | 0.142                   | **Unlabeled** |
|  4    | 0.239                | 0.238                   | 0.238                   | **Labeled** |
|  6    | 0.238                | 0.237                   | 0.236                   | **Labeled** |
|  8    | 0.226                | 0.229                   | 0.224                   | **Unlabeled** |
| 10    | 0.208                | 0.199                   | 0.204                   | **Labeled** |

## Summary Statistics

| Metric | Labeled | Unlabeled | Corrupted |
|:-------|--------:|----------:|----------:|
| **Labeled** | | | |
| Peak Silhouette | 0.239 | | |
| Mean Silhouette | 0.195 | | |
| Std Silhouette | 0.047 | | |
| **Unlabeled** | | | |
| Peak Silhouette | 0.238 | | |
| Mean Silhouette | 0.195 | | |
| Std Silhouette | 0.045 | | |
| **Corrupted** | | | |
| Peak Silhouette | 0.238 | | |
| Mean Silhouette | 0.194 | | |
| Std Silhouette | 0.046 | | |
