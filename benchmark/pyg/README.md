## PyG Benchmark

### Graph Property Prediction: ogbg-molhiv

run the baseline code:
```bash
python graph_pred.py --model gin/gcn
```

|     Trick     |       GCN       |       GIN       |
|:-------------:|:---------------:|:---------------:|
|       —       | 0.7690 ± 0.0053 | 0.7778 ± 0.0130 |
| +Virtual Node | 0.7581 ± 0.0135 | 0.7713 ± 0.0036 |
|     +FLAG     | 0.7627 ± 0.0124 | 0.7764 ± 0.0083 |

### Node Property Prediction: ogbn-arxiv
|     Trick     |       GCN       |       SAGE      |
|:-------------:|:---------------:|:---------------:|
|       —       | 0.7152 ± 0.0024 | 0.7153 ± 0.0028 |
|     +FLAG     | 0.7187 ± 0.0020 | 0.7206 ± 0.0013 |