## DGL Benchmark

### Graph Property Prediction: ogbg-molhiv

run the baseline code:
```bash
python graph_pred.py --model gin/gcn
```

|     Trick     |       GCN       |       GIN       |
|:-------------:|:---------------:|:---------------:|
|       —       | 0.7683 ± 0.0107 | 0.7708 ± 0.0138 |
| +Virtual Node | 0.7330 ± 0.0293 | 0.7673 ± 0.0082 |
|     +FLAG     | 0.7588 ± 0.0098 | 0.7652 ± 0.0161 |

### Node Property Prediction: ogbn-arxiv
|     Trick     |       GCN       |       SAGE      |
|:-------------:|:---------------:|:---------------:|
|       —       | 0.7165 ± 0.0017 | 0.7157 ± 0.0025 |
|     +FLAG     | 0.7201 ± 0.0016 | 0.7189 ± 0.0017 |