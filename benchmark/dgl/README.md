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
|     +FLAG     | 0.7615 ± 0.0126 | 0.7582 ± 0.0160 |