# PyG Benchmark

## Graph Property Prediction

run the baseline code:
```bash
python graph_pred.py --model gin/gcn
```

### ogbg-molhiv

|     Trick     |       GCN       |       GIN       |
|:-------------:|:---------------:|:---------------:|
|       —       | 0.7690 ± 0.0053 | 0.7778 ± 0.0130 |
| +Virtual Node | 0.7581 ± 0.0135 | 0.7713 ± 0.0036 |
|     +FLAG     | 0.7627 ± 0.0124 | 0.7764 ± 0.0083 |

* Random Forest + Fingerprint: 0.8218 ± 0.0022


### ogbg-ppa

|     Trick     |       GCN       |       GIN       |
|:-------------:|:---------------:|:---------------:|
|       —       | 0.6787 ± 0.0091 | 0.6833 ± 0.0087 |
| +Virtual Node | 0.6747 ± 0.0060 | 0.6901 ± 0.0277 |

## Node Property Prediction

run the baseline code:
```bash
python node_pred.py --model gcn/sage
```

### ogbn-arxiv

|     Trick     |       GCN       |       SAGE      |
|:-------------:|:---------------:|:---------------:|
|       —       | 0.7152 ± 0.0024 | 0.7153 ± 0.0028 |
|     +FLAG     | 0.7187 ± 0.0020 | 0.7206 ± 0.0013 |