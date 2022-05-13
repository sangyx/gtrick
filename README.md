# gtrick: Bag of Tricks for Graph Neural Networks

> Trick is all you need.

## Trick

|     Trick    | Example | Task | Reference |
|:------------:|:------------:|:------------:|:-----:|
| Virtual Node |  [DGL](https://nbviewer.org/github/sangyx/gtrick/blob/main/benchmark/dgl/VirtualNode.ipynb)<br>[PyG](https://nbviewer.org/github/sangyx/gtrick/blob/main/benchmark/pyg/VirtualNode.ipynb) | graph | [OGB Graph Property Prediction Examples](https://github.com/snap-stanford/ogb/tree/master/examples/graphproppred/mol) |
| FLAG |  [DGL](https://nbviewer.org/github/sangyx/gtrick/blob/main/benchmark/dgl/FLAG.ipynb)<br>[PyG](https://nbviewer.org/github/sangyx/gtrick/blob/main/benchmark/pyg/FLAG.ipynb) | node*<br>graph | [Robust Optimization as Data Augmentation for Large-scale Graphs](https://arxiv.org/abs/2010.09891) |

## Installation

*Note: This is a developmental release.*

```bash
pip install gtrick
```

## Benchmark

The results listed below are implemented by PyG. You can find the results of DGL in [DGL Benchmark](benchmark/dgl/README.md).

### Graph Property Prediction

#### ogbg-molhiv

|     Trick     |       GCN       |       GIN       |
|:-------------:|:---------------:|:---------------:|
|       —       | 0.7690 ± 0.0053 | 0.7778 ± 0.0130 |
| +Virtual Node | 0.7581 ± 0.0135 | 0.7713 ± 0.0036 |
|     +FLAG     | 0.7627 ± 0.0124 | 0.7764 ± 0.0083 |

#### ogbg-ppa

|     Trick     |       GCN       |       GIN       |
|:-------------:|:---------------:|:---------------:|
|       —       | 0.6664 ± 0.0097 | 0.6849 ± 0.0308 |
| +Virtual Node | 0.6695 ± 0.0013 | 0.7090 ± 0.0187 |

### Node Property Prediction

#### ogbn-arxiv

|     Trick     |       GCN       |       SAGE      |
|:-------------:|:---------------:|:---------------:|
|       —       | 0.7152 ± 0.0024 | 0.7153 ± 0.0028 |
|     +FLAG     | 0.7187 ± 0.0020 | 0.7206 ± 0.0013 |