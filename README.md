# gtrick: Bag of Tricks for Graph Neural Networks

> Trick is all you need.

## Trick

|     Trick    | Example | Task | Reference |
|:------------:|:------------:|:------------:|:-----:|
| Virtual Node |  [DGL](https://nbviewer.org/github/sangyx/gtrick/blob/main/benchmark/dgl/VirtualNode.ipynb) [PyG](https://nbviewer.org/github/sangyx/gtrick/blob/main/benchmark/pyg/VirtualNode.ipynb) | graph | [Neural Message Passing for Quantum Chemistry](https://arxiv.org/pdf/1704.01212.pdf) |
| FLAG |  [DGL](https://nbviewer.org/github/sangyx/gtrick/blob/main/benchmark/dgl/FLAG.ipynb) [PyG](https://nbviewer.org/github/sangyx/gtrick/blob/main/benchmark/pyg/FLAG.ipynb) | node graph | [Robust Optimization as Data Augmentation for Large-scale Graphs](https://arxiv.org/abs/2010.09891) |

## Benchmark

The results listed below are implemented by PyG. You can find the results of DGL in [DGL Benchmark](benchmark/dgl/README.md).

### Graph Property Prediction: ogbg-molhiv

|     Trick     |       GCN       |       GIN       |
|:-------------:|:---------------:|:---------------:|
|       —       | 0.7690 ± 0.0053 | 0.7778 ± 0.0130 |
| +Virtual Node | 0.7581 ± 0.0135 | 0.7713 ± 0.0036 |
|     +FLAG     | 0.7683 ± 0.0136 | 0.7736 ± 0.0050 |