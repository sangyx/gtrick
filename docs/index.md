---
hide:
  - navigation
---

gtrick is an easy-to-use Python package collecting tricks for Graph Neural Networks. It tests and provides powerful tricks to boost your models' performance.

Trick is all you need! ([中文简介](https://zhuanlan.zhihu.com/p/508876898>))

## Library Highlights

* **Easy-to-use**: All it takes is to add a few lines of code to apply a powerful trick, with as little changes of existing code as possible.

* **Verified Trick**: All tricks implemented in gtrick are tested on our selected datasets. Only the tricks indeed improving model's performance can be collected by gtrick.

* **Backend Free**: We provide all tricks both in [DGL](https://www.dgl.ai/) and [PyG](https://www.pyg.org/). Whatever graph learning library you use, feel free to try it.

## Installation

!!! warning 
    This is a developmental release.

You can install gtrick by pip:

```bash
pip install gtrick
```


## Usage Example

It is very easy to get start with gtrick. Suppose you have a model for node classification:

```python
from torch_geometric.nn import GCNConv
from torch_geometric.datasets import TUDataset


dataset = TUDataset(name='ENZYMES')
data = dataset[0]
model = GCNConv(dataset.num_node_features, dataset.num_classes)

out = model(data.x, data.edge_index)
```

You can enhance your GNN model with only a few lines of code:

```python
from torch_geometric.nn import GCNConv
from torch_geometric.datasets import TUDataset
from gtrick import random_feature

dataset = TUDataset(name='ENZYMES')
data = dataset[0]
model = GCNConv(dataset.num_node_features, dataset.num_classes)
h = random_feature(data.x)
out = model(h, data.edge_index)
```

## Implemented Tricks

|     Trick    | Example | Task | Reference |
|:------------:|:------------:|:------------:|:-----:|
| VirtualNode |  [DGL](https://nbviewer.org/github/sangyx/gtrick/blob/main/benchmark/dgl/VirtualNode.ipynb)<br>[PyG](https://nbviewer.org/github/sangyx/gtrick/blob/main/benchmark/pyg/VirtualNode.ipynb) | graph | [OGB Graph Property Prediction Examples](https://github.com/snap-stanford/ogb/tree/master/examples/graphproppred/mol) |
| FLAG |  [DGL](https://nbviewer.org/github/sangyx/gtrick/blob/main/benchmark/dgl/FLAG.ipynb)<br>[PyG](https://nbviewer.org/github/sangyx/gtrick/blob/main/benchmark/pyg/FLAG.ipynb) | node*<br>graph | [Robust Optimization as Data Augmentation for Large-scale Graphs](https://arxiv.org/abs/2010.09891) |
| Fingerprint |  [DGL](https://nbviewer.org/github/sangyx/gtrick/blob/main/benchmark/dgl/Fingerprint.ipynb)<br>[PyG](https://nbviewer.org/github/sangyx/gtrick/blob/main/benchmark/pyg/Fingerprint.ipynb) | molecular graph* | [Extended-Connectivity Fingerprints](https://pubs.acs.org/doi/10.1021/ci100050t) |
| Random Feature |  [DGL](https://nbviewer.org/github/sangyx/gtrick/blob/main/benchmark/dgl/RandomFeature.ipynb)<br>[PyG](https://nbviewer.org/github/sangyx/gtrick/blob/main/benchmark/pyg/RandomFeature.ipynb) | graph* | [Random Features Strengthen Graph Neural Networks](http://arxiv.org/abs/2002.03155) |
| Label Propagation |  [DGL](https://nbviewer.org/github/sangyx/gtrick/blob/main/benchmark/dgl/LabelProp.ipynb)<br>[PyG](https://nbviewer.org/github/sangyx/gtrick/blob/main/benchmark/pyg/LabelProp.ipynb) | node* | [Learning from Labeled and Unlabeled Datawith Label Propagation](http://mlg.eng.cam.ac.uk/zoubin/papers/CMU-CALD-02-107.pdf) |
| Correct & Smooth |  [DGL](https://nbviewer.org/github/sangyx/gtrick/blob/main/benchmark/dgl/C&S.ipynb)<br>[PyG](https://nbviewer.org/github/sangyx/gtrick/blob/main/benchmark/pyg/C&S.ipynb) | node* | [Combining Label Propagation And Simple Models Out-performs Graph Neural Networks](https://arxiv.org/abs/2010.13993) |
| Common Neighbors |  [DGL](https://nbviewer.org/github/sangyx/gtrick/blob/main/benchmark/dgl/EdgeFeat.ipynb)<br>[PyG](https://nbviewer.org/github/sangyx/gtrick/blob/main/benchmark/pyg/EdgeFeat.ipynb) | link* | [Link Prediction with Structural Information](https://github.com/lustoo/OGB_link_prediction/blob/main/Link%20prediction%20with%20structural%20information.pdf) |
| Resource Allocation |  [DGL](https://nbviewer.org/github/sangyx/gtrick/blob/main/benchmark/dgl/EdgeFeat.ipynb)<br>[PyG](https://nbviewer.org/github/sangyx/gtrick/blob/main/benchmark/pyg/EdgeFeat.ipynb) | link* | [Link Prediction with Structural Information](https://github.com/lustoo/OGB_link_prediction/blob/main/Link%20prediction%20with%20structural%20information.pdf) |
| Adamic Adar |  [DGL](https://nbviewer.org/github/sangyx/gtrick/blob/main/benchmark/dgl/EdgeFeat.ipynb)<br>[PyG](https://nbviewer.org/github/sangyx/gtrick/blob/main/benchmark/pyg/EdgeFeat.ipynb) | link* | [Link Prediction with Structural Information](https://github.com/lustoo/OGB_link_prediction/blob/main/Link%20prediction%20with%20structural%20information.pdf) |
| Anchor Distance |  [DGL](https://nbviewer.org/github/sangyx/gtrick/blob/main/benchmark/dgl/EdgeFeat.ipynb)<br>[PyG](https://nbviewer.org/github/sangyx/gtrick/blob/main/benchmark/pyg/EdgeFeat.ipynb) | link* | [Link Prediction with Structural Information](https://github.com/lustoo/OGB_link_prediction/blob/main/Link%20prediction%20with%20structural%20information.pdf) |

