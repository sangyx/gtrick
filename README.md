# gtrick: Bag of Tricks for Graph Neural Networks.

gtrick is an easy-to-use Python package collecting tricks for Graph Neural Networks. It tests and provides powerful tricks to boost your models' performance.

Trick is all you need!([Chinese Introduction](https://zhuanlan.zhihu.com/p/508876898))

## Trick

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


## Installation

*Note: This is a developmental release.*

```bash
pip install gtrick
```

## Benchmark

The results listed below are implemented by PyG. You can find the results of DGL in [DGL Benchmark](benchmark/dgl/README.md).

### Graph Property Prediction

<table>
<thead>
  <tr>
    <th>Dataset</th>
    <th colspan="2">ogbg-molhiv</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td>Trick</td>
    <td>GCN</td>
    <td>GIN</td>
  </tr>
  <tr>
    <td>—</td>
    <td>0.7690 ± 0.0053</td>
    <td>0.7778 ± 0.0130</td>
  </tr>
  <tr>
    <td>+Virtual Node</td>
    <td>0.7581 ± 0.0135</td>
    <td>0.7713 ± 0.0036</td>
  </tr>
  <tr>
    <td>+FLAG</td>
    <td>0.7627 ± 0.0124</td>
    <td>0.7764 ± 0.0083</td>
  </tr>
  <tr>
    <td>+Random Feature</td>
    <td>0.7743 ± 0.0134</td>
    <td>0.7692 ± 0.0065</td>
  </tr>
  <tr>
    <td>Random Forest + Fingerprint</td>
    <td colspan="2">0.8218 ± 0.0022</td>
  </tr>
</tbody>
</table>


### Node Property Prediction

<table class="tg">
<thead>
  <tr>
    <th class="tg-baqh">Dataset</th>
    <th class="tg-baqh" colspan="2">ogbn-arxiv</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td class="tg-baqh">Trick</td>
    <td class="tg-baqh">GCN</td>
    <td class="tg-baqh">SAGE</td>
  </tr>
  <tr>
    <td class="tg-baqh">—</td>
    <td class="tg-baqh">0.7167 ± 0.0022</td>
    <td class="tg-baqh">0.7167 ± 0.0025</td>
  </tr>
  <tr>
    <td class="tg-baqh">+FLAG</td>
    <td class="tg-baqh">0.7187 ± 0.0020</td>
    <td class="tg-baqh">0.7206 ± 0.0013</td>
  </tr>
  <tr>
    <td class="tg-baqh">+Label Propagation</td>
    <td class="tg-baqh">0.7212 ± 0.0006</td>
    <td class="tg-baqh">0.7197 ± 0.0020</td>
  </tr>
  <tr>
    <td class="tg-baqh">+Correct & Smooth</td>
    <td class="tg-baqh">0.7220 ± 0.0037</td>
    <td class="tg-baqh">0.7264 ± 0.0004</td>
  </tr>
</tbody>
</table>


### Link Property Prediction

<table class="tg">
<thead>
  <tr>
    <th class="tg-baqh">Dataset</th>
    <th class="tg-baqh" colspan="2">ogbn-collab</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td class="tg-baqh">Trick</td>
    <td class="tg-baqh">GCN</td>
    <td class="tg-baqh">SAGE</td>
  </tr>
  <tr>
    <td class="tg-baqh">—</td>
    <td class="tg-baqh">0.4718 ± 0.0093</td>
    <td class="tg-baqh">0.5140 ± 0.0040</td>
  </tr>
  <tr>
    <td class="tg-baqh">+Common Neighbors</td>
    <td class="tg-baqh">0.5332 ± 0.0019</td>
    <td class="tg-baqh">0.5370 ± 0.0034</td>
  </tr>
  <tr>
    <td class="tg-baqh">+Resource Allocation</td>
    <td class="tg-baqh">0.5024 ± 0.0092</td>
    <td class="tg-baqh">0.4787 ± 0.0060</td>
  </tr>
  <tr>
    <td class="tg-baqh">+Adamic Adar</td>
    <td class="tg-baqh">0.5283 ± 0.0048</td>
    <td class="tg-baqh">0.5291 ± 0.0032</td>
  </tr>
  <tr>
    <td class="tg-baqh">+AnchorDistance</td>
    <td class="tg-baqh">0.4740 ± 0.0135</td>
    <td class="tg-baqh">0.4290 ± 0.0107</td>
  </tr>
</tbody>
</table>
