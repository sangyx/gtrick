# PyG Benchmark

## Graph Property Prediction

<table class="tg">
<thead>
  <tr>
    <th class="tg-baqh">Dataset</th>
    <th class="tg-baqh" colspan="2">ogbg-molhiv</th>
    <th class="tg-baqh" colspan="2">ogbg-ppa</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td class="tg-baqh">Trick</td>
    <td class="tg-baqh">GCN</td>
    <td class="tg-baqh">GIN</td>
    <td class="tg-baqh">GCN</td>
    <td class="tg-baqh">GIN</td>
  </tr>
  <tr>
    <td class="tg-baqh">—</td>
    <td class="tg-baqh">0.7690 ± 0.0053</td>
    <td class="tg-baqh">0.7778 ± 0.0130</td>
    <td class="tg-nrix">0.6787 ± 0.0091</td>
    <td class="tg-nrix">0.6833 ± 0.0087</td>
  </tr>
  <tr>
    <td class="tg-baqh">+Virtual Node</td>
    <td class="tg-baqh">0.7581 ± 0.0135</td>
    <td class="tg-baqh">0.7713 ± 0.0036</td>
    <td class="tg-nrix">0.6747 ± 0.0060</td>
    <td class="tg-nrix">0.6901 ± 0.0277</td>
  </tr>
  <tr>
    <td class="tg-baqh">+FLAG</td>
    <td class="tg-baqh">0.7627 ± 0.0124</td>
    <td class="tg-baqh">0.7764 ± 0.0083</td>
    <td class="tg-baqh"></td>
    <td class="tg-baqh"></td>
  </tr>
  <tr>
    <td class="tg-baqh">+Random Feature</td>
    <td class="tg-baqh">0.7743 ± 0.0134</td>
    <td class="tg-baqh">0.7692 ± 0.0065</td>
    <td class="tg-baqh"></td>
    <td class="tg-baqh"></td>
  </tr>
  <tr>
    <td class="tg-baqh">Random Forest + Fingerprint</td>
    <td class="tg-baqh" colspan="2">0.8218 ± 0.0022</td>
    <td class="tg-baqh"></td>
    <td class="tg-baqh"></td>
  </tr>
</tbody>
</table>

* *Run the baseline code: `python graph_pred.py --model gin/gcn`*

## Node Property Prediction

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

* *Run the baseline code: `python node_pred.py --model gcn/sage`*

## Link Property Prediction

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
    <td class="tg-baqh">0.4763 ± 0.0279</td>
    <td class="tg-baqh">0.5171 ± 0.0100</td>
  </tr>
  <tr>
    <td class="tg-baqh">+Common Neighbors</td>
    <td class="tg-baqh">0.5422 ± 0.0023</td>
    <td class="tg-baqh">0.5368 ± 0.0020</td>
  </tr>
  <tr>
    <td class="tg-baqh">+Resource Allocation</td>
    <td class="tg-baqh">0.5033 ± 0.0142</td>
    <td class="tg-baqh">0.5093 ± 0.0055</td>
  </tr>
  <tr>
    <td class="tg-baqh">+Adamic Adar</td>
    <td class="tg-baqh">0.5345 ± 0.0023</td>
    <td class="tg-baqh">0.5376 ± 0.0010</td>
  </tr>
  <tr>
    <td class="tg-baqh">+AnchorDistance</td>
    <td class="tg-baqh">0.4605 ± 0.0120</td>
    <td class="tg-baqh">0.4553 ± 0.0237</td>
  </tr>
</tbody>
</table>

* *Run the baseline code: `python link_pred.py --model gcn/sage`*