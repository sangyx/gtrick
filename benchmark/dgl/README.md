# DGL Benchmark

## Graph Property Prediction

<table class="tg">
<thead>
  <tr>
    <th class="tg-baqh">Dataset</th>
    <th class="tg-baqh" colspan="2"><span style="font-weight:normal;font-style:normal">ogbg-molhiv</span></th>
    <th class="tg-baqh" colspan="2"><span style="font-weight:normal;font-style:normal">ogbg-ppa</span></th>
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
    <td class="tg-baqh">0.7683 ± 0.0107</td>
    <td class="tg-baqh">0.7708 ± 0.0138</td>
    <td class="tg-baqh"><span style="font-weight:400;font-style:normal">0.6664 ± 0.0097</span></td>
    <td class="tg-baqh"><span style="font-weight:400;font-style:normal">0.6849 ± 0.0308</span></td>
  </tr>
  <tr>
    <td class="tg-baqh">+Virtual Node</td>
    <td class="tg-baqh">0.7330 ± 0.0293</td>
    <td class="tg-baqh">0.7673 ± 0.0082</td>
    <td class="tg-baqh"><span style="font-weight:400;font-style:normal">0.6695 ± 0.0013</span></td>
    <td class="tg-baqh"><span style="font-weight:400;font-style:normal">0.7090 ± 0.0187</span></td>
  </tr>
  <tr>
    <td class="tg-baqh">+FLAG</td>
    <td class="tg-baqh">0.7588 ± 0.0098</td>
    <td class="tg-baqh">0.7652 ± 0.0161</td>
    <td class="tg-baqh"></td>
    <td class="tg-baqh"></td>
  </tr>
  <tr>
    <td class="tg-baqh">+Random Feature</td>
    <td class="tg-baqh">0.7721 ± 0.0143</td>
    <td class="tg-baqh">0.7655 ± 0.0092</td>
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
    <th class="tg-baqh" colspan="2"><span style="font-weight:normal;font-style:normal">ogbn-arxiv</span></th>
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
    <td class="tg-baqh">0.7175 ± 0.0026</td>
    <td class="tg-baqh">0.7165 ± 0.0026</td>
  </tr>
  <tr>
    <td class="tg-baqh">+FLAG</td>
    <td class="tg-baqh">0.7201 ± 0.0016</td>
    <td class="tg-baqh">0.7189 ± 0.0017</td>
  </tr>
  <tr>
    <td class="tg-baqh">+Label Propagation</td>
    <td class="tg-baqh">0.7194 ± 0.0013</td>
    <td class="tg-baqh">0.7188  ± 0.0021</td>
  </tr>
  <tr>
    <td class="tg-baqh">+Correct & Smooth</td>
    <td class="tg-baqh">0.7274 ± 0.0039</td>
    <td class="tg-baqh">0.7347 ± 0.0017</td>
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

* *Run the baseline code: `python link_pred.py --model gcn/sage`*