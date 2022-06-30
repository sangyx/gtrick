"""DGL Module for Label Propagation"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn

"""
The code are adapted from
https://github.com/dmlc/dgl/tree/master/examples/pytorch/label_propagation
"""

class LabelPropagation(nn.Module):
    r"""The label propagation operator from the ["Learning from Labeled and
    Unlabeled Datawith Label Propagation"](http://mlg.eng.cam.ac.uk/zoubin/papers/CMU-CALD-02-107.pdf) paper.

    This trick is helpful for **Node Level Task**.

    $$
        \mathbf{Y}^{\prime} = \alpha \cdot \mathbf{D}^{-1/2} \mathbf{A}
        \mathbf{D}^{-1/2} \mathbf{Y} + (1 - \\alpha) \mathbf{Y},
    $$

    where unlabeled data is inferred by labeled data via propagation.

    Examples:
        [LabelPropagation (DGL)](https://nbviewer.org/github/sangyx/gtrick/blob/main/benchmark/dgl/LabelProp.ipynb)

    Args:
        num_layers (int): The number of propagations.
        alpha (float): The $\alpha$ coefficient.
    """
    def __init__(self, num_layers, alpha):
        super(LabelPropagation, self).__init__()

        self.num_layers = num_layers
        self.alpha = alpha
    
    @torch.no_grad()
    def forward(self, graph, y, mask=None, edge_weight=None, post_step=lambda y: y.clamp_(0., 1.)):
        r"""
        Args:
            graph (dgl.DGLGraph): The graph.
            y (torch.Tensor): The ground-truth label information of training nodes.
            mask (torch.LongTensor or BoolTensor): A mask or index tensor denoting which nodes were used for training. 
            edge_weight (torch.Tensor, optional): The edge weights. 
            post_step (Callable[[torch.Tensor], torch.Tensor]): The post-process function.

        Returns:
            (torch.Tensor): The obtained prediction.
        """
        with graph.local_scope():
            if y.dtype == torch.long:
                y = F.one_hot(y.view(-1)).to(torch.float32)
            
            out = y
            if mask is not None:
                out = torch.zeros_like(y)
                out[mask] = y[mask]

            aggregate_fn = fn.copy_src('h', 'm')
            if edge_weight is not None:
                assert edge_weight.shape[0] == graph.number_of_edges()
                graph.edata['_edge_weight'] = edge_weight
                aggregate_fn = fn.u_mul_e('h', '_edge_weight', 'm')
            
            degs = graph.in_degrees().float().clamp(min=1)
            norm = torch.pow(degs, -0.5).to(y.device).unsqueeze(1)

            last = (1 - self.alpha) * out
            for _ in range(self.num_layers):
                # Assume the graphs to be undirected
                out = norm * out
                
                graph.ndata['h'] = out
                graph.update_all(aggregate_fn, fn.sum('m', 'h'))
                out = self.alpha * graph.ndata.pop('h')

                out = out * norm
                
                out = post_step(last + out)
            
            return out