"""DGL Module for Correct & Smooth"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .label_prop import LabelPropagation

'''
The code are adapted from
https://github.com/dmlc/dgl/tree/master/examples/pytorch/correct_and_smooth
'''

class CorrectAndSmooth(nn.Module):
    r"""The correct and smooth (C&S) post-processing model from the
    ["Combining Label Propagation And Simple Models Out-performs Graph Neural
    Networks](https://arxiv.org/abs/2010.13993) paper, where soft predictions
    $\mathbf{Z}$ (obtained from a simple base predictor) are
    first corrected based on ground-truth training
    label information $\mathbf{Y}$ and residual propagation

    $$
        \mathbf{e}^{(0)}_i = \begin{cases}
            \mathbf{y}_i - \mathbf{z}_i, \text{if }i
            \text{ is training node,}\\
            \mathbf{0}, \text{else}
        \end{cases}
    $$

    $$
        \mathbf{E}^{(\ell)} = \alpha_1 \mathbf{D}^{-1/2}\mathbf{A}
        \mathbf{D}^{-1/2} \mathbf{E}^{(\ell - 1)} +
        (1 - \alpha_1) \mathbf{E}^{(\ell - 1)}
    $$

    $$ 
        \mathbf{\hat{Z}} = \mathbf{Z} + \gamma \cdot \mathbf{E}^{(L_1)} 
    $$

    where $\gamma$ denotes the scaling factor (either fixed or
    automatically determined), and then smoothed over the graph via label
    propagation

    $$
        \mathbf{\hat{z}}^{(0)}_i = \begin{cases}
            \mathbf{y}_i, \text{if }i\text{ is training node,}\\
            \mathbf{\hat{z}}_i, \text{else}
        \end{cases}
    $$

    $$
        \mathbf{\hat{Z}}^{(\ell)} = \alpha_2 \mathbf{D}^{-1/2}\mathbf{A}
        \mathbf{D}^{-1/2} \mathbf{\hat{Z}}^{(\ell - 1)} +
        (1 - \alpha_1) \mathbf{\hat{Z}}^{(\ell - 1)}
    $$

    to obtain the final prediction $\mathbf{\hat{Z}}^{(L_2)}$. 
    
    This trick is helpful for **Node Level Task**.

    Note:
        To use this trick, call `correct` at first, then call `smooth`.

    Examples: 
        [CorrectAndSmooth (DGL)](https://nbviewer.org/github/sangyx/gtrick/blob/main/benchmark/dgl/C&S.ipynb)

    Args:
        num_correction_layers (int): The number of propagations $L_1$.
        correction_alpha (float): The $\alpha_1$ coefficient.
        num_smoothing_layers (int): The number of propagations $L_2$.
        smoothing_alpha (float): The $\alpha_2$ coefficient.
        autoscale (bool, optional): If set to `True`, will automatically
            determine the scaling factor $\gamma$.
        scale (float, optional): The scaling factor $\gamma$, in case
            `autoscale = False`.
    """
    def __init__(self,
                 num_correction_layers,
                 correction_alpha,
                 num_smoothing_layers,
                 smoothing_alpha,
                 autoscale=True,
                 scale=1.):
        super(CorrectAndSmooth, self).__init__()
        
        self.autoscale = autoscale
        self.scale = scale

        self.prop1 = LabelPropagation(num_correction_layers,
                                      correction_alpha,
                                      )
        self.prop2 = LabelPropagation(num_smoothing_layers,
                                      smoothing_alpha,
                                      )

    def correct(self, graph, y_soft, y_true, mask, edge_weight=None):
        r"""
        Args:
            graph (dgl.DGLGraph): The graph.
            y_soft (Tensor): The soft predictions $\mathbf{Z}$ obtained
                from a simple base predictor.
            y_true (Tensor): The ground-truth label information
                $\mathbf{Y}$ of training nodes.
            mask (LongTensor or BoolTensor): A mask or index tensor denoting
                which nodes were used for training.
            edge_weight (Tensor, optional): The edge weights.
        
        Returns:
            (torch.Tensor): The corrected prediction.
        """

        with graph.local_scope():
            assert abs(float(y_soft.sum()) / y_soft.size(0) - 1.0) < 1e-2

            numel = int(mask.sum()) if mask.dtype == torch.bool else mask.size(0)
            assert y_true.size(0) == numel

            if y_true.dtype == torch.long:
                y_true = F.one_hot(y_true.view(-1), y_soft.size(-1)).to(y_soft.dtype)
            
            error = torch.zeros_like(y_soft)
            error[mask] = y_true - y_soft[mask]

            if self.autoscale:
                smoothed_error = self.prop1(graph, error, edge_weight=edge_weight, post_step=lambda x: x.clamp_(-1., 1.))

                sigma = error[mask].abs().sum() / numel
                scale = sigma / smoothed_error.abs().sum(dim=1, keepdim=True)
                scale[scale.isinf() | (scale > 1000)] = 1.0

                result = y_soft + scale * smoothed_error
                result[result.isnan()] = y_soft[result.isnan()]
                return result
            else:
                def fix_input(x):
                    x[mask] = error[mask]
                    return x
                
                smoothed_error = self.prop1(graph, error, edge_weight=edge_weight, post_step=fix_input)

                result = y_soft + self.scale * smoothed_error
                result[result.isnan()] = y_soft[result.isnan()]
                return result

    def smooth(self, graph, y_soft, y_true, mask, edge_weight=None):
        r"""
        Args:
            graph (dgl.DGLGraph): The graph.
            y_soft (Tensor): The soft predictions $\mathbf{Z}$ obtained
                from a simple base predictor.
            y_true (Tensor): The ground-truth label information
                $\mathbf{Y}$ of training nodes.
            mask (LongTensor or BoolTensor): A mask or index tensor denoting
                which nodes were used for training.
            edge_weight (Tensor, optional): The edge weights.
        
        Returns:
            (torch.Tensor): The final prediction.
        """

        with graph.local_scope():
            numel = int(mask.sum()) if mask.dtype == torch.bool else mask.size(0)
            assert y_true.size(0) == numel

            if y_true.dtype == torch.long:
                y_true = F.one_hot(y_true.view(-1), y_soft.size(-1)).to(y_soft.dtype)
            
            y_soft[mask] = y_true
            return self.prop2(graph, y_soft, edge_weight=edge_weight)