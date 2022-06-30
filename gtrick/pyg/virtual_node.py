"""PyG Module for Virtual Node"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import global_add_pool

"""
The code are adapted from
https://github.com/snap-stanford/ogb/tree/master/examples/graphproppred
"""


class VirtualNode(nn.Module):
    r"""Virtual Node from [OGB Graph Property Prediction Examples](https://github.com/snap-stanford/ogb/tree/master/examples/graphproppred/mol).

    It adds an virtual node to all nodes in the graph. This trick is helpful for **Graph Level Task**.

    Note:
        To use this trick, call `update_node_emb` at first, then call `update_vn_emb`.

    Examples:
        [VirtualNode (PyG)](https://nbviewer.org/github/sangyx/gtrick/blob/main/benchmark/pyg/VirtualNode.ipynb)

    Args:
        in_feats (int): Feature size before conv layer.
        out_feats (int): Feature size after conv layer.
        dropout (float, optional): Dropout rate on virtual node embedding.
        residual (bool, optional): If True, use residual connection.
    """

    def __init__(self, in_feats, out_feats, dropout=0.5, residual=False):
        super(VirtualNode, self).__init__()
        self.dropout = dropout
        # Add residual connection or not
        self.residual = residual

        # Set the initial virtual node embedding to 0.
        self.vn_emb = nn.Embedding(1, in_feats)
        # nn.init.constant_(self.virtualnode_embedding.weight.data, 0)

        if in_feats == out_feats:
            self.linear = nn.Identity()
        else:
            self.linear = nn.Linear(in_feats, out_feats)

        # MLP to transform virtual node at every layer
        self.mlp_vn = nn.Sequential(
            nn.Linear(out_feats, 2 * out_feats),
            nn.BatchNorm1d(2 * out_feats),
            nn.ReLU(),
            nn.Linear(2 * out_feats, out_feats),
            nn.BatchNorm1d(out_feats),
            nn.ReLU())
        
        self.reset_parameters()

    def reset_parameters(self):
        if not isinstance(self.linear, nn.Identity):
            self.linear.reset_parameters()

        for c in self.mlp_vn.children():
            if hasattr(c, 'reset_parameters'):
                c.reset_parameters()
        nn.init.constant_(self.vn_emb.weight.data, 0)

    def update_node_emb(self, x, edge_index, batch, vx=None):
        r""" Add message from virtual nodes to graph nodes.
        Args:
            x (torch.Tensor): The input node feature.
            edge_index (torch.LongTensor): Graph connectivity.
            batch (torch.LongTensor): Batch vector, which assigns each node to a specific example.
            vx (torch.Tensor, optional): Optional virtual node embedding.

        Returns:
            (torch.Tensor): The output node feature.
            (torch.Tensor): The output virtual node embedding.
        """
        # Virtual node embeddings for graphs
        if vx is None:
            vx = self.vn_emb(torch.zeros(
                batch[-1].item() + 1).to(edge_index.dtype).to(edge_index.device))

        # Add message from virtual nodes to graph nodes
        h = x + vx[batch]
        return h, vx

    def update_vn_emb(self, x, batch, vx):
        r""" Add message from graph nodes to virtual node.
        Args:
            x (torch.Tensor): The input node feature.
            batch (LongTensor): Batch vector, which assigns each node to a specific example.
            vx (torch.Tensor): Optional virtual node embedding.

        Returns:
            (torch.Tensor): The output virtual node embedding.
        """

        # Add message from graph nodes to virtual nodes
        vx = self.linear(vx)
        vx_temp = global_add_pool(x, batch) + vx

        # transform virtual nodes using MLP
        vx_temp = self.mlp_vn(vx_temp)

        if self.residual:
            vx = vx + F.dropout(
                vx_temp, self.dropout, training=self.training)
        else:
            vx = F.dropout(
                vx_temp, self.dropout, training=self.training)

        return vx
