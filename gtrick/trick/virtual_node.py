import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F

from dgl.nn import SumPooling


# Virtual GNN to generate node embedding
class VirtualNode(nn.Module):
    """
    Output:
        node representations
    """

    def __init__(self, conv, in_feats, out_feats, dropout=0.5, residual=False):
        '''
            num_layers (int): number of GNN message passing layers
            emb_dim (int): node embedding dimensionality
        '''

        super(VirtualNode, self).__init__()
        self.dropout = dropout
        self.conv = conv
        # Add residual connection or not
        self.residual = residual

        # Set the initial virtual node embedding to 0.
        self.virtualnode_embedding = nn.Embedding(1, in_feats)
        # nn.init.constant_(self.virtualnode_embedding.weight.data, 0)

        # Batch norms applied to node embeddings
        self.batch_norm = nn.BatchNorm1d(out_feats)

        self.linear = nn.Linear(in_feats, out_feats)

        # MLP to transform virtual node at every layer
        self.mlp_virtualnode = nn.Sequential(
            nn.Linear(out_feats, out_feats),
            nn.BatchNorm1d(out_feats),
            nn.ReLU(),
            nn.Linear(out_feats, out_feats),
            nn.BatchNorm1d(out_feats),
            nn.ReLU())

        self.pool = SumPooling()
    
    def reset_parameters(self):
        self.conv.reset_parameters()
        self.batch_norm.reset_parameters()
        self.linear.reset_parameters()
        for c in self.mlp_virtualnode.children():
            if hasattr(c, 'reset_parameters'):
                c.reset_parameters()
        nn.init.constant_(self.virtualnode_embedding.weight.data, 0)


    def forward(self, g, x, ex=None, virtualnode_embedding=None):
        # Virtual node embeddings for graphs
        if virtualnode_embedding is None:
            virtualnode_embedding = self.virtualnode_embedding(
                torch.zeros(g.batch_size).long().to(x.device))

        if g.batch_size > 1:
            batch_id = dgl.broadcast_nodes(g, torch.arange(
            g.batch_size).to(x.device).view(g.batch_size, -1)).flatten()
        else:
            batch_id = 0

        # Add message from virtual nodes to graph nodes
        h = x + virtualnode_embedding[batch_id]

        # Message passing among graph nodes
        if ex is None:
            h = self.conv(g, h)
        else:
            h = self.conv(g, h, ex)
        h = self.batch_norm(h)
        h = F.dropout(h, self.dropout, training=self.training)

        if self.residual:
            h = h + x

        # Add message from graph nodes to virtual nodes
        virtualnode_embedding = self.linear(virtualnode_embedding)
        virtualnode_embedding_temp = self.pool(
            g, h) + virtualnode_embedding

        # transform virtual nodes using MLP
        virtualnode_embedding_temp = self.mlp_virtualnode(
            virtualnode_embedding_temp)

        if self.residual:
            virtualnode_embedding = virtualnode_embedding + F.dropout(
                virtualnode_embedding_temp, self.dropout, training=self.training)
        else:
            virtualnode_embedding = F.dropout(
                virtualnode_embedding_temp, self.dropout, training=self.training)

        return h, virtualnode_embedding
