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

    def __init__(self, conv, emb_dim, drop_ratio=0.5, residual=False):
        '''
            num_layers (int): number of GNN message passing layers
            emb_dim (int): node embedding dimensionality
        '''

        super(VirtualNode, self).__init__()
        self.drop_ratio = drop_ratio
        self.conv = conv
        # Add residual connection or not
        self.residual = residual

        # Set the initial virtual node embedding to 0.
        self.virtualnode_embedding = nn.Embedding(1, emb_dim)
        nn.init.constant_(self.virtualnode_embedding.weight.data, 0)

        # Batch norms applied to node embeddings
        self.batch_norm = nn.BatchNorm1d(emb_dim)

        # MLP to transform virtual node at every layer
        self.mlp_virtualnode = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.BatchNorm1d(emb_dim),
            nn.ReLU(),
            nn.Linear(emb_dim, emb_dim),
            nn.BatchNorm1d(emb_dim),
            nn.ReLU())

        self.pool = SumPooling()

    def forward(self, g, x, virtualnode_embedding=None):
        # Virtual node embeddings for graphs
        if virtualnode_embedding is None:
            virtualnode_embedding = self.virtualnode_embedding(
                torch.zeros(g.batch_size).to(x.dtype).to(x.device))

        batch_id = dgl.broadcast_nodes(
            g, torch.arange(g.batch_size).to(x.device))

        # Add message from virtual nodes to graph nodes
        h = x + virtualnode_embedding[batch_id]

        # Message passing among graph nodes
        h = self.conv(g, h)
        h = self.batch_norm(h)
        h = F.dropout(h, self.drop_ratio, training=self.training)

        if self.residual:
            h = h + x

        # Add message from graph nodes to virtual nodes
        virtualnode_embedding_temp = self.pool(
            g, h) + virtualnode_embedding

        # transform virtual nodes using MLP
        virtualnode_embedding_temp = self.mlp_virtualnode(
            virtualnode_embedding_temp)

        if self.residual:
            virtualnode_embedding = virtualnode_embedding + F.dropout(
                virtualnode_embedding_temp, self.drop_ratio, training=self.training)
        else:
            virtualnode_embedding = F.dropout(
                virtualnode_embedding_temp, self.drop_ratio, training=self.training)

        return h, virtualnode_embedding
