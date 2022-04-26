import torch
from torch import nn
import dgl


class DropEdge(nn.Module):
    def __init__(self, p) -> None:
        super(DropEdge).__init__()
        self.p = p

    def forward(self, g):
        eids = g.edges(form='eid')
        mask = torch.rand(eids.shape) < (1 - self.p)
        ng = dgl.remove_edges(g, eids[mask])
        return ng
