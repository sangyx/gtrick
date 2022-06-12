"""PyG Modules for Edge Feature"""

import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch_geometric.utils import to_networkx
import networkx as nx

from torch_sparse import SparseTensor

'''
The code are adapted from
https://github.com/lustoo/OGB_link_prediction
'''


class CommonNeighbors:
    def __init__(self, edge_index, edge_attr=None, batch_size=64) -> None:
        if edge_attr is None:
            edge_attr = torch.ones(edge_index.shape[1])

        self.edge_index = edge_index
        self.A = SparseTensor.from_edge_index(
            edge_index, edge_attr=edge_attr).t()

        self.batch_size = batch_size

    def __call__(self, edges):
        idx_loader = DataLoader(range(edges.shape[0]), self.batch_size)
        cn = []

        print('Calculating common neighbors as edge feature...')
        for idx in tqdm(idx_loader):
            src, dst = edges[idx, 0], edges[idx, 1]
            cn.append(torch.sum(self.A[src].to_dense()
                      * self.A[dst].to_dense(), 1))

        cn = torch.cat(cn, 0)
        return cn.view(-1, 1)


class ResourceAllocation:
    def __init__(self, edge_index, edge_attr=None, batch_size=64) -> None:
        if edge_attr is None:
            edge_attr = torch.ones(edge_index.shape[1])

        self.A = SparseTensor.from_edge_index(
            edge_index, edge_attr=edge_attr).t()

        w = 1 / self.A.sum(dim=0)
        w[torch.isinf(w)] = 0
        self.D = self.A * w.view(1, -1)

        self.batch_size = batch_size

    def __call__(self, edges):
        idx_loader = DataLoader(range(edges.shape[0]), self.batch_size)

        ra = []

        print('Calculating resource allocation as edge feature...')
        for idx in tqdm(idx_loader):
            src, dst = edges[idx, 0], edges[idx, 1]
            ra.append(torch.sum(self.A[src].to_dense()
                      * self.D[dst].to_dense(), 1))

        ra = torch.cat(ra, 0)
        return ra.view(-1, 1)


class AdamicAdar:
    def __init__(self, edge_index, edge_attr=None, batch_size=64) -> None:
        if edge_attr is None:
            edge_attr = torch.ones(edge_index.shape[1])

        self.A = SparseTensor.from_edge_index(
            edge_index, edge_attr=edge_attr).t()

        temp = torch.log(self.A.sum(dim=0))
        temp = 1 / temp
        temp[torch.isinf(temp)] = 1
        self.D_log = self.A * temp.view(1, -1)

        self.batch_size = batch_size

    def __call__(self, edges=None):
        idx_loader = DataLoader(range(edges.shape[0]), self.batch_size)

        aa = []

        print('Calculating adamic adar as edge feature...')
        for idx in tqdm(idx_loader):
            src, dst = edges[idx, 0], edges[idx, 1]
            aa.append(torch.sum(self.A[src].to_dense()
                      * self.D_log[dst].to_dense(), 1))

        aa = torch.cat(aa, 0)
        return aa.view(-1, 1)


class AnchorDistance:
    def __init__(self, data, num_samples, k, ka, max_spd=5, to_undirected=True):
        self.node_subset = random.sample(range(data.num_nodes), k)

        node_mask = []
        for _ in range(num_samples):
            node_mask.append(np.random.choice(k, size=ka, replace=False))
        self.node_mask = np.array(node_mask)

        ng = to_networkx(data, to_undirected=to_undirected)
        self.max_spd = max_spd
        self.spd = self.get_spd_matrix(
            g=ng, s=self.node_subset, max_spd=max_spd)

    def get_spd_matrix(self, g, s, max_spd):
        print('Calculating anchor distance...')
        spd_matrix = torch.zeros(g.number_of_nodes(), len(s))

        i = 0
        for s in tqdm(s):
            for node, length in nx.shortest_path_length(g, source=s).items():
                spd_matrix[node, i] = min(length, max_spd)
            i += 1
        return spd_matrix

    def __call__(self, edges):
        edges = edges.T
        ad = self.spd[edges, :].mean(0)[:, self.node_mask].mean(2)

        a_max = torch.max(ad, dim=0, keepdim=True)[0]
        a_min = torch.min(ad, dim=0, keepdim=True)[0]
        ad = (ad - a_min) / (a_max - a_min + 1e-6)
        return ad
