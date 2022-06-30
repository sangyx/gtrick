"""DGL Modules for Edge Feature"""

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch_sparse import SparseTensor
from dgl import to_networkx
import random
import networkx as nx

'''
The code are adapted from
https://github.com/lustoo/OGB_link_prediction
'''


class CommonNeighbors:
    r"""Compute the common neighbors of two nodes in a graph.

    Example:
        [EdgeFeat (dgl)](https://nbviewer.org/github/sangyx/gtrick/blob/main/benchmark/dgl/EdgeFeat.ipynb)

    Args:
        adj (SparseTensor): Adjacency matrix.
        edge_attr (torch.Tensor, optional): Edge feature. 
        batch_size (int, optional): The batch size to compute common neighbors. 
    """

    def __init__(self, adj, edge_attr=None, batch_size=64) -> None:
        edge_index = adj.coalesce().indices()

        if edge_attr is None:
            edge_attr = torch.ones(edge_index.shape[1])

        self.edge_index = edge_index
        self.A = SparseTensor.from_edge_index(
            edge_index, edge_attr=edge_attr).t()

        self.batch_size = batch_size

    def __call__(self, edges):
        r"""
        Args:
            edges (torch.Tensor): The edges with the shape (num_edges, 2).
        
        Returns:
            (torch.Tensor): The calculated common neighbors feature.
        """

        idx_loader = DataLoader(range(edges.shape[0]), self.batch_size)
        cn = []

        print('Calculating common neighbors as edge feature...')
        for idx in tqdm(idx_loader):
            src, dst = edges[idx, 0], edges[idx, 1]
            cn.append(torch.sum(self.A[src].to_dense()
                      * self.A[dst].to_dense(), 1))

        cn = torch.cat(cn, 0)
        return cn.view(-1, 1)


class ResourceAllocation(object):
    r"""Compute the resource allocation of two nodes in a graph.

    Resource allocation of $u$ and $v$ is defined as

    $$
    \sum_{w \in \Gamma(u) \cap \Gamma(v)} \frac{1}{|\Gamma(w)|}
    $$
    
    where $\Gamma(u)$ denotes the set of neighbors of $u$.

    Example:
        [EdgeFeat (PyG)](https://nbviewer.org/github/sangyx/gtrick/blob/main/benchmark/pyg/EdgeFeat.ipynb)

    Args:
        adj (SparseTensor): Adjacency matrix.
        edge_attr (torch.Tensor, optional): Edge feature.
        batch_size (int, optional): The batch size to compute common neighbors.
    """

    def __init__(self, adj, edge_attr=None, batch_size=64) -> None:
        edge_index = adj.coalesce().indices()

        if edge_attr is None:
            edge_attr = torch.ones(edge_index.shape[1])

        self.A = SparseTensor.from_edge_index(
            edge_index, edge_attr=edge_attr).t()

        w = 1 / self.A.sum(dim=0)
        w[torch.isinf(w)] = 0
        self.D = self.A * w.view(1, -1)

        self.batch_size = batch_size

    def __call__(self, edges):
        r"""
        Args:
            edges (torch.Tensor): The edges with the shape (num_edges, 2).
        
        Returns:
            (torch.Tensor): The calculated resource allocation feature.
        """

        idx_loader = DataLoader(range(edges.shape[0]), self.batch_size)

        ra = []

        print('Calculating resource allocation as edge feature...')
        for idx in tqdm(idx_loader):
            src, dst = edges[idx, 0], edges[idx, 1]
            ra.append(torch.sum(self.A[src].to_dense()
                      * self.D[dst].to_dense(), 1))

        ra = torch.cat(ra, 0)
        return ra.view(-1, 1)


class AdamicAdar(object):
    r"""Computes the adamic adar of two nodes in a graph.

    Adamic-Adar index of $u$ and $v$ is defined as

    $$
    \sum_{w \in \Gamma(u) \cap \Gamma(v)} \frac{1}{\log |\Gamma(w)|}
    $$
 
    where $\Gamma(u)$ denotes the set of neighbors of $u$. This index leads to zero-division for nodes only connected via self-loops. It is intended to be used when no self-loops are present.

    Example:
        [EdgeFeat (PyG)](https://nbviewer.org/github/sangyx/gtrick/blob/main/benchmark/pyg/EdgeFeat.ipynb)

    Args:
        adj (SparseTensor): Adjacency matrix.
        edge_attr (torch.Tensor, optional): Edge feature.
        batch_size (int, optional): The batch size to compute common neighbors.
    """

    def __init__(self, adj, edge_attr=None, batch_size=64) -> None:
        edge_index = adj.coalesce().indices()

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
        r"""
        Args:
            edges (torch.Tensor): The edges with the shape (num_edges, 2).
        
        Returns:
            (torch.Tensor): The calculated adamic adar feature.
        """

        idx_loader = DataLoader(range(edges.shape[0]), self.batch_size)

        aa = []

        print('Calculating adamic adar as edge feature...')
        for idx in tqdm(idx_loader):
            src, dst = edges[idx, 0], edges[idx, 1]
            aa.append(torch.sum(self.A[src].to_dense()
                      * self.D_log[dst].to_dense(), 1))

        aa = torch.cat(aa, 0)
        return aa.view(-1, 1)


class AnchorDistance(object):
    r"""Computes the anchor distance of two nodes in a graph.

    The anchor distance randomly selects $k_a$ nodes from $V$ to be anchor nodes and then calculates the shortest path starting from these anchor nodes to any other nodes. After that, the distance between $u$ and $v$ can be estimated by:

    $$
    d_{u, v}=\frac{1}{K_{A}} \sum_{i=1}^{K_{A}} d_{u, a_{i}}+d_{v, a_{i}}
    $$

    To reduce the randomness, it uses $k$ anchor sets to generate multiple distance features.

    Example:
        [EdgeFeat (PyG)](https://nbviewer.org/github/sangyx/gtrick/blob/main/benchmark/pyg/EdgeFeat.ipynb)

    Args:
        g (dgl.DGLGraph): Graph data.
        num_samples (int): The number of times to sample anchor sets.
        k (int): The size of sampled anchor sets.
        ka (int): The number of anchor nodes.
        max_spd (int, optional): The max shortest distance.
    """

    def __init__(self, g, num_samples, k, ka, max_spd=5):
        self.node_subset = random.sample(range(g.num_nodes()), k)

        node_mask = []
        for _ in range(num_samples):
            node_mask.append(np.random.choice(k, size=ka, replace=False))
        self.node_mask = np.array(node_mask)

        ng = to_networkx(g)
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
        r"""
        Args:
            edges (torch.Tensor): The edges with the shape (num_edges, 2).
        
        Returns:
            (torch.Tensor): The calculated anchor distance feature.
        """

        edges = edges.T
        ad = self.spd[edges, :].mean(0)[:, self.node_mask].mean(2)

        a_max = torch.max(ad, dim=0, keepdim=True)[0]
        a_min = torch.min(ad, dim=0, keepdim=True)[0]
        ad = (ad - a_min) / (a_max - a_min + 1e-6)
        return ad
