import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool, global_add_pool, GCNConv, SAGEConv
from torch_geometric.utils import degree

from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder


class EGINConv(MessagePassing):
    def __init__(self, emb_dim, mol):
        '''
            emb_dim (int): node embedding dimensionality
        '''

        super(EGINConv, self).__init__(aggr="add")

        self.mol = mol

        self.mlp = torch.nn.Sequential(torch.nn.Linear(emb_dim, 2 * emb_dim), torch.nn.BatchNorm1d(
            2 * emb_dim), torch.nn.ReLU(), torch.nn.Linear(2 * emb_dim, emb_dim))
        self.eps = torch.nn.Parameter(torch.Tensor([0]))

        if self.mol:
            self.edge_encoder = BondEncoder(emb_dim)
        else:
            self.edge_encoder = nn.Linear(7, emb_dim)

    def reset_parameters(self):
        for c in self.mlp.children():
            if hasattr(c, 'reset_parameters'):
                c.reset_parameters()
        nn.init.constant_(self.eps.data, 0)

        if self.mol:
            for emb in self.edge_encoder.bond_embedding_list:
                nn.init.xavier_uniform_(emb.weight.data)
        else:
            self.edge_encoder.reset_parameters()

    def forward(self, x, edge_index, edge_attr):
        edge_embedding = self.edge_encoder(edge_attr)
        out = self.mlp((1 + self.eps) * x +
                       self.propagate(edge_index, x=x, edge_attr=edge_embedding))

        return out

    def message(self, x_j, edge_attr):
        return F.relu(x_j + edge_attr)

    def update(self, aggr_out):
        return aggr_out


class EGCNConv(MessagePassing):
    def __init__(self, emb_dim, mol):
        super(EGCNConv, self).__init__(aggr='add')

        self.mol = mol

        self.linear = torch.nn.Linear(emb_dim, emb_dim)
        self.root_emb = torch.nn.Embedding(1, emb_dim)

        if self.mol:
            self.edge_encoder = BondEncoder(emb_dim)
        else:
            self.edge_encoder = nn.Linear(7, emb_dim)

    def reset_parameters(self):
        self.linear.reset_parameters()
        self.root_emb.reset_parameters()
        
        if self.mol:
            for emb in self.edge_encoder.bond_embedding_list:
                nn.init.xavier_uniform_(emb.weight.data)
        else:
            self.edge_encoder.reset_parameters()

    def forward(self, x, edge_index, edge_attr):
        x = self.linear(x)
        edge_embedding = self.edge_encoder(edge_attr)

        row, col = edge_index

        #edge_weight = torch.ones((edge_index.size(1), ), device=edge_index.device)
        deg = degree(row, x.size(0), dtype=x.dtype) + 1
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        return self.propagate(edge_index, x=x, edge_attr=edge_embedding, norm=norm) + F.relu(x + self.root_emb.weight) * 1./deg.view(-1, 1)

    def message(self, x_j, edge_attr, norm):
        return norm.view(-1, 1) * F.relu(x_j + edge_attr)

    def update(self, aggr_out):
        return aggr_out


class EGNN(nn.Module):

    def __init__(self, hidden_channels, out_channels, num_layers,
                 dropout, conv_type, mol=False, pooling_type='mean', use_mlp_after_graph_embed=False):

        super(EGNN, self).__init__()

        self.mol = mol

        if mol:
            self.node_encoder = AtomEncoder(hidden_channels)
        else:
            self.node_encoder = nn.Embedding(1, hidden_channels)

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        self.num_layers = num_layers

        for i in range(self.num_layers):
            if conv_type == 'gin':
                self.convs.append(
                    EGINConv(hidden_channels, self.mol))
            elif conv_type == 'gcn':
                self.convs.append(
                    EGCNConv(hidden_channels, self.mol))

            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))

        self.dropout = dropout

        if pooling_type == 'mean':
            self.pool = global_mean_pool
        elif pooling_type == 'add':
            self.pool = global_add_pool
        else:
            raise Exception(f"Invalid pooling type: {pooling_type}; only 'mean' and 'add' supported")

        self.use_mlp_after_graph_embed = use_mlp_after_graph_embed
        if self.use_mlp_after_graph_embed:
            self.graph_embed_hidden = nn.Linear(hidden_channels, hidden_channels)

        self.out = nn.Linear(hidden_channels, out_channels)

    def reset_parameters(self):
        if self.mol:
            for emb in self.node_encoder.atom_embedding_list:
                nn.init.xavier_uniform_(emb.weight.data)
        else:
            nn.init.xavier_uniform_(self.node_encoder.weight.data)

        for i in range(self.num_layers):
            self.convs[i].reset_parameters()
            self.bns[i].reset_parameters()

        self.out.reset_parameters()
        if self.use_mlp_after_graph_embed:
            self.graph_embed_hidden.reset_parameters()

    def forward(self, batch_data):
        x, edge_index, edge_attr, batch = batch_data.x, batch_data.edge_index, batch_data.edge_attr, batch_data.batch

        h = self.node_encoder(x)

        for i, conv in enumerate(self.convs[:-1]):
            h = conv(h, edge_index, edge_attr)
            h = self.bns[i](h)
            h = F.relu(h)
            
            h = F.dropout(h, p=self.dropout, training=self.training)

        h = self.convs[-1](h, edge_index, edge_attr)

        if not self.mol:
            h = self.bns[-1](h)

        h = F.dropout(h, self.dropout, training=self.training)

        h = self.pool(h, batch)

        # do not require graph embeddings to be linearly separable
        # graph embeds from raw pooling of node embeds may not be
        if self.use_mlp_after_graph_embed:
            h = F.dropout(h, self.dropout, training=self.training)
            h = self.graph_embed_hidden(h)
            h = F.relu(h)
            h = F.dropout(h, self.dropout, training=self.training)

        h = self.out(h)

        return h


class GNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout, conv_type):
        super(GNN, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()

        for i in range(num_layers):
            if conv_type == 'gcn':
                if i == 0:
                    self.convs.append(
                        GCNConv(in_channels, hidden_channels, cached=True))
                elif i == num_layers - 1:
                    self.convs.append(
                        GCNConv(hidden_channels, out_channels, cached=True))
                else:
                    self.convs.append(
                        GCNConv(hidden_channels, hidden_channels, cached=True))
            elif conv_type == 'sage':
                if i == 0:
                    self.convs.append(SAGEConv(in_channels, hidden_channels))
                elif i == num_layers - 1:
                    self.convs.append(
                        SAGEConv(hidden_channels, out_channels))
                else:
                    self.convs.append(
                        SAGEConv(hidden_channels, hidden_channels))

            if i != num_layers - 1:
                self.bns.append(torch.nn.BatchNorm1d(hidden_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, adj_t):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, adj_t)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        return x
