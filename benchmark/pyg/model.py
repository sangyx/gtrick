import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool, GCNConv, SAGEConv
from torch_geometric.utils import degree

from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder

### GIN convolution along the graph structure
class EGINConv(MessagePassing):
    def __init__(self, emb_dim):
        '''
            emb_dim (int): node embedding dimensionality
        '''

        super(EGINConv, self).__init__(aggr="add")

        self.mlp = torch.nn.Sequential(torch.nn.Linear(emb_dim, 2*emb_dim), torch.nn.BatchNorm1d(2*emb_dim), torch.nn.ReLU(), torch.nn.Linear(2*emb_dim, emb_dim))
        self.eps = torch.nn.Parameter(torch.Tensor([0]))

        self.edge_encoder = BondEncoder(emb_dim=emb_dim)
    
    def reset_parameters(self):
        for c in self.mlp.children():
            if hasattr(c, 'reset_parameters'):
                c.reset_parameters()
        nn.init.constant_(self.eps.data, 0)
        for emb in self.edge_encoder.bond_embedding_list:
            nn.init.xavier_uniform_(emb.weight.data)

    def forward(self, x, edge_index, edge_attr):
        edge_embedding = self.edge_encoder(edge_attr)
        out = self.mlp((1 + self.eps) *x + self.propagate(edge_index, x=x, edge_attr=edge_embedding))

        return out

    def message(self, x_j, edge_attr):
        return F.relu(x_j + edge_attr)

    def update(self, aggr_out):
        return aggr_out

### GCN convolution along the graph structure
class EGCNConv(MessagePassing):
    def __init__(self, emb_dim):
        super(EGCNConv, self).__init__(aggr='add')

        self.linear = torch.nn.Linear(emb_dim, emb_dim)
        self.root_emb = torch.nn.Embedding(1, emb_dim)
        self.edge_encoder = BondEncoder(emb_dim = emb_dim)
    
    def reset_parameters(self):
        self.linear.reset_parameters()
        self.root_emb.reset_parameters()
        for emb in self.edge_encoder.bond_embedding_list:
            nn.init.xavier_uniform_(emb.weight.data)

    def forward(self, x, edge_index, edge_attr):
        x = self.linear(x)
        edge_embedding = self.edge_encoder(edge_attr)

        row, col = edge_index

        #edge_weight = torch.ones((edge_index.size(1), ), device=edge_index.device)
        deg = degree(row, x.size(0), dtype = x.dtype) + 1
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        return self.propagate(edge_index, x=x, edge_attr=edge_embedding, norm=norm) + F.relu(x + self.root_emb.weight) * 1./deg.view(-1,1)

    def message(self, x_j, edge_attr, norm):
        return norm.view(-1, 1) * F.relu(x_j + edge_attr)

    def update(self, aggr_out):
        return aggr_out


class EGCN(nn.Module):

    def __init__(self, hidden_channels, out_channels, num_layers,
                 dropout):

        super(EGCN, self).__init__()

        self.node_encoder = AtomEncoder(hidden_channels)

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        for i in range(num_layers):
            self.convs.append(
                EGCNConv(hidden_channels))
            if i != num_layers - 1:
                self.bns.append(nn.BatchNorm1d(hidden_channels))

        self.dropout = dropout

        self.out = nn.Linear(hidden_channels, out_channels)

    def reset_parameters(self):
        for emb in self.node_encoder.atom_embedding_list:
            torch.nn.init.xavier_uniform_(emb.weight.data)

        num_layers = len(self.convs)

        for i in range(num_layers):
            self.convs[i].reset_parameters()
            if i != num_layers - 1:
                self.bns[i].reset_parameters()

        self.out.reset_parameters()

    def forward(self, batched_data):
        x, edge_index, edge_attr, batch = batched_data.x, batched_data.edge_index, batched_data.edge_attr, batched_data.batch

        h = self.node_encoder(x)

        for i, conv in enumerate(self.convs[:-1]):
            h = conv(h, edge_index, edge_attr)
            h = self.bns[i](h)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
        h = self.convs[-1](h, edge_index, edge_attr)
        h = F.dropout(h, self.dropout, training=self.training)

        h = global_mean_pool(h, batch)
        h = self.out(h)

        return h


class EGIN(nn.Module):
    def __init__(self, hidden_channels, out_channels, num_layers,
                 dropout):

        super(EGIN, self).__init__()

        self.node_encoder = AtomEncoder(hidden_channels)

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        for i in range(num_layers):
            self.convs.append(
                EGINConv(hidden_channels))
            if i != num_layers - 1:
                self.bns.append(nn.BatchNorm1d(hidden_channels))

        self.dropout = dropout

        self.out = nn.Linear(hidden_channels, out_channels)

    def reset_parameters(self):
        for emb in self.node_encoder.atom_embedding_list:
            nn.init.xavier_uniform_(emb.weight.data)
            
        num_layers = len(self.convs)

        for i in range(num_layers):
            self.convs[i].reset_parameters()
            if i != num_layers - 1:
                self.bns[i].reset_parameters()

        self.out.reset_parameters()

    def forward(self, batched_data):
        x, edge_index, edge_attr, batch = batched_data.x, batched_data.edge_index, batched_data.edge_attr, batched_data.batch

        h = self.node_encoder(x)

        for i, conv in enumerate(self.convs[:-1]):
            h = conv(h, edge_index, edge_attr)
            h = self.bns[i](h)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
        h = self.convs[-1](h, edge_index, edge_attr)
        h = F.dropout(h, self.dropout, training=self.training)

        h = global_mean_pool(h, batch)
        h = self.out(h)

        return h


class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(GCN, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels, cached=True))
        self.bns = torch.nn.ModuleList()
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(
                GCNConv(hidden_channels, hidden_channels, cached=True))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        self.convs.append(GCNConv(hidden_channels, out_channels, cached=True))

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


class SAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(SAGE, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        self.bns = torch.nn.ModuleList()
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels))

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