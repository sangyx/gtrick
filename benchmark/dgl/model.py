import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl.function as fn
from dgl.nn import GraphConv, SAGEConv, AvgPooling

from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder

class EGINConv(nn.Module):
    def __init__(self, emb_dim, mol):
        '''
        emb_dim (int): node embedding dimensionality
        '''

        super(EGINConv, self).__init__()

        self.mol = mol

        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, 2 * emb_dim),
            nn.BatchNorm1d(2 * emb_dim),
            nn.ReLU(),
            nn.Linear(2 * emb_dim, emb_dim)
        )
        self.eps = nn.Parameter(torch.Tensor([0]))

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

    def forward(self, g, x, ex):
        with g.local_scope():
            eh = self.edge_encoder(ex)
            g.ndata['x'] = x
            g.apply_edges(fn.copy_u('x', 'm'))
            g.edata['m'] = F.relu(g.edata['m'] + eh)
            g.update_all(fn.copy_e('m', 'm'), fn.sum('m', 'new_x'))
            out = self.mlp((1 + self.eps) * x + g.ndata['new_x'])

            return out

class EGCNConv(nn.Module):
    def __init__(self, emb_dim, mol):
        super(EGCNConv, self).__init__()

        self.mol = mol

        self.linear = nn.Linear(emb_dim, emb_dim)
        self.root_emb = nn.Embedding(1, emb_dim)

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

    def forward(self, g, x, ex):
        with g.local_scope():
            eh = self.edge_encoder(ex)
            x = self.linear(x)

            # Molecular graphs are undirected
            # g.out_degrees() is the same as g.in_degrees()
            degs = (g.out_degrees().float() + 1).to(x.device)
            norm = torch.pow(degs, -0.5).unsqueeze(-1)                # (N, 1)
            g.ndata['norm'] = norm
            g.apply_edges(fn.u_mul_v('norm', 'norm', 'norm'))

            g.ndata['x'] = x
            g.apply_edges(fn.copy_u('x', 'm'))

            g.edata['m'] = g.edata['norm'] * \
                F.relu(g.edata['m'] + eh)
            g.update_all(fn.copy_e('m', 'm'), fn.sum('m', 'new_x'))
            out = g.ndata['new_x'] + \
                F.relu(x + self.root_emb.weight) * 1. / degs.view(-1, 1)

            return out


class EGNN(nn.Module):

    def __init__(self, hidden_channels, out_channels, num_layers,
                 dropout, conv_type, mol=False):

        super(EGNN, self).__init__()

        self.mol = mol

        if mol:
            self.node_encoder = AtomEncoder(hidden_channels)
        else:
            self.node_encoder = nn.Embedding(1, hidden_channels)

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        for i in range(num_layers):
            if conv_type == 'gin':
                self.convs.append(
                    EGINConv(hidden_channels, self.mol))
            elif conv_type == 'gcn':
                self.convs.append(
                    EGCNConv(hidden_channels, self.mol))

            self.bns.append(nn.BatchNorm1d(hidden_channels))

        self.dropout = dropout

        self.pool = AvgPooling()

        self.out = nn.Linear(hidden_channels, out_channels)

    def reset_parameters(self):
        if self.mol:
            for emb in self.node_encoder.atom_embedding_list:
                nn.init.xavier_uniform_(emb.weight.data)
        else:
            nn.init.xavier_uniform_(self.node_encoder.weight.data)

        num_layers = len(self.convs)
        for i in range(num_layers):
            self.convs[i].reset_parameters()
            self.bns[i].reset_parameters()

        self.out.reset_parameters()

    def forward(self, g, x, ex):
        h = self.node_encoder(x)

        for i, conv in enumerate(self.convs[:-1]):
            h = conv(g, h, ex)
            h = self.bns[i](h)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)

        h = self.convs[-1](g, h, ex)
        
        if not self.mol:
            h = self.bns[-1](h)
        
        h = F.dropout(h, self.dropout, training = self.training)

        h = self.pool(g, h)

        h = self.out(h)

        return h

class GNN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout, conv_type):
        super(GNN, self).__init__()

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        for i in range(num_layers):
            if conv_type == 'gcn':
                if i == 0:
                    self.convs.append(GraphConv(in_channels, hidden_channels))
                elif i == num_layers - 1:
                    self.convs.append(GraphConv(hidden_channels, out_channels))
                else:
                    self.convs.append(
                        GraphConv(hidden_channels, hidden_channels))
            elif conv_type == 'sage':
                if i == 0:
                    self.convs.append(SAGEConv(in_channels, hidden_channels, 'mean'))
                elif i == num_layers - 1:
                    self.convs.append(SAGEConv(hidden_channels, out_channels, 'mean'))
                else:
                    self.convs.append(
                        SAGEConv(hidden_channels, hidden_channels, 'mean'))
            
            if i != num_layers - 1:
                self.bns.append(nn.BatchNorm1d(hidden_channels))

        self.dropout = dropout


    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, g, x):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(g, x)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](g, x)

        return x