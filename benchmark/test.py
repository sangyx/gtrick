import sys 
sys.path.append("..")

import torch
import torch.nn as nn
import torch.nn.functional as F

from dgl.nn import AvgPooling
import dgl.function as fn
from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder

from gtrick import VirtualNode

from model import EGINConv, EGCNConv


class EGIN(nn.Module):

    def __init__(self, hidden_channels, out_channels, num_layers,
                 dropout):

        super(EGIN, self).__init__()

        self.node_encoder = AtomEncoder(hidden_channels)

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.vns = nn.ModuleList()

        for i in range(num_layers):
            self.convs.append(
                EGINConv(hidden_channels))

            if i != num_layers - 1:
                self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
                self.vns.append(VirtualNode(hidden_channels, hidden_channels, dropout=dropout))

        self.dropout = dropout

        self.pool = AvgPooling()

        self.out = nn.Linear(hidden_channels, out_channels)

    def reset_parameters(self):
        for emb in self.node_encoder.atom_embedding_list:
            torch.nn.init.xavier_uniform_(emb.weight.data)

        num_layers = len(self.convs)
        for i in range(num_layers):
            self.convs[i].reset_parameters()
            if i != num_layers - 1:
                self.bns[i].reset_parameters()
                self.vns[i].reset_parameters()

        self.out.reset_parameters()

    def forward(self, g, x, ex):
        h = self.node_encoder(x)

        vx = None

        for i, conv in enumerate(self.convs[:-1]):
            h, vx = self.vns[i].update_node_emb(g, h, vx)

            h = conv(g, h, ex)
            h = self.bns[i](h)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
            
            vx = self.vns[i].update_vn_emb(g, h, vx)

        h = self.convs[-1](g, h, ex)
        h = F.dropout(h, self.dropout, training = self.training)

        h = self.pool(g, h)

        h = self.out(h)

        return h


class EGCN(nn.Module):

    def __init__(self, hidden_channels, out_channels, num_layers,
                 dropout):

        super(EGCN, self).__init__()

        self.node_encoder = AtomEncoder(hidden_channels)

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.vns = nn.ModuleList()

        for i in range(num_layers):
            self.convs.append(
                EGCNConv(hidden_channels))

            if i != num_layers - 1:
                self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
                self.vns.append(VirtualNode(hidden_channels, hidden_channels, dropout=dropout))

        self.dropout = dropout

        self.pool = AvgPooling()

        self.out = nn.Linear(hidden_channels, out_channels)

    def reset_parameters(self):
        for emb in self.node_encoder.atom_embedding_list:
            torch.nn.init.xavier_uniform_(emb.weight.data)

        num_layers = len(self.convs)
        for i in range(num_layers):
            self.convs[i].reset_parameters()
            if i != num_layers - 1:
                self.bns[i].reset_parameters()
                self.vns[i].reset_parameters()

        self.out.reset_parameters()

    def forward(self, g, x, ex):
        h = self.node_encoder(x)

        vx = None

        for i, conv in enumerate(self.convs[:-1]):
            h, vx = self.vns[i].update_node_emb(g, h, vx)

            h = conv(g, h, ex)
            h = self.bns[i](h)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
            
            vx = self.vns[i].update_vn_emb(g, h, vx)

        h = self.convs[-1](g, h, ex)
        h = F.dropout(h, self.dropout, training = self.training)

        h = self.pool(g, h)

        h = self.out(h)

        return h


import argparse
from ogb.graphproppred import DglGraphPropPredDataset
from graph_pred import run_graph_pred

parser = argparse.ArgumentParser(
    description='train graph property prediction')
parser.add_argument("--dataset", type=str, default="ogbg-molhiv",
                    choices=["ogbg-molhiv"])
parser.add_argument("--dataset_path", type=str, default="/home/ubuntu/.dgl_dataset",
                    help="path to dataset")
parser.add_argument('--device', type=int, default=0)
parser.add_argument('--log_steps', type=int, default=1)
parser.add_argument('--num_layers', type=int, default=5)
parser.add_argument('--hidden_channels', type=int, default=300)
parser.add_argument('--dropout', type=float, default=0.5)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument("--batch_size", type=int, default=32,
                    help="batch size")
parser.add_argument('--num_workers', type=int, default=0,
                    help='number of workers (default: 0)')
parser.add_argument('--model', type=str, default='gin')
parser.add_argument('--epochs', type=int, default=500)
parser.add_argument('--runs', type=int, default=5)
parser.add_argument('--patience', type=int, default=30)
args = parser.parse_args()
# args = parser.parse_args()
print(args)

dataset = DglGraphPropPredDataset(
    name=args.dataset, root=args.dataset_path)

if args.model == 'gin':
    model = EGIN(args.hidden_channels,
                    dataset.num_tasks, args.num_layers,
                    args.dropout)
elif args.model == 'gcn':
    model = EGCN(args.hidden_channels,
                    dataset.num_tasks, args.num_layers,
                    args.dropout)

run_graph_pred(args, model, dataset)