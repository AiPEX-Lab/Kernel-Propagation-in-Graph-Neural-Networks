import argparse
import os
import numpy as np
import networkx as nx
import scipy.sparse as sp
import scipy as sc
from scipy.sparse.linalg.eigen.arpack import eigsh
from logger import Logger
from ogb.nodeproppred import PygNodePropPredDataset, Evaluator
from utils import sparse_mx_to_torch_sparse_tensor, T_summer, graph_srw_transition_matrix, normalize, normalize_adj
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn import Linear
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv, SAGEConv, GATConv, APPNP, JumpingKnowledge
from layers import GraphConvolution
from torch_sparse import SparseTensor
import scipy.sparse
PATH = os.getcwd()

class AKPGCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(AKPGCN, self).__init__()

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

class AKPGAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, heads,
                 dropout, att_dropout):
        super(AKPGAT, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(GATConv(in_channels, hidden_channels, heads=heads, dropout=att_dropout, concat=True))
        self.bns = torch.nn.ModuleList()
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels * heads))
        for _ in range(num_layers-2):
            self.convs.append(
                GATConv(hidden_channels * heads, hidden_channels, heads=heads, dropout=att_dropout, concat=True))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels * heads))
        self.convs.append(GATConv(hidden_channels * heads, out_channels, heads=heads, dropout=att_dropout, concat=False))

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
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        return x

class GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, heads,
                 dropout, att_dropout):
        super(GAT, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(GATConv(in_channels, hidden_channels, heads=heads, dropout=att_dropout, concat=True))
        self.bns = torch.nn.ModuleList()
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels * heads))
        for _ in range(num_layers - 2):
            self.convs.append(
                GATConv(hidden_channels * heads, hidden_channels, heads=heads, dropout=att_dropout, concat=True))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels * heads))
        self.convs.append(GATConv(hidden_channels * heads, out_channels, heads=heads, dropout=att_dropout, concat=False))

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
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        return x

class JKNet(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout, mode='concat'):
        super().__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels, cached=False))
        self.bns = torch.nn.ModuleList()
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_channels, hidden_channels, cached=False))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        self.jump = JumpingKnowledge(mode)
        if mode == 'cat':
            self.lin1 = Linear(num_layers * hidden_channels, hidden_channels)
        else:
            self.lin1 = Linear(hidden_channels, hidden_channels)

        self.lin2 = Linear(hidden_channels, out_channels)
        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

        self.jump.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, x, adj_t):
        xs = []
        for i, conv in enumerate(self.convs):
            x = conv(x, adj_t)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            xs += [x]

        x = self.jump(xs)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)
        return x

class APPNPNet(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, K, alpha,
        dropout):
        super(APPNPNet, self).__init__()

        self.lins = torch.nn.ModuleList()
        self.lins.append(Linear(in_channels, hidden_channels))
        self.bns = torch.nn.ModuleList()
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.lins.append(Linear(hidden_channels, hidden_channels))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        self.lins.append(Linear(hidden_channels, out_channels))
        self.prop = APPNP(K, alpha)

        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, adj_t):
        for i, lin in enumerate(self.lins[:-1]):
            x = lin(x)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.lins[-1](x)
        x = self.prop(x, adj_t)
        return x

def train(model, data, train_idx, optimizer):
    model.train()
    criterion = torch.nn.BCEWithLogitsLoss()

    optimizer.zero_grad()
    out = model(data.x, data.adj_t)[train_idx]
    loss = criterion(out, data.y[train_idx].to(torch.float))
    loss.backward()
    optimizer.step()

    return loss.item()


@torch.no_grad()
def test(model, data, split_idx, evaluator):
    model.eval()

    y_pred = model(data.x, data.adj_t)

    train_rocauc = evaluator.eval({
        'y_true': data.y[split_idx['train']],
        'y_pred': y_pred[split_idx['train']],
    })['rocauc']
    valid_rocauc = evaluator.eval({
        'y_true': data.y[split_idx['valid']],
        'y_pred': y_pred[split_idx['valid']],
    })['rocauc']
    test_rocauc = evaluator.eval({
        'y_true': data.y[split_idx['test']],
        'y_pred': y_pred[split_idx['test']],
    })['rocauc']

    return train_rocauc, valid_rocauc, test_rocauc


def main():
    parser = argparse.ArgumentParser(description='OGBN-Arxiv (GNN)')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--log_steps', type=int, default=1)
    parser.add_argument('--model', type=str, default='gcn')
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--hidden_channels', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--runs', type=int, default=2)
    parser.add_argument('--att_dropout', type=float, default=0.0)
    parser.add_argument('--heads', type=int, default=4)
    parser.add_argument('--mode', type=str, default='cat')
    parser.add_argument('--eval_steps', type=int, default=5)
    parser.add_argument('--thresh', type=int, default=99)
    parser.add_argument('--K', type=int, default=10)
    parser.add_argument('--alpha', type=float, default=0.1)


    args = parser.parse_args()
    print(args)

    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    scores_val = []
    scores_train = []
    scores_test = []

    dataset = PygNodePropPredDataset(name='ogbn-proteins',
                                     transform=T.ToSparseTensor())
    data = dataset[0]

    # Move edge features to node features.
    data.x = data.adj_t.mean(dim=1)
    data.adj_t.set_value_(None)

    # # Pre-compute GCN normalization.
    adj_t = data.adj_t.set_diag()
    deg = adj_t.sum(dim=1).to(torch.float)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
    adj_t = deg_inv_sqrt.view(-1, 1) * adj_t * deg_inv_sqrt.view(1, -1)
    data.adj_t = adj_t

    ######### Uncomment this chunk to use identity matrix as features
    # data.x = SparseTensor.eye(132534, 132534)
    # rowptr, col, value = data.x.csr()
    # rowptr = rowptr.numpy()
    # col = col.numpy()
    # value = value.numpy()
    # data.x = scipy.sparse.csr_matrix((value, col, rowptr), data.x.sizes())
    # sparse_mx = data.x.tocoo().astype(np.float32)
    # indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    # values = torch.from_numpy(sparse_mx.data)
    # shape = torch.Size(sparse_mx.shape)
    # data.x = torch.sparse.FloatTensor(indices, values, shape)
    ############################################

    A = data.adj_t.to_dense()  # Adjacency Matrix (A)
    TT = graph_srw_transition_matrix(A)  # Graph Transition Matrix
    thresh = args.thresh/100.0
    T_Power = T_summer(TT, 0.99)  # Two-hop Graph Transition ## uncomment if anything other than pubmed
    T_Power = normalize_adj(T_Power)
    T_Power = sparse_mx_to_torch_sparse_tensor(T_Power)
    torch.save(T_Power, 'ogbn-proteins-summed-T.pt')
    # T_Power = torch.load('ogbn-proteins-summed-T.pt')
    # data.x = T_Power
    # data.x = sparse_mx_to_torch_sparse_tensor(sp.csr_matrix(data.x))
    data.x = torch.cat([T_Power,data.x],1)
    torch.save(data, 'ogbn-proteins_{}_{}.pt'.format(args.hidden_channels, args.model))

    split_idx = dataset.get_idx_split()
    train_idx = split_idx['train'].to(device)
    data = data.to(device)

    if args.model == 'sage':
        model = SAGE(data.num_features, args.hidden_channels,
                    112, args.num_layers,
                    args.dropout).to(device)
    elif args.model == 'gcn':
        model = GCN(data.num_features, args.hidden_channels, 112,
                    args.num_layers, args.dropout).to(device)
    elif args.model == 'gat':
        model = GAT(data.num_features, args.hidden_channels,
                    112, args.num_layers, args.heads,
                    args.dropout, args.att_dropout).to(device)
    elif args.model == 'akpgcn':
        model = AKPGCN(data.num_features, args.hidden_channels,
                    112, args.num_layers,
                    args.dropout).to(device)
    elif args.model == 'akpgat':
        model = AKPGAT(data.num_features, args.hidden_channels,
                    112, args.num_layers, args.heads,
                    args.dropout, args.att_dropout).to(device)
    elif args.model == 'jk':
        model = JKNet(data.num_features, args.hidden_channels,
                    112, args.num_layers,
                    args.dropout, args.mode).to(device)
    elif args.model == 'appnp':
        model = APPNPNet(data.num_features, args.hidden_channels,
                         112, args.num_layers,
                         args.K, args.alpha, args.dropout).to(device)

    evaluator = Evaluator(name='ogbn-proteins')
    logger = Logger(args.runs, scores_val, scores_train, scores_test, args)

    for run in range(args.runs):
        model.reset_parameters()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        for epoch in range(1, 1 + args.epochs):
            loss = train(model, data, train_idx, optimizer)

            if epoch % args.eval_steps == 0:
                result = test(model, data, split_idx, evaluator)
                logger.add_result(run, result)

                if epoch % args.log_steps == 0:
                    train_rocauc, valid_rocauc, test_rocauc = result
                    print(f'Run: {run + 1:02d}, '
                          f'Epoch: {epoch:02d}, '
                          f'Loss: {loss:.4f}, '
                          f'Train: {100 * train_rocauc:.2f}%, '
                          f'Valid: {100 * valid_rocauc:.2f}% '
                          f'Test: {100 * test_rocauc:.2f}%')

        logger.print_statistics(run)
    logger.print_statistics()

    scores_val = np.asarray(scores_val)
    scores_train = np.asarray(scores_train)
    scores_test = np.asarray(scores_test)
    np.save("scores_val_{}_{}.npy".format(args.thresh, args.model), scores_val)
    np.save("scores_train_{}_{}.npy".format(args.thresh, args.model), scores_train)
    np.save("scores_test_{}_{}.npy".format(args.thresh, args.model), scores_test)


if __name__ == "__main__":
    main()
