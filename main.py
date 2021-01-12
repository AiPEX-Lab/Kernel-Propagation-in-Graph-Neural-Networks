import argparse

import torch
import torch.nn.functional as F
from logger_baseline import Logger
from data import data, data_full, split_idx, num_classes
from ogb.nodeproppred import Evaluator
from models import GCN, SAGE, GAT, APPNPNet, JKNet


def train(model, data, train_idx, optimizer):
    model.train()

    optimizer.zero_grad()
    out = model(data.x, data.adj_t)[train_idx]
    loss = F.nll_loss(out, data.y.squeeze(1)[train_idx])
    loss.backward()
    optimizer.step()

    return loss.item()

@torch.no_grad()
def test(model, data, split_idx, evaluator):
    model.eval()

    out = model(data.x, data.adj_t)
    y_pred = out.argmax(dim=-1, keepdim=True)

    train_acc = evaluator.eval({
        'y_true': data.y[split_idx['train']],
        'y_pred': y_pred[split_idx['train']],
    })['acc']
    valid_acc = evaluator.eval({
        'y_true': data.y[split_idx['valid']],
        'y_pred': y_pred[split_idx['valid']],
    })['acc']
    test_acc = evaluator.eval({
        'y_true': data.y[split_idx['test']],
        'y_pred': y_pred[split_idx['test']],
    })['acc']

    return train_acc, valid_acc, test_acc

parser = argparse.ArgumentParser(description='OGBN-Arxiv (GNN)')
parser.add_argument('--model', type=str, default='gcn')
parser.add_argument('--device', type=int, default=0)
parser.add_argument('--log_steps', type=int, default=1)
parser.add_argument('--num_layers', type=int, default=3)
parser.add_argument('--hidden_channels', type=int, default=256)
parser.add_argument('--dropout', type=float, default=0.5)
parser.add_argument('--att_dropout', type=float, default=0.0)
parser.add_argument('--heads', type=int, default=4)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--epochs', type=int, default=500)
parser.add_argument('--runs', type=int, default=10)
parser.add_argument('--K', type=int, default=10)
parser.add_argument('--alpha', type=float, default=0.1)
parser.add_argument('--mode', type=str, default='concat')
args = parser.parse_args()
print(args)

logger = Logger(args.runs, args)
evaluator = Evaluator(name='ogbn-arxiv')

device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
device = torch.device(device)

model = None
if args.model == 'gcn':
    model = GCN(data.num_features, args.hidden_channels,
                num_classes, args.num_layers,
                args.dropout).to(device)
elif args.model == 'sage':
    model = SAGE(data.num_features, args.hidden_channels,
                 num_classes, args.num_layers,
                 args.dropout).to(device)
elif args.model == 'gat':
    model = GAT(data.num_features, args.hidden_channels,
                num_classes, args.num_layers, args.heads,
                args.dropout, args.att_dropout).to(device)
elif args.model == 'appnp':
    model = APPNPNet(data.num_features, args.hidden_channels,
                     num_classes, args.num_layers,
                     args.K, args.alpha, args.dropout)
elif args.model == 'jknet':
    model = JKNet(data.num_features, args.hidden_channels,
                  num_classes, args.num_layers,
                  args.dropout, args.mode)


def run_model(model, train_idx, evaluator):
    for run in range(args.runs):
        model.reset_parameters()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

        for epoch in range(1, 1 + args.epochs):
            loss = train(model, data, train_idx, optimizer)
            result = test(model, data_full, split_idx, evaluator)
            logger.add_result(run, result)

            if epoch % args.log_steps == 0:
                train_acc, valid_acc, test_acc = result
                print(f'Run: {run + 1:02d}, '
                      f'Epoch: {epoch:02d}, '
                      f'Loss: {loss:.4f}, '
                      f'Train: {100 * train_acc:.2f}%, '
                      f'Valid: {100 * valid_acc:.2f}% '
                      f'Test: {100 * test_acc:.2f}%')
        logger.print_statistics(run)
    logger.print_statistics()

run_model(model, split_idx['train'], Evaluator)

