from __future__ import division
from __future__ import print_function

import time
import argparse
import numpy as np
import os
PATH = os.getcwd()
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
torch.cuda.empty_cache()
from utils import load_data, accuracy
from models import GCN
import matplotlib.pyplot as plt

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=200,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.02,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=64,
                    help='Number of hidden units.')
parser.add_argument('--nheads', type=int, default=4,
                    help='Number of heads.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--dataset', type=str, default="cora",
                    help='Dataset')
parser.add_argument('--saver', type=str, default="cora",
                    help='Saving String')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# Load data
adj, features, labels, idx_train, idx_val, idx_test = load_data(args.dataset)
# adj, features, labels, idx_train, idx_val, idx_test = load_data()

train_loss = []
validation_loss = []
train_acc = []
validation_acc = []
# Model and optimizer
model = GCN(nfeat=features.shape[1],
            nhid=args.hidden,
            nclass=labels.max().item() + 1,
            dropout=args.dropout,
            nheads=args.nheads,
            dataset=args.dataset,
            saver=args.saver)
optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
# model.load_state_dict(torch.load(PATH + '/' + 'Weights' + '/' + 'weights_only_{}.pth'.format(args.saver)))
# model.load_state_dict(torch.load(PATH + '/' + 'Weights' + '/' + 'weights_only_{}_{}.pth'.format(args.dataset, args.saver)))

if args.cuda:
    model.cuda()
    features = features.cuda()
    adj = adj.cuda()
    labels = labels.cuda()
    idx_train = idx_train.cuda()
    idx_val = idx_val.cuda()
    idx_test = idx_test.cuda()


def train(epoch):
    t = time.time()
    model.train()
    optimizer.zero_grad()
    output = model(features, adj)
    loss_train = F.nll_loss(output[idx_train], labels[idx_train])
    acc_train = accuracy(output[idx_train], labels[idx_train])
    loss_train.backward()
    optimizer.step()

    if not args.fastmode:
        # Evaluate validation set performance separately,
        # deactivates dropout during validation run.
        model.eval()
        output = model(features, adj)

    loss_val = F.nll_loss(output[idx_val], labels[idx_val])
    acc_val = accuracy(output[idx_val], labels[idx_val])
    train_loss.append(loss_train.item())
    validation_loss.append(loss_val.item())
    train_acc.append(acc_train.item())
    validation_acc.append(acc_val.item())

    print('Epoch: {:04d}'.format(epoch+1),
          'loss_train: {:.4f}'.format(loss_train.item()),
          'acc_train: {:.4f}'.format(acc_train.item()),
          'loss_val: {:.4f}'.format(loss_val.item()),
          'acc_val: {:.4f}'.format(acc_val.item()),
          'time: {:.4f}s'.format(time.time() - t))


def test():
    model.eval()
    output = model(features, adj)
    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.4f}".format(acc_test.item()))


# Train model
t_total = time.time()
for epoch in range(args.epochs):
    train(epoch)
print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

# Testing
test()



### Uncomment only for plotting the loss and accuracy curves. Please take note that you will need to create a folder named "Loss" and "Accuracy" prior to uncommenting this portion.
# x_epochs = np.linspace(1, args.epochs, args.epochs)
# plt.plot(x_epochs, train_loss, color = 'red')
# plt.plot(x_epochs, validation_loss, color = 'green')
# plt.xticks(visible=True)
# plt.yticks(visible=True)
# plt.xlabel("Epochs")
# plt.ylabel("Loss")
# plt.xlim(0,args.epochs)
# # plt.ylim(0.0,4.0)
# plt.legend(['Training Loss', 'Validation Loss'], loc='upper right')
# plt.savefig(PATH + '/' + 'Loss' + '/' + '{}_{}_Loss.png'.format(args.dataset, args.saver), dpi=300)
# plt.show()

# plt.plot(x_epochs, train_acc, color = 'red')
# plt.plot(x_epochs, validation_acc, color = 'green')
# plt.xticks(visible=True)
# plt.yticks(visible=True)
# plt.xlabel("Epochs")
# plt.ylabel("Accuracy")
# plt.xlim(0,args.epochs)
# plt.ylim(0.0,1.0)
# plt.legend(['Training Accuracy', 'Validation Accuracy'], loc='lower right')
# plt.savefig(PATH + '/' + 'Accuracy' + '/' + '{}_{}_Accuracy.png'.format(args.dataset, args.saver), dpi=300)
# plt.show()
# torch.save(model.state_dict(), PATH + '/' + 'Weights' + '/' + 'weights_only_{}_{}.pth'.format(args.dataset, args.saver))
# param = model.out_att.state_dict()["W"]
# np.save(PATH + '/' + 'Weights' + '/' + 'weights_{}_{}.npy'.format(args.dataset, args.saver),param.cpu().detach().numpy())
