import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from layers import GraphConvolution
import os
PATH = os.getcwd()

class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, nheads, dataset, saver):
        super().__init__()
        nhid1 = 32
        nhid2 = 4
        self.gc1 = GraphConvolution(nfeat, nhid, nheads, dropout, alpha=0.2, concat=False, attencion=False)
        self.attentions = [GraphConvolution(nhid, nclass, nheads, dropout, alpha=0.2, concat=True, attencion=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)
        # self.out_att = GraphConvolution(nhid1*nheads, nclass, nheads, dropout, alpha=0.2, concat=False, attencion=True)
        self.gc2 = GraphConvolution(nhid, nclass, nheads, dropout, alpha=0.2, concat=False, attencion=False)
        self.bn = torch.nn.BatchNorm1d(nhid)
        self.dropout = dropout
        self.dataset = dataset
        self.saver = saver
        self.nheads = nheads
        self.nclass = nclass

    def forward(self, x, adj):
        x = self.gc1(x, adj)
        x = F.relu(self.bn(x))
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        # x = F.dropout(x, self.dropout, training=self.training)
        # x = F.elu(self.out_att(x, adj))
        # x = F.dropout(x, self.dropout, training=self.training)
        # x = self.gc2(x, adj)
        xx = x
        xx = xx.cpu().detach().numpy()
        np.save(PATH + '/' + 'Weights' + '/' + '{}_{}.npy'.format(self.dataset, self.saver),xx)
        return F.log_softmax(x, dim=1)
