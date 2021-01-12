import math
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch.nn.functional as F
from sklearn.metrics.pairwise import cosine_similarity


class GraphConvolution(Module):
    def __init__(self, in_features, out_features, nheads, dropout, bias=False, alpha=0.2, concat=True, attencion=False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat
        self.attencion = attencion
        self.nheads = nheads
        self.dropout = dropout
        self.W = Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414).cuda()
        self.a = Parameter(torch.zeros(size=(out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.a_1 = Parameter(torch.zeros(size=(out_features, 1)))
        nn.init.xavier_uniform_(self.a_1.data, gain=1.414)
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.W.size(1))
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        if self.attencion:
            h = torch.mm(input, self.W)
            N = h.size()[0]
            if self.concat:
                a_input = torch.cat([h.repeat(1, N).view(N * N, -1), h.repeat(N, 1)], dim=1).view(N, -1, self.out_features)
                # a_input = torch.cat([h.repeat(1, N).view(N * N, -1), h.repeat(N, 1) - h.repeat(1, N).view(N * N, -1)], dim=1).view(N, -1, 2*self.out_features)
                # a_input = torch.add(a_input[:,:torch.div(a_input.size()[1], 2)], a_input[:,torch.div(a_input.size()[1], 2):]).view(N, -1, self.out_features)
                # a_input = torch.mul(a_input[:,:torch.div(a_input.size()[1], 2)], a_input[:,torch.div(a_input.size()[1], 2):]).view(N, -1, self.out_features)
                a_input = torch.mul(a_input[:, :a_input.size()[1]//2], a_input[:, a_input.size()[1]//2:]).view(N, -1, self.out_features)

                e = torch.matmul(a_input, self.a).squeeze(2)
                zero_vec = -9e15*torch.ones_like(e)
                attention = torch.where(adj > 0, e, zero_vec)
                attention = F.softmax(torch.div(attention, 8), dim=1)
                attention = F.dropout(attention, self.dropout, training=self.training)
                h_prime = torch.matmul(attention, h)
                return F.elu(h_prime)
            else:
                a_input = torch.cat([h.repeat(1, N).view(N * N, -1), h.repeat(N, 1)], dim=1).view(N, -1, self.out_features)
                # a_input = torch.cat([h.repeat(1, N).view(N * N, -1), h.repeat(N, 1) - h.repeat(1, N).view(N * N, -1)], dim=1).view(N, -1, 2*self.out_features)
                # a_input = torch.add(a_input[:,:torch.div(a_input.size()[1], 2)], a_input[:,torch.div(a_input.size()[1], 2):]).view(N, -1, self.out_features)
                a_input = torch.mul(a_input[:,:torch.div(a_input.size()[1], 2)], a_input[:,torch.div(a_input.size()[1], 2):]).view(N, -1, self.out_features)
                e = torch.matmul(a_input, self.a_1).squeeze(2)
                zero_vec = -9e15*torch.ones_like(e)
                attention = torch.where(adj > 0, e, zero_vec)
                scale = torch.sqrt(torch.FloatTensor([self.nheads]))
                attention = F.softmax(torch.div(attention, 8), dim=1)
                # attention = F.softmax(attention, dim=1)
                attention = F.dropout(attention, self.dropout, training=self.training)
                h_prime = torch.matmul(attention, h)
                return h_prime
        else:
            support = torch.mm(input, self.W)
            output = torch.spmm(adj, support)
            if self.bias is not None:
                return output + self.bias
            else:
                return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'
