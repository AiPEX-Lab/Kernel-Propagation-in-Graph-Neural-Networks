import numpy as np
import scipy.sparse as sp
import torch
import sys
import networkx as nx
from keras.utils import to_categorical
import pickle as pkl
import scipy as sc
from scipy.sparse.linalg.eigen.arpack import eigsh
from scipy import sparse

np.set_printoptions(threshold=sys.maxsize)
torch.set_printoptions(profile="full")

def laplacian(a):
    n_nodes, _ = a.shape
    posinv = np.vectorize(lambda x: 1.0/np.sqrt(x) if x > 1e-10 else 1)
    d = sc.sparse.diags(np.array(posinv(a.sum(0))).reshape([-1, ]), 0)
    lap = sc.sparse.eye(n_nodes) - d.dot(a.dot(d))
    return lap

def sparse_to_tuple(sparse_mx):
    """Convert sparse matrix to tuple representation."""
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape
    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)
    return sparse_mx

def graph_srw_transition_matrix(A):
    """Preprocessing of adjacency matrix for conversion to sparse transition matrix representation."""
    (I, J, V) = sp.find(A)
    n = A.shape[0]
    P = sp.lil_matrix((n, n))
    nnz = I.shape[0]
    row_start = 0
    while row_start < nnz:
        row = I[row_start]
        row_end = row_start
        while row_end < nnz and I[row_end] == row:
            row_end = row_end+1
        p = 1. / (row_end-row_start)
        for row_entry in range(row_start, row_end):
            P[row, J[row_entry]] = p
        row_start = row_end
    return P.tocsr()

def T_summer(T, tt):
    gamma = tt
    gam = 1.0 - gamma
    gamma_gam = 0.0
    sum = sp.csr_matrix((T.shape[0], T.shape[1]), dtype=np.float64)
    for i in range(10):
        if i != 0 :
            sum = sum + (sp.csr_matrix.power(T,i) * gamma)
            gamma = gamma * gamma

            # sum = sum + (sp.csr_matrix.power(T,i) * gamma_gam)
            # gamma_gam = gamma * float (pow(gam,i))
        else:
            sum = sum + sp.csr_matrix.power(T,i)
    return sum

def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def degree_normalize_adj(mx):
    """Row-normalize sparse matrix"""
    mx = mx.tolil()
    if mx[0, 0] == 0 :
        mx = mx + sp.eye(mx.shape[0])
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    # mx = mx.dot(r_mat_inv)
    mx = r_mat_inv.dot(mx)
    return mx
