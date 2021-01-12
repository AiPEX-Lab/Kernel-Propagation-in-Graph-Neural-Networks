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
TAUS = [7.5, 60, 70, 80]
TAUSS = [7.5, 60, 70, 80]
ORDER = 30
PROC = 'exact'
PROCC = 'exact'
ETA_MAX = 0.95
ETA_MIN = 0.80
NB_FILTERS = 1

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

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
    return adj_normalized

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

def isfloat(value):
  try:
    float(value)
    return True
  except ValueError:
    return False

def generate_UID(last):
    UID = last + 1
    return UID

def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot

def T_summer(T):
    gamma = 0.99
    sum = np.zeros((T.shape[0], T.shape[1]))
    for i in range(8):
        if i != 0 :
            sum = sum + np.dot(np.linalg.matrix_power(T, i), gamma)
            gamma = gamma * gamma
        else:
            sum = sum + np.linalg.matrix_power(T, i)
    return sum

def loadd_data(dataset_str):
    print('Loading {} dataset...'.format(dataset_str))
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("dataset/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file("dataset/ind.{}.test.index".format(dataset_str))
    test_idx_range = np.sort(test_idx_reorder)

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]

    features = sp.csr_matrix(features, dtype=np.float32)[list(np.where(np.sum(labels,1)==1)[0]),:]
    adj = sp.csr_matrix(adj, dtype=np.float32)[:,list(np.where(np.sum(labels,1)==1)[0])][list(np.where(np.sum(labels,1)==1)[0]),:]
    return adj, features, labels

def load_data(dataset):
    path = ("dataset/{}/".format(dataset))
    if dataset == 'citeseer':
        print('Loading {} dataset...'.format(dataset))

        idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset),
                                            dtype=np.dtype(str))
        features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
        labels = encode_onehot(idx_features_labels[:, -1])
        ints = [np.where(r==1)[0][0] for r in labels]
        # np.savetxt("citeseer_Labels.csv", ints, delimiter=",")

        idx = np.array(idx_features_labels[:, 0], dtype=object)
        idx_map = {j: i for i, j in enumerate(idx)}
        idx_set = list(idx_map.keys())
        last = 1110000
        UID_dict = {}
        for each in idx_set:
            if isfloat(each) == False:
                UID = generate_UID(last)
                UID_dict[UID] = each
                last = UID
            else:
                UID_dict[each] = each

        edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset),dtype=object)

        for i in range(edges_unordered.shape[0]):
            for j in range(edges_unordered[i].shape[0]):
                edges_unordered[i][j] = edges_unordered[i][j].decode("utf-8")
                for key, value in UID_dict.items():
                    if value == edges_unordered[i][j]:
                        edges_unordered[i][j] = str(key)

        isolated_nodes_list = []

        for ii in range(edges_unordered.shape[0]):
            for jj in range(edges_unordered[ii].shape[0]):
                if isfloat(edges_unordered[ii][jj]) == False:
                    isolated_nodes_list.append(ii)

        edges_unordered = np.delete(edges_unordered, isolated_nodes_list, axis=0)
        edges_unordered = edges_unordered.astype(dtype=np.int32)
        counter = 0
        for key, value in UID_dict.items():
            UID_dict[key] = counter
            counter += 1

        UID_dict = {int(k):int(v) for k,v in UID_dict.items()}

        flattened_edges = edges_unordered.flatten()

        for x in flattened_edges:
            if isinstance(x, type(None)):
                print (x)
            else:
                pass

        edges = np.array(list(map(UID_dict.get, list(flattened_edges))))
        edges = edges.reshape(edges_unordered.shape)
        none_type_list = []
        for y in range(edges.shape[0]):
            for k in range(edges.shape[1]):
                if isinstance(edges[y][k], type(None)):
                    none_type_list.append(y)
                else:
                    pass
        edges = np.delete(edges, none_type_list, axis=0)
        edges = edges.astype(dtype=np.int32)

        adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),shape=(labels.shape[0], labels.shape[0]), dtype=np.float32)
        print ((adj.todense()).shape)

        idx_train = range(500)
        idx_val = range(500, 900)
        idx_test = range(500, 900)
    elif dataset == 'cora':
        print('Loading {} dataset...'.format(dataset))

        idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset), dtype=np.dtype(str))
        features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
        labels = encode_onehot(idx_features_labels[:, -1])
        ints = [np.where(r==1)[0][0] for r in labels]
        # np.savetxt("cora_Labels.csv", ints, delimiter=",")

        # build graph
        idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
        idx_map = {j: i for i, j in enumerate(idx)}
        edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset),
                                        dtype=np.int32)
        edges = np.array(list(map(idx_map.get, edges_unordered.flatten())), dtype=np.int32).reshape(edges_unordered.shape)

        adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])), shape=(labels.shape[0], labels.shape[0]), dtype=np.float32)
        print ((adj.todense()).shape)
        idx_train = range(400)
        idx_val = range(400, 900)
        idx_test = range(400, 900)
    else:
        adj, features, labels = loadd_data('pubmed')
        idx_train = range(2958)
        idx_val = range(2958, 5000)
        idx_test = range(18717, 19717)
        # T_Power = np.load('pubmed_T_SUM.npy')

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = adj + sp.eye(adj.shape[0])
    adj = normalize_adj(adj)

    A = adj.todense()  # Adjacency Matrix (A)
    T = graph_srw_transition_matrix(A)  # Graph Transition Matrix
    T = T.todense()
    T_Power = T_summer(T)  # Two-hop Graph Transition ## uncomment if anything other than pubmed
    T_Power = normalize_adj(T_Power)

    features_1 = torch.FloatTensor(T_Power.todense())
    features_2 = torch.FloatTensor(features.todense())
    features = torch.cat((features_1, features_2), dim=1)
    adj = torch.FloatTensor(adj.todense())
    labels = torch.LongTensor(np.where(labels)[1])

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_val, idx_test

def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index

def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)
