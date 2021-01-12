import torch
import torch_geometric.transforms as T
from ogb.nodeproppred import PygNodePropPredDataset

dataset = PygNodePropPredDataset(name="ogbn-arxiv", transform=None)
data = dataset[0]

split_idx = dataset.get_idx_split()
test_idx = split_idx['test']

edge_idx = data.edge_index
test_idx_set = set(test_idx.numpy())#.flatten()

# remove edges of test nodes.
new_edge = [[], []]
for i in range(edge_idx.shape[1]):
    if edge_idx[0, i].item() not in test_idx_set and edge_idx[1, i].item() not in test_idx_set:
        new_edge[0].append(edge_idx[0, i].item())
        new_edge[1].append(edge_idx[1, i].item())

data.edge_index = torch.tensor(new_edge)

data_full = data.clone()
data_full.edge_index = edge_idx

tosparse = T.ToSparseTensor()
data = tosparse(data)
data_full = tosparse(data_full)
data.adj_t = data.adj_t.to_symmetric()
data_full.adj_t = data_full.adj_t.to_symmetric()
num_classes = dataset.num_classes

if __name__ == '__main__':
    print(id(data), id(data_full))