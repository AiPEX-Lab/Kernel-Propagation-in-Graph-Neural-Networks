# Kernel-Propagation-in-Graph-Neural-Networks (KP-GNN)

This is the official repository of the paper titled, "Node Classification using Kernel Propagation in Graph Neural Networks" which is currently under review in the journal Expert Systems with Applications.

## To generate the results as reported in the paper for the large-scale real world OGB networks, please follow the following instructions:

1. In order to be able to run the scripts which train, validate and test on the datasets used in the paper, you need to use a Dockerfile. It is recommended to use a computer which has an Nvidia GPU since the training, testing and validation can consume a lot of time if run on CPU.
2. Please download/clone this repository onto the computer you will be using for re-running the experiments and name the folder as "kpgnn". Now proceed to build the docker image by running the following command, "sudo docker build -t kpgnn ."
3. Once the docker image has been built, run the following command to train, validate and test a KP-GNN (where GNN can be a GCN, GAT or JK Net). The following command will train a KP-GCN on the OGB-products network with the higher-order structural features concatenated with node features. Please note that running the following command will download the required dataset before starting the training.\
"sudo docker run -it --rm --gpus all -v $HOME/aipex/kpgnn:/app kpgnn python3 gnn_products.py --device 0 --model 'gcn' --num_layers 2 --hidden_channels 128 --mode 'cat' --epochs 200 --runs 3"\
You can modify the hyper-parameters according to your network or task. But in order to generate the results as shown in the paper you will have to run the command above. With all other parameters remaining the same, you can change the KP-GCN model to KP-GAT or KP-JK Net by replacing 'gcn' with 'gat' or 'jk'.
4. Alternatively, it is fairly easy to switch from training on the OGB-products network to OGB-proteins or OGB-arxiv by running the following commands respectively for KP-GCN.\
For OGB-arxiv:\
"sudo docker run -it --rm --gpus all -v $HOME/aipex/kpgnn:/app kpgnn python3 gnn_arxiv.py --device 0 --model 'gcn' --num_layers 2 --hidden_channels 128 --mode 'cat' --epochs 200 --runs 3"\
For OGB-proteins:\
"sudo docker run -it --rm --gpus all -v $HOME/aipex/kpgnn:/app kpgnn python3 gnn_proteins.py --device 0 --model 'gcn' --num_layers 2 --hidden_channels 128 --mode 'cat' --epochs 200 --runs 3"\
As before, you can replace KP-GCN with KP-GAT or KP-JK Net by replacing 'gcn' with 'gat' or 'jk'.\

## To generate the results as reported in the paper for the small-scale real world networks such as Cora, Citeseer and PubMed, please follow the following instructions:

1. In order to replicate the results from the paper, you will need to use a Dockerfile as before.
2. From within the kpgnn folder, cd into "small_scale". Proceed to build the docker image by running the following command, "sudo docker build -t small_scale ."
3. Once the docker image has been built, run the following command to train, validate and test a KP-GNN (where GNN can be a GCN) or AKP-GCN. The following command will train a 2 layer KP-GCN on the cora network with the higher-order structural features concatenated with node features.\
"sudo docker run -it --rm --gpus all -v $HOME/aipex/kpgnn/small_scale:/app small_scale python3 train.py --dataset 'cora' --epochs 50 --hidden 64"\
You can modify the hyper-parameters according to your network or task. But in order to generate the results as shown in the paper you will have to run the command above. With all other parameters remaining the same, you can change the KP-GCN model to AKP-GCN by commenting line 35 in the script "models.py" and uncommenting line 31. This replaces KP-GCN with AKP-GCN which uses the proposed attention mechanism.
4. Alternatively, it is fairly easy to switch from training on the Cora network to PubMed by running the following commands respectively for KP-GCN.\
For PubMed:\
"sudo docker run -it --rm --gpus all -v $HOME/aipex/kpgnn/small_scale:/app small_scale python3 train.py --dataset 'pubmed' --epochs 50 --hidden 64"\

## Acknowledgements:

We would like to acknowledge the following repositories for allowing us to use the official versions of the GNNs and some helper functions implemented in this paper:

1. https://github.com/snap-stanford/ogb/tree/master/examples/nodeproppred
2. https://github.com/PetarV-/GAT
3. https://github.com/rusty1s/pytorch_geometric/blob/master/examples/gcn.py
4. https://github.com/tkipf/gcn







