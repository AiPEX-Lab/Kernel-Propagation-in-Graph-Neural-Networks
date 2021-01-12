# Kernel-Propagation-in-Graph-Neural-Networks (KP-GNN)

This is the official repository of the paper titled, "Node Classification using Kernel Propagation in Graph Neural Networks" which is currently under review in the journal Expert Systems with Applications.

In order to generate the results as reported in the paper, please follow the following instructions:

1. In order to be able to run the scripts which train, validate and test on the datasets used in the paper, you need to use a Dockerfile. It is recommended to use a computer which has an Nvidia GPU since the training, testing and validation can consume a lot of time if run on CPU
2. Please download/clone this repository onto the computer you will be using for re-running the experiments and name the folder as "kpgnn". Now proceed to build the docker image by running the following command, "sudo docker build -t kpgnn ."
3. Once the docker image has been built, run the following command to train, validate and test a KP-GNN (where GNN can be a GCN, GAT or JK Net). The following command will train a KP-GCN on the OGB-products network with the higher-order structural features concatenated with node features.
"sudo docker run -it --rm --gpus all -v $HOME/aipex/kpgnn:/app kpgnn python3 gnn_products.py --device 0 --model 'gcn' --num_layers 2 --hidden_channels 128 --mode 'cat' --epochs 200 --runs 3"
You can modify the hyper-parameters according to your need. But in order to generate the results as shown in the paper you will have to run the command above. With all other parameters remaining the same, you can change the KP-GCN model to KP-GAT or KP-JK Net by replacing 'gcn' with 'gat' or 'jk'.
4. Alternatively, it is fairly easy to switch from training on the OGB-products network to OGB-proteins or OGB-arxiv by running the following commands respectively for KP-GCN.
For OGB-arxiv:
"sudo docker run -it --rm --gpus all -v $HOME/aipex/kpgnn:/app kpgnn python3 gnn_arxiv.py --device 0 --model 'gcn' --num_layers 2 --hidden_channels 128 --mode 'cat' --epochs 200 --runs 3"
For OGB-proteins:
"sudo docker run -it --rm --gpus all -v $HOME/aipex/kpgnn:/app kpgnn python3 gnn_proteins.py --device 0 --model 'gcn' --num_layers 2 --hidden_channels 128 --mode 'cat' --epochs 200 --runs 3"
As before, you can replace KP-GCN with KP-GAT or KP-JK Net by replacing 'gcn' with 'gat' or 'jk'.




