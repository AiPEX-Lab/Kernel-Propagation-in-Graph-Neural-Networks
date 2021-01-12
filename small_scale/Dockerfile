# Specify CUDA version and cudnn as well as OS version
FROM nvcr.io/nvidia/cuda:10.2-cudnn7-devel-ubuntu18.04
# Set the directory
COPY . /app/
WORKDIR /app
# Updating Ubuntu packages
RUN DEBIAN_FRONTEND=noninteractive apt-get -qq update \
 && DEBIAN_FRONTEND=noninteractive apt-get -qqy install python3 python3-pip ffmpeg git less nano libsm6 libxext6 libxrender-dev wget bzip2 \
 && rm -rf /var/lib/apt/lists/*
# Adding pytorch
RUN DEBIAN_FRONTEND=noninteractive pip3 install --upgrade pip
RUN DEBIAN_FRONTEND=noninteractive pip3 install scipy matplotlib numpy keras==2.3.1 networkx scikit-learn tensorflow==1.15
RUN DEBIAN_FRONTEND=noninteractive pip3 install torch==1.6.0 torchvision
RUN DEBIAN_FRONTEND=noninteractive pip3 install torch-scatter -f https://pytorch-geometric.com/whl/torch-1.6.0+cu102.html
RUN DEBIAN_FRONTEND=noninteractive pip3 install torch-sparse -f https://pytorch-geometric.com/whl/torch-1.6.0+cu102.html
RUN DEBIAN_FRONTEND=noninteractive pip3 install torch-cluster -f https://pytorch-geometric.com/whl/torch-1.6.0+cu102.html
RUN DEBIAN_FRONTEND=noninteractive pip3 install torch-spline-conv -f https://pytorch-geometric.com/whl/torch-1.6.0+cu102.html
RUN DEBIAN_FRONTEND=noninteractive pip3 install torch-geometric
RUN DEBIAN_FRONTEND=noninteractive pip3 install ogb
RUN DEBIAN_FRONTEND=noninteractive apt-get -qq update && yes|apt-get upgrade
