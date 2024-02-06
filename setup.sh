#!/bin/bash

user="${1}"
if [[ "$user" == "" ]]; then
    echo -e "\e[93;1mMust specify username\e[0m"
    exit 1
fi

# Miniconda3

wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
echo -e "\e[91;1m│ Install conda to /home/$user/miniconda3 │\e[0m"
bash Miniconda3-latest-Linux-x86_64.sh
source ~/.bashrc


conda init
conda env create -n bread python=3.11
conda activate bread
pip install --no-cache-dir -r requirements.txt
# install other packages

# install torch
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia

# install torch geometric
conda install pyg -c pyg
# pip install torch_geometric
# Optional dependencies for torch_geometric:
#pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.0.0+cu117.html

# install the package
pip install -e .
conda deactivate

