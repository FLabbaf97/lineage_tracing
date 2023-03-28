conda create -n lpbs_bread python=3.7
conda activate lpbs_bread
<!-- see https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html -->
pip install pyg-lib torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-1.12.0+cu102.html
pip install -r requirements.txt
pip install -e .
<!-- then manually install jupyter, depending on your setup -->