conda create -n lpbs_bread python=3.9
conda activate lpbs_bread
<!-- see https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html -->
<!-- pip install pyg-lib torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-1.12.0+cu102.html -->
pip install torch_geometric
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.0.0+cpu.html
pip install -r requirements.txt
pip install -e .
<!-- then manually install jupyter, depending on your setup -->


-----------------------------
cd bread
source activate paper2022 && cd bread && pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.0.0+cu118.html && pip install torch_geometric && pip install -r requirements.txt && pip install -e .

pip install -r requirements.txt
pip install -e .

python notebooks/testing_nicoles/train_p.py --config config_fourier_and_f10_NL && python notebooks/testing_nicoles/train_p.py --config config_fourier_and_f10_NL && python notebooks/testing_nicoles/train_p.py --config config_fourier_and_f10_NL && python notebooks/testing_nicoles/train_p.py --config config_fourier_and_f10_NL && python notebooks/testing_nicoles/train_p.py --config config_fourier_and_f10_NL 

python -m bread.algo.tracking.build_cellgraphs --config config_fourier50_and_f10_NL && python -m bread.algo.tracking.build_assgraphs --config config_fourier50_and_f10_NL && python bread/notebooks/testing_nicoles/train_p.py --config config_fourier50_and_f10_NL