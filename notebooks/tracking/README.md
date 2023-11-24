1. generate cell graphs

python -m bread.algo.tracking.build_cellgraphs --segmentations data/data_from_LPBS/colony005_segmentation.h5 --out data/generated/tracking/cellgraphs --nn-threshold=12 --scale-length=1 --scale-time=1

2. generate assignment graphs

python -m bread.algo.tracking.build_assgraphs --cellgraphs "data/generated/tracking/cellgraphs/*/" --out "data/generated/tracking/assgraphs" --framediff-min=1 --framediff-max=4

3. perform cv hyperparameter search

edit gridsearchcv.py and edit the param search and output csv file. models are placed in data/generated/tracking/models/gridsearchcv ...
use results_gridsearch.ipynb to visualize results (hyperparameter plots)

4. train

use python train.py --help to display arguments

launch training session with :

python train.py --max-epochs=40 --gamma=0.96 --step-size=512

an example log is provided in trainlog.html

5. evaluate

see where the gcn is wrong in compare.ipynb