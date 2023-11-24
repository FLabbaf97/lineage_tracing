# %%
from bread.data import *
from bread.algo.tracking import AssignmentDataset, GNNTracker, AssignmentClassifier, GraphLoader, accuracy_assignment

from glob import glob
from pathlib import Path
import datetime
import torch, os
import pandas as pd
from tqdm import tqdm

from sklearn.model_selection import KFold
from skorch.dataset import ValidSplit
from skorch.callbacks import LRScheduler, Checkpoint, ProgressBar, Checkpoint, EpochScoring
from sklearn.model_selection import ParameterGrid

from trainutils import SaveHyperParams, seed


datadir = Path('../../data/generated/tracking/assgraphs/')
dataset_filepaths = {
	'colony_1234__dt_12345678__t_all': list(sorted(glob(str(datadir / 'colony00[1-4]_segmentation__assgraph__dt_00[1-8]__*.pt')))),
	'colony_5__dt_12345678__t_all': list(sorted(glob(str(datadir / 'colony005_segmentation__assgraph__dt_00[1-8]__*.pt')))),
	'colony_1234__dt_1234__t_all': list(sorted(glob(str(datadir / 'colony00[1-4]_segmentation__assgraph__dt_00[1-4]__*.pt')))),
	'colony_5__dt_1234__t_all': list(sorted(glob(str(datadir / 'colony005_segmentation__assgraph__dt_00[1-4]__*.pt')))),
}
dataset_name_train = 'colony_1234__dt_1234__t_all'


# %%
# filter files under 10MB
dataset = AssignmentDataset([ filepath for filepath in dataset_filepaths[dataset_name_train] if os.stat(filepath).st_size/2**20 < 10 ])
seed(42)
# take a subset of 512 datapoints
dataset = dataset.shuffle()[:512]
print('-- training dataset --')
print(dataset_name_train, dataset)


# %%
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

net = AssignmentClassifier(
	GNNTracker,
	module__num_node_attr=len(dataset.node_attr),
	module__num_edge_attr=len(dataset.edge_attr),
	module__dropout_rate=0.0,
	module__encoder_hidden_channels=120,
	module__encoder_num_layers=4,
	module__conv_hidden_channels=120,
	module__conv_num_layers=5,  # 5 convolutions should span a good chunk of the colony
	module__num_classes=1,  # fixed, we do binary classification

	max_epochs=10,
	device=device,

	criterion=torch.nn.BCEWithLogitsLoss(
		# attribute more weight to the y == 1 samples, because they are more rare
		pos_weight=torch.tensor((dataset.num_class_positive + dataset.num_class_negative)/dataset.num_class_positive)
	),

	optimizer=torch.optim.Adam,
	optimizer__lr=1e-3,
	optimizer__weight_decay=1e-6,  # L2 regularization

	iterator_train=GraphLoader,
	iterator_valid=GraphLoader,
	iterator_train__shuffle=True,
	batch_size=1,

	callbacks=[
		# multiply lr by 0.5 every 4 steps
		LRScheduler(policy='StepLR', step_every='batch', step_size=512, gamma=0.5),
		Checkpoint(monitor='valid_loss_best', load_best=True),
		SaveHyperParams(),
		# EpochScoring(scoring=accuracy_assignment, lower_is_better=False, name='valid_acc_ass'),
	],
)

# %%
# experiment (A)
# grid = ParameterGrid({
# 	'module__encoder_hidden_channels': [100, 60, 20],
# 	'module__encoder_num_layers': [8, 5, 3],
# 	'module__conv_hidden_channels': [120, 100, 60],
# 	'module__conv_num_layers': [10, 8, 4],
# })
# experiment (B)
# grid = ParameterGrid({
# 	'module__dropout_rate': [1e-2, 1e-4, 1e-6],
# 	'optimizer__weight_decay': [1e-2, 1e-4, 1e-6, 0.0],
# })
# experiment (C)
grid = ParameterGrid({
	'module__dropout_rate': [1e-5, 1e-7, 0.0],
	'optimizer__weight_decay': [1e-3, 1e-5, 0.0],
	'module__encoder_hidden_channels': [140, 110, 80],
})

results = []

resultroot = Path(f'../../data/generated/tracking/models/gridsearchcv {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')

for params in tqdm(grid):
	params_str = ' '.join([f'{k.replace("module__", "")}={v}' for k, v in params.items()])
	tqdm.write(params_str)

	kfold = KFold(n_splits=3, shuffle=True, random_state=42)
	for kth, (idx_train, idx_val) in enumerate(kfold.split(dataset)):
		tqdm.write(f'split {kth+1}/{kfold.n_splits}')

		resultdir = resultroot / params_str / f'split {kth+1}'

		seed(42)
		net.set_params(
			train_split=lambda dataset: (dataset[idx_train], dataset[idx_val]),
			verbose=1,
			callbacks__Checkpoint__dirname=resultdir,
			callbacks__SaveHyperParams__dirname=resultdir,
			**params,
		)
		
		os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
		torch.cuda.empty_cache()
		net.fit(dataset, y=None)

		d = { k.replace('module__', ''): v for k, v in params.items() }
		d.update({
			'valid_loss': min(net.history[:, 'valid_loss']),
			'valid_err': 1-min(net.history[:, 'valid_acc'])
		})
		results.append(d)

# %%
df = pd.DataFrame(results)
df.to_csv('results_C.csv')

# %%
df\
	.groupby(list(df.drop(columns='valid_loss').columns))\
	[['valid_loss']]\
	.agg(['mean', 'std'])\
	.reset_index()\
	.nsmallest(20, ('valid_loss', 'mean'))
