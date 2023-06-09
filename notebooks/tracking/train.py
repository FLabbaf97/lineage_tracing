from typing import Optional
from bread.data import *
from bread.algo.tracking import AssignmentDataset, GNNTracker, AssignmentClassifier, GraphLoader, accuracy_assignment

from glob import glob
from pathlib import Path
import datetime, json, argparse
import torch, os
from pprint import pprint

from skorch.dataset import ValidSplit
from skorch.callbacks import LRScheduler, Checkpoint, ProgressBar, Checkpoint, EpochScoring

from trainutils import SaveHyperParams, seed

import wandb
from skorch.callbacks import WandbLogger
WANDB_API_KEY="e0f887ce4be7bebfe48930ffcff4027f49b02425"
os.environ['WANDB_API_KEY'] = WANDB_API_KEY
os.environ['WANDB_CONSOLE'] = "off"
os.environ['WANDB_JOB_TYPE'] = 'features_test'


if __name__ == '__main__':

	# just a placeholder
	data_dir = Path('../../data/generated/tracking/assgraphs/') 
	dataset_filepaths = {
		'colony_1234__dt_12345678__t_all': list(sorted(glob(str(data_dir / 'colony00[1-4]_segmentation__assgraph__dt_00[1-8]__*.pt')))),
		'colony_5__dt_12345678__t_all': list(sorted(glob(str(data_dir / 'colony005_segmentation__assgraph__dt_00[1-8]__*.pt')))),
		'colony_1234__dt_1234__t_all': list(sorted(glob(str(data_dir / 'colony00[1-4]_segmentation__assgraph__dt_00[1-4]__*.pt')))),
		'colony_5__dt_1234__t_all': list(sorted(glob(str(data_dir / 'colony005_segmentation__assgraph__dt_00[1-4]__*.pt')))),
		'colony_12345__dt_1234__t_all': list(sorted(glob(str(data_dir / 'colony00[1-5]_segmentation__assgraph__dt_00[1-4]__*.pt')))),
		'colony_012345__dt_1234__t_all': list(sorted(glob(str(data_dir / 'colony00[0-5]_segmentation__assgraph__dt_00[1-4]__*.pt')))),
		'colony_0123__dt_1234__t_all': list(sorted(glob(str(data_dir / 'colony00[0-3]_segmentation__assgraph__dt_00[1-4]__*.pt')))),
	}
	
	parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument('--assgraphs', dest='ass_graphs', required=True, type=str, help='folder path to the assignment graphs (be careful, should be a directory not file.)')
	parser.add_argument('--result-dir', dest='result_dir', default=f'../../data/generated/tracking/models/', type=str, help='folder path to save the results')
	parser.add_argument('--dataset', dest='dataset', type=str, default='colony_1234__dt_1234__t_all', choices=list(dataset_filepaths.keys()), help='training data name')
	parser.add_argument('--filter-file-mb', dest='filter_file_mb', type=float, default=10, help='filter training file sizes exceeding `filter_file_mb` MiB')
	parser.add_argument('--dropout-rate', dest='dropout_rate', type=float, default=1e-7, help='dropout rate')
	parser.add_argument('--encoder-hidden-channels', dest='encoder_hidden_channels', default=120, type=int, help='encoder hidden channels')
	parser.add_argument('--encoder-num-layers', dest='encoder_num_layers', default=4, type=int, help='encoder num layers')
	parser.add_argument('--conv-hidden-channels', dest='conv_hidden_channels', default=120, type=int, help='convhiddenchannels')
	parser.add_argument('--conv-num-layers', dest='conv_num_layers', default=5, type=int, help='conv num layers')
	parser.add_argument('--max-epochs', dest='max_epochs', default=20, type=int, help='max epochs')
	parser.add_argument('--lr', dest='lr', default=1e-3, type=float, help='base lr')
	parser.add_argument('--weight-decay', dest='weight_decay', default=0.01, type=float, help='weight decay (L2 regularization)')
	parser.add_argument('--step-size', dest='step_size', default=512, type=int, help='steplr step size in number of batches')
	parser.add_argument('--gamma', dest='gamma', default=0.5, type=float, help='steplr gamma')
	parser.add_argument('--cv', dest='cv', default=5, type=Optional[int], help='number of cv folds')
	parser.add_argument('--patience', dest='patience', default=6, type=Optional[int], help='number of epoch to wait for improvement before early stopping')


	args = parser.parse_args()
	data_dir = Path(args.ass_graphs)
	dataset_filepaths = {
		'colony_1234__dt_12345678__t_all': list(sorted(glob(str(data_dir / 'colony00[1-4]_segmentation__assgraph__dt_00[1-8]__*.pt')))),
		'colony_5__dt_12345678__t_all': list(sorted(glob(str(data_dir / 'colony005_segmentation__assgraph__dt_00[1-8]__*.pt')))),
		'colony_1234__dt_1234__t_all': list(sorted(glob(str(data_dir / 'colony00[1-4]_segmentation__assgraph__dt_00[1-4]__*.pt')))),
		'colony_5__dt_1234__t_all': list(sorted(glob(str(data_dir / 'colony005_segmentation__assgraph__dt_00[1-4]__*.pt')))),
		'colony_12345__dt_1234__t_all': list(sorted(glob(str(data_dir / 'colony00[1-5]_segmentation__assgraph__dt_00[1-4]__*.pt')))),
		'colony_012345__dt_1234__t_all': list(sorted(glob(str(data_dir / 'colony00[0-5]_segmentation__assgraph__dt_00[1-4]__*.pt')))),
		'colony_0123__dt_1234__t_all': list(sorted(glob(str(data_dir / 'colony00[0-3]_segmentation__assgraph__dt_00[1-4]__*.pt')))),
	}
	dataset_regex = {
		'colony_1234__dt_12345678__t_all': "^colony00(1|2|3|4)_segmentation__assgraph__dt_00(1|2|3|4|5|6|7|8)__.*pt$",
		'colony_5__dt_12345678__t_all': "^colony005_segmentation__assgraph__dt_00(1|2|3|4|5|6|7|8)__.*pt$",
		'colony_1234__dt_1234__t_all': "^colony00(1|2|3|4)_segmentation__assgraph__dt_00(1|2|3|4)__.*pt$",
		'colony_5__dt_1234__t_all': "^colony005_segmentation__assgraph__dt_00(1|2|3|4)__.*pt$",
		'colony_12345__dt_1234__t_all': "^colony00(1|2|3|4|5)_segmentation__assgraph__dt_00(1|2|3|4)__.*pt$",
		'colony_012345__dt_1234__t_all': "^colony00(0|1|2|3|4|5)_segmentation__assgraph__dt_00(1|2|3|4)__.*pt$",
		'colony_0123__dt_1234__t_all': "^colony00(0|1|2|3)_segmentation__assgraph__dt_00(1|2|3|4)__.*pt$",
	}
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	resultdir = Path(f'{args.result_dir}/{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
	print("-- result directory --")
	print(resultdir)
	os.makedirs(resultdir, exist_ok=True)

	print('-- train arguments --')
	print(vars(args))
	with open(resultdir / 'metadata.json', 'w') as file:
		json.dump(vars(args), file)

	print(f'device : {device}')

	# filter out files that are too large
	file_array = []
	import re
	regex = dataset_regex[args.dataset]
	for filename in os.listdir(args.ass_graphs):
		if re.search(regex, filename):
			file_array.append(os.path.join(args.ass_graphs, filename))
	dataset = AssignmentDataset(file_array)
	# dataset = AssignmentDataset([ filepath for filepath in dataset_filepaths[args.dataset] if os.stat(filepath).st_size/2**20 < args.filter_file_mb ])
	print('-- training dataset --')
	print(dataset)

	seed(42)
	dataset = dataset.shuffle()

	cv = None
	if args.cv is not None:
		cv = ValidSplit(
			# This generates one train and validation dataset for the entire fit
			# Use sklearn cross validation to properly score a model
			cv=args.cv,
			stratified=False,  # Since we are using y=None
		)
	
	# Create a wandb Run
	# Alternative: Create a wandb Run without a W&B account
	wandb_run = wandb.init()
	# set wandb config
	config = {
		'dataset': args.dataset,
		'filter_file_mb': args.filter_file_mb,
		'dropout_rate': args.dropout_rate,
		'encoder_hidden_channels': args.encoder_hidden_channels,
		'encoder_num_layers': args.encoder_num_layers,
		'conv_hidden_channels': args.conv_hidden_channels,
		'conv_num_layers': args.conv_num_layers,
		'max_epochs': args.max_epochs,
		'lr': args.lr,
		'weight_decay': args.weight_decay,
		'step_size': args.step_size,
		'gamma': args.gamma,
		'cv': args.cv,
		'patience': args.patience,
	}
	# Log hyper-parameters 
	wandb_run.config.update(config)
	net = AssignmentClassifier(
		GNNTracker,
		module__num_node_attr=len(dataset.node_attr),
		module__num_edge_attr=len(dataset.edge_attr),
		module__dropout_rate=args.dropout_rate,
		module__encoder_hidden_channels=args.encoder_hidden_channels,
		module__encoder_num_layers=args.encoder_num_layers,
		module__conv_hidden_channels=args.conv_hidden_channels,
		module__conv_num_layers=args.conv_num_layers,
		module__num_classes=1,  # fixed, we do binary classification

		max_epochs=args.max_epochs,
		device=device,

		criterion=torch.nn.BCEWithLogitsLoss(
			# attribute more weight to the y == 1 samples, because they are more rare
			pos_weight=torch.tensor((dataset.num_class_positive + dataset.num_class_negative)/dataset.num_class_positive)
		),

		optimizer=torch.optim.Adam,
		optimizer__lr=args.lr,
		optimizer__weight_decay=args.weight_decay,  # L2 regularization

		iterator_train=GraphLoader,
		iterator_valid=GraphLoader,
		iterator_train__shuffle=True,
		iterator_valid__shuffle=False,
		batch_size=1,

		train_split=cv,

		callbacks=[
			LRScheduler(policy='StepLR', step_every='batch', step_size=args.step_size, gamma=args.gamma),
			Checkpoint(monitor='valid_loss_best', dirname=resultdir, f_pickle='pickle.pkl'),
			SaveHyperParams(dirname=resultdir),
			ProgressBar(detect_notebook=False),
			WandbLogger(wandb_run, save_model=True),
			# EpochScoring(scoring=accuracy_assignment, lower_is_better=False, name='valid_acc_ass'),
		],
	)

	os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'

	torch.cuda.empty_cache()
	print('-- starting training --')
	net.fit(dataset, y=None)