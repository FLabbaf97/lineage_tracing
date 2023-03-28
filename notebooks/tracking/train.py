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


datadir = Path('../../data/generated/tracking/assgraphs/')
dataset_filepaths = {
	'colony_1234__dt_12345678__t_all': list(sorted(glob(str(datadir / 'colony00[1-4]_segmentation__assgraph__dt_00[1-8]__*.pt')))),
	'colony_5__dt_12345678__t_all': list(sorted(glob(str(datadir / 'colony005_segmentation__assgraph__dt_00[1-8]__*.pt')))),
	'colony_1234__dt_1234__t_all': list(sorted(glob(str(datadir / 'colony00[1-4]_segmentation__assgraph__dt_00[1-4]__*.pt')))),
	'colony_5__dt_1234__t_all': list(sorted(glob(str(datadir / 'colony005_segmentation__assgraph__dt_00[1-4]__*.pt')))),
}

if __name__ == '__main__':
	parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
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

	args = parser.parse_args()

	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	resultdir = Path(f'../../data/generated/tracking/models/{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
	os.makedirs(resultdir, exist_ok=True)

	print('-- arguments --')
	pprint(vars(args))
	with open(resultdir / 'metadata.json', 'w') as file:
		json.dump(vars(args), file)

	print(f'device : {device}')

	# filter out files that are too large
	dataset = AssignmentDataset([ filepath for filepath in dataset_filepaths[args.dataset] if os.stat(filepath).st_size/2**20 < args.filter_file_mb ])
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
			# EpochScoring(scoring=accuracy_assignment, lower_is_better=False, name='valid_acc_ass'),
		],
	)

	os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'

	torch.cuda.empty_cache()
	print('-- starting training --')
	net.fit(dataset, y=None)