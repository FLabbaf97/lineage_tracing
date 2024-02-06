from typing import Optional
from bread.data import *
from bread.algo.tracking import AssignmentDataset, GNNTracker, AssignmentClassifier, GraphLoader, accuracy_assignment, f1_assignment

from glob import glob
from pathlib import Path
import datetime, json, argparse
import torch, os
from pprint import pprint
from sklearn.metrics import confusion_matrix


from skorch.dataset import ValidSplit
from skorch.callbacks import LRScheduler, Checkpoint, ProgressBar, EpochScoring, WandbLogger, EarlyStopping

from trainutils import SaveHyperParams, seed

from importlib import import_module

import tqdm

	
def test_pipeline(net, test_array, result_dir, data_name, algorithm='hungarian'):
	results = {}
	results['gcn'] = {
        't1': [],
        't2': [],
		'colony':[],
        'confusion': [],
		'num_cells1': [],
		'num_cells2': [],
    }

	for file in tqdm.tqdm(test_array):
		# read graph from .pt file
		graph = torch.load(file)
		yhat = net.predict_assignment(graph, algorithm).flatten()
		y = graph.y.squeeze().cpu().numpy()

		results['gcn']['confusion'].append(confusion_matrix(y, yhat))
		idt1_pattern = r'__(\d{3})_to'
		idt2_pattern = r'_to_(\d{3})'
		dt_pattern = r'_dt_(\d{3})'
		# data_name is from first underline to where we have "_segmentation"
		data_name_pattern = r'(\w+)_segmentation'
		
		idt1 = int(re.findall(idt1_pattern, file)[0])
		idt2 = int(re.findall(idt2_pattern, file)[0])
		dt = int(re.findall(dt_pattern, file)[0])
		data_name = re.findall(data_name_pattern, file)[0]
		

		results['gcn']['t1'].append(idt1)
		results['gcn']['t2'].append(idt2)
		results['gcn']['colony'].append(data_name)
		results['gcn']['num_cells1'].append(len(set([list(graph.cell_ids)[i][0] for i in range(len(graph.cell_ids))])))
		results['gcn']['num_cells2'].append(len(set([list(graph.cell_ids)[i][1] for i in range(len(graph.cell_ids))])))

		
	
	import pandas as pd
	res = pd.DataFrame(results['gcn'])
	res['method'] = 'gcn'
    
	# num_cells = [len(seg.cell_ids(idt)) for idt in range(len(seg))]

	res['tp'] = res['confusion'].map(lambda c: c[1, 1])
	res['fp'] = res['confusion'].map(lambda c: c[0, 1])
	res['tn'] = res['confusion'].map(lambda c: c[0, 0])
	res['fn'] = res['confusion'].map(lambda c: c[1, 0])
	res['acc'] = (res['tp'] + res['tn']) / \
		(res['tp'] + res['fp'] + res['tn'] + res['fn'])
	res['f1'] = 2*res['tp'] / (2*res['tp'] + res['fp'] + res['fn'])
	# res['num_cells1'] = res['t1'].map(lambda t: num_cells[int(t)])
	# res['num_cells2'] = res['t2'].map(lambda t: num_cells[int(t)])
	res['timediff'] = 5 * (res['t2'] - res['t1'])
	res.drop(columns='confusion', inplace=True)

	# save results
	res.to_csv(resultdir/f'result_{data_name}.csv', index=False)
	res_sum = res.groupby(['timediff', 'colony'])[
	['f1', 'tp', 'fp', 'fn', 'tn', 'num_cells1', 'num_cells2']].sum()
	res_sum['precision'] = res_sum['tp'] / (res_sum['tp'] + res_sum['fp'])
	res_sum['recall'] = res_sum['tp'] / (res_sum['tp'] + res_sum['fn'])
	res_sum['f1'] = 2*res_sum['tp'] / \
		(2*res_sum['tp'] + res_sum['fp'] + res_sum['fn'])
	res_sum[['f1', 'precision', 'recall']]
	print('result on test data: ', data_name)
	res_sum.to_csv(resultdir/f'result_sum_{data_name}.csv')
	print(res_sum)
	

if __name__ == '__main__':
	parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument('--config', dest='config', required=True, type=str, help='config file')
	parser.add_argument('--model', dest='model', required=True, type=str, help='model directory')
	parser.add_argument('--algo', dest='algo', default='hungarian', type=str, help='postprocessing algorithm')
 

	args = parser.parse_args()
	config = import_module("bread.config." + args.config).configuration
	algo = args.algo
	
	pretty_config = json.dumps(config, indent=4)
	print(pretty_config)

	# train_config = config.get('train_config')

	test_config = config.get('test_config')

	model_dir = Path(args.model)

	data_dir = Path(test_config["ass_graphs"])

	dataset_regex = {
		'colony_1234__dt_12345678__t_all': "^colony00(1|2|3|4)_segmentation__assgraph__dt_00(1|2|3|4|5|6|7|8)__.*pt$",
		'colony_5__dt_12345678__t_all': "^colony005_segmentation__assgraph__dt_00(1|2|3|4|5|6|7|8)__.*pt$",
		'colony_1234__dt_1234__t_all': "^colony00(1|2|3|4)_segmentation__assgraph__dt_00(1|2|3|4)__.*pt$",
		'colony_5__dt_1234__t_all': "^colony005_segmentation__assgraph__dt_00(1|2|3|4)__.*pt$",
		'colony_12345__dt_1234__t_all': "^colony00(1|2|3|4|5)_segmentation__assgraph__dt_00(1|2|3|4)__.*pt$",
		'colony_012345__dt_1234__t_all': "^colony00(0|1|2|3|4|5)_segmentation__assgraph__dt_00(1|2|3|4)__.*pt$",
		'colony_01234__dt_1234__t_all': "^colony00(0|1|2|3|4)_segmentation__assgraph__dt_00(1|2|3|4)__.*pt$",
		'colony_0123__dt_1234__t_all': "^colony00(0|1|2|3)_segmentation__assgraph__dt_00(1|2|3|4)__.*pt$",
		'colony_6__dt_1234__t_all': "^colony006_segmentation__assgraph__dt_00(1|2|3|4)__.*åpt$",
		'colony_6__dt_1__t_all': "^colony006_segmentation__assgraph__dt_001__.*pt$",
		'colony_0123__dt_1234__t_all': "^colony00(0|1|2|3)_segmentation__assgraph__dt_00(1|2|3|4)__.*pt$",
		'colony_0__dt_1__t_all': "^colony000_segmentation__assgraph__dt_001__.*pt$",
		'colony_6__dt_1234__t_all': "^colony006_segmentation__assgraph__dt_00(1|2|3|4)__.*pt$",
		'colony_56__dt_1234__t_all': "^colony00(5|6)_segmentation__assgraph__dt_00(1|2|3|4)__.*pt$",
		'colony_5678__dt_1234__t_all': "^colony00(5|6|7|8)_segmentation__assgraph__dt_00(1|2|3|4)__.*pt$",
		'colony_5678_test_set_1234567_dt_1234_t_all': "^(colony00(5|6|7|8)|test_set_(1|2|3|4|5|6|7))_segmentation__assgraph__dt_00(1|2|3|4)__.*pt$",
		'colony_0123478_test_set_1234567_dt_1234_t_all': "^(colony00(0|1|2|3|4|7|8)|test_set_(1|2|3|4|5|6|7))_segmentation__assgraph__dt_00(1|2|3|4)__.*pt$",
		'colony_01234678_test_set_1234567_dt_1234_t_all': "^(colony00(0|1|2|3|4|6|7|8)|test_set_(1|2|3|4|5|6|7))_segmentation__assgraph__dt_00(1|2|3|4)__.*pt$",
		'test_set_1234567_dt_1234_t_all': "^test_set_(1|2|3|4|5|6|7)_segmentation__assgraph__dt_00(1|2|3|4)__.*pt$",	
		'colony_012345678_test_set_1234567_dt_1234_t_all': "^(colony00(0|1|2|3|4|5|6|7|8)|test_set_(1|2|3|4|5|6|7))_segmentation__assgraph__dt_00(1|2|3|4)__.*pt$",
	}
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	# device = 'cpu'
	print(f'device : {device}')

	resultdir = Path(f'{model_dir}/results/{algo}_{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
	
	print("-- result directory --")
	print(resultdir)
	os.makedirs(resultdir, exist_ok=True)
	
	with open(resultdir / 'metadata.json', 'w') as file:
		json.dump(config, file)


	# filter out files that are too large
	file_array = []
	import re
	regex = dataset_regex[test_config["dataset"]]
	for filename in os.listdir(test_config["ass_graphs"]):
		if re.search(regex, filename):
			file_array.append(os.path.join(test_config["ass_graphs"], filename))
	file_array = [ filepath for filepath in file_array if os.stat(filepath).st_size/2**20 < test_config["filter_file_mb"] ]
	print(file_array[-1])
	dataset = AssignmentDataset(file_array)
	
	with open(model_dir / 'hyperparams.json') as file:
		hparams = json.load(file)

	new_net = AssignmentClassifier(
		GNNTracker,
		module__num_node_attr=hparams['num_node_attr'],
		module__num_edge_attr=hparams['num_edge_attr'],
		module__dropout_rate=hparams['dropout_rate'],
		module__encoder_hidden_channels=hparams['encoder_hidden_channels'],
		module__encoder_num_layers=hparams['encoder_num_layers'],
		module__conv_hidden_channels=hparams['conv_hidden_channels'],
		module__conv_num_layers=hparams['conv_num_layers'],
		module__num_classes=1,
		iterator_train=GraphLoader,
		iterator_valid=GraphLoader,
		criterion=torch.nn.BCEWithLogitsLoss,
	).initialize()
	new_net.load_params(model_dir / 'params.pt')
	
 	# Put the network in test mode
	new_net.module_.train(False)  # Set the network to evaluation mode
	
	os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'

	torch.cuda.empty_cache()
	
	
	# filter out files that are too large
	test_array = []
	test_config = config.get('test_config')
	regex = dataset_regex[test_config["dataset"]]
	for filename in os.listdir(test_config["ass_graphs"]):
		if re.search(regex, filename):
			test_array.append(os.path.join(test_config["ass_graphs"], filename))
	original_test_array = test_array
	test_array = [ filepath for filepath in test_array if os.stat(filepath).st_size/2**20 < test_config["filter_file_mb"] ]
	
	filtered_result = test_pipeline(new_net, test_array, resultdir, f'{test_config["dataset"]}_filtered', algorithm=algo )
	torch.cuda.empty_cache()
