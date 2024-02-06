if __name__ == '__main__':
	from bread.data import Features, Segmentation
	import bread.algo.tracking as tracking
	from pathlib import Path
	from glob import glob
	from tqdm import tqdm
	import pickle, json
	import argparse
	import os, sys, datetime
	import torch
	from importlib import import_module

	
	parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument('--config', dest='config', required=True, type=str, help='config file')

	# parser.add_argument('--cellgraphs', dest='cellgraph_dirs', required=True, type=str, help='filepaths to the cellgraph directories')
	# parser.add_argument('--output_folder', dest='output_folder', required=True, type=Path, help='output directory')
	# parser.add_argument('--framediff-min', dest='framediff_min', type=int, default=1, help='minimum number of frame difference between two trackings')
	# parser.add_argument('--framediff-max', dest='framediff_max', type=int, default=12, help='maximum number of frame difference between two trackings')
	# parser.add_argument('--t1-max', dest='t1_max', type=int, default=-1, help='maximum first frame')

	args = parser.parse_args()
	config = import_module("bread.config." + args.config).configuration.get('ass_graph_config')
	print("build assignment graph with args: ", config)

	# config["cellgraph_dirs"] = [ Path(fp) for fp in sorted(glob(config["cellgraph_dirs"])) ]
	os.makedirs(Path(config["output_folder"]), exist_ok=True)
	extra = { 'node_attr': None, 'edge_attr': None, 'num_class_positive': 0, 'num_class_negative': 0 }
	node_attr = []
	edge_attr = []

	
	for cellgraph_dir in tqdm(config["cellgraph_dirs"], desc='cellgraph'):
		cellgraph_paths = list(sorted(glob(str(Path(cellgraph_dir) / 'cellgraph__*.pkl'))))
		cellgraphs = []
		for cellgraph_path in cellgraph_paths:
			with open(cellgraph_path, 'rb') as file:
				graph = pickle.load(file)
			cellgraphs.append(graph)

		if config["t1_max"] == -1:
			config["t1_max"] = len(cellgraphs)
		# check if "t1_min" is in the config dictionary
		if ("t1_min" in config):
			t1_min = config["t1_min"]
		else:
			t1_min = 0
			
		name = Path(cellgraph_dir).stem
		big_file_count = 0
		for t1 in tqdm(range(max(0,t1_min),min(len(cellgraphs), config["t1_max"])), desc='t1', leave=False):
			for t2 in tqdm(range(min(t1+config["framediff_min"], len(cellgraphs)), min(t1+config["framediff_max"]+1, len(cellgraphs))), desc='t2', leave=False):
				nxgraph = tracking.build_assgraph(cellgraphs[t1], cellgraphs[t2], include_target_feature=True)
				graph, node_attr, edge_attr = tracking.to_data(nxgraph, include_target_feature=True)
				save_path = Path(config["output_folder"]) / f'{name}__assgraph__dt_{t2-t1:03d}__{t1:03d}_to_{t2:03d}.pt'
				torch.save(graph, save_path)
				
				# # Check the size of the saved file
				# file_size = os.path.getsize(save_path)
				# # If the file size is greater than 20 Mega Bytes (20 * 1024 * 1024 bytes), increase big_file_count
				# if file_size > 20 * 1024 * 1024:
				# 	print('hit a big file, ',file_size )
				# 	big_file_count += 1
				# 	if big_file_count >= 5:
				# 		break
				# else:
				# 	big_file_count = 0

				extra['num_class_positive'] += (graph.y == 1).sum()
				extra['num_class_negative'] += (graph.y == 0).sum()
				

	extra['node_attr'] = node_attr
	extra['edge_attr'] = edge_attr
	# convert a 0-dim tensor to a number
	extra['num_class_positive'] = int(extra['num_class_positive'])
	extra['num_class_negative'] = int(extra['num_class_negative'])

	with open(Path(config["output_folder"]) / 'metadata.txt', 'w') as file:
		file.write(f'Generated on {datetime.datetime.now()} with arguments {sys.argv}\n\n{config}')

	with open(Path(config["output_folder"]) / 'extra.json', 'w') as file:
		json.dump(extra, file)