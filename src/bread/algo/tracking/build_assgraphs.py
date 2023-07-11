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
	
	parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument('--cellgraphs', dest='cellgraph_dirs', required=True, type=str, help='filepaths to the cellgraph directories')
	parser.add_argument('--out', dest='out', required=True, type=Path, help='output directory')
	parser.add_argument('--framediff-min', dest='framediff_min', type=int, default=1, help='minimum number of frame difference between two trackings')
	parser.add_argument('--framediff-max', dest='framediff_max', type=int, default=12, help='maximum number of frame difference between two trackings')
	parser.add_argument('--t1-max', dest='t1_max', type=int, default=-1, help='maximum first frame')

	args = parser.parse_args()
	args_dict = vars(args)
	print("build ass graph args: ", args_dict)
	args.cellgraph_dirs = [ Path(fp) for fp in sorted(glob(args.cellgraph_dirs)) ]
	os.makedirs(args.out, exist_ok=True)
	extra = { 'node_attr': None, 'edge_attr': None, 'num_class_positive': 0, 'num_class_negative': 0 }
	node_attr = []
	edge_attr = []

	
	for cellgraph_dir in tqdm(args.cellgraph_dirs, desc='cellgraph'):
		cellgraph_paths = list(sorted(glob(str(cellgraph_dir / 'cellgraph__*.pkl'))))
		cellgraphs = []
		for cellgraph_path in cellgraph_paths:
			with open(cellgraph_path, 'rb') as file:
				graph = pickle.load(file)
			cellgraphs.append(graph)

		if args.t1_max == -1:
			args.t1_max = len(cellgraphs)

		name = cellgraph_dir.stem
		
		for t1 in tqdm(range(min(len(cellgraphs), args.t1_max)), desc='t1', leave=False):
			for t2 in tqdm(range(min(t1+args.framediff_min, len(cellgraphs)), min(t1+args.framediff_max+1, len(cellgraphs))), desc='t2', leave=False):
				nxgraph = tracking.build_assgraph(cellgraphs[t1], cellgraphs[t2], include_target_feature=True)
				graph, node_attr, edge_attr = tracking.to_data(nxgraph, include_target_feature=True)
				torch.save(graph, args.out / f'{name}__assgraph__dt_{t2-t1:03d}__{t1:03d}_to_{t2:03d}.pt')
				extra['num_class_positive'] += (graph.y == 1).sum()
				extra['num_class_negative'] += (graph.y == 0).sum()

	extra['node_attr'] = node_attr
	extra['edge_attr'] = edge_attr
	# convert a 0-dim tensor to a number
	extra['num_class_positive'] = int(extra['num_class_positive'])
	extra['num_class_negative'] = int(extra['num_class_negative'])

	with open(args.out / 'metadata.txt', 'w') as file:
		file.write(f'Generated on {datetime.datetime.now()} with arguments {sys.argv}\n\n{args_dict}')

	with open(args.out / 'extra.json', 'w') as file:
		json.dump(extra, file)