if __name__ == '__main__':
	from bread.data import Features, Segmentation, SegmentationFile, Microscopy
	import bread.algo.tracking as tracking
	import argparse
	from pathlib import Path
	from glob import glob
	from tqdm import tqdm
	import os, sys
	import pickle
	import datetime
	from importlib import import_module

	
	parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument('--config', dest='config', required=True, type=str, help='config file')
	parser.add_argument('--fov', dest = 'fov', default = 'FOV0', type=str, help='if your segmentation file has multiple FOV, detrmine which one correlated with the microscopy files of the same name.')
	# parser.add_argument('--segmentations', dest='fp_segmentations', required=True, type=str, help='filepaths to the segmentations')
	# parser.add_argument('--out', dest='out', required=True, type=Path, help='output directory')

	# parser.add_argument('--nn-threshold', dest='nn_threshold', type=float, default=Features.nn_threshold, help='nearest neighbour threshold to assign cell neighbours, in px')
	# parser.add_argument('--scale-length', dest='scale_length', type=float, default=Features.scale_length, help='units of the segmentation pixels, in [length unit]/px, used for feature extraction (passed to ``Features``)')
	# parser.add_argument('--scale-time', dest='scale_time', type=float, default=Features.scale_time, help='units of the segmentation frame separation, in [time unit]/frame, used for feature extraction (passed to ``Features``)')
	# parser.add_argument('--cell-fid', dest='cell_features_id', type=int, default=0, help='determine which features to use for cells in cell graph')
	# parser.add_argument('--edge-fid', dest='edge_features_id', type=int, default=0, help='determine which features to use for edges in cell graph')

	args = parser.parse_args()
	config = import_module("bread.config." + args.config).configuration.get('cell_graph_config')
	print("build cell graph args: ", config)

	edge_features = config['edge_f_list']
	cell_features = config['cell_f_list']
	file_array = []
	for filename in config['input_segmentations']:
		if filename.endswith(".h5"):
			file_array.append(filename)
	config['input_segmentations'] = file_array

	for fp_segmentation in tqdm(config['input_segmentations'], desc='segmentation'):
		seg = SegmentationFile.from_h5(fp_segmentation).get_segmentation(args.fov)
		try:
			microscopy_file = Path(fp_segmentation).parent.parent /'microscopies'/ Path(fp_segmentation).stem
			# replace the name segmentation with microscopy in microscopy_file
			microscopy_file = str(microscopy_file).replace('segmentation', 'microscopy') + '.tif'
			microscopy = Microscopy.from_tiff(microscopy_file)
		except:
			microscopy = None
		feat = Features(seg, nn_threshold=config['nn_threshold'], scale_length=config['scale_length'], scale_time=config['scale_time'], microscopy=microscopy)
		name = Path(fp_segmentation).stem
		os.makedirs(Path(config["output_folder"]) / name, exist_ok=True)

		for time_id in tqdm(range(len(seg)), desc='frame', leave=False):
			graph = tracking.build_cellgraph(feat, time_id=time_id, 
			cell_features=cell_features,
			edge_features=edge_features
			)
			with open(Path(config["output_folder"]) / name / f'cellgraph__{time_id:03d}.pkl', 'wb') as file:
				pickle.dump(graph, file)

	with open(Path(config["output_folder"]) / 'metadata.txt', 'w') as file:
		file.write(f'Generated on {datetime.datetime.now()} with arguments {args.config}\n\n{config}')