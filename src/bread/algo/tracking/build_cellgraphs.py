if __name__ == '__main__':
	from bread.data import Features, Segmentation
	import bread.algo.tracking as tracking
	import argparse
	from pathlib import Path
	from glob import glob
	from tqdm import tqdm
	import os, sys
	import pickle
	import datetime
	
	parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument('--segmentations', dest='fp_segmentations', required=True, type=str, help='filepaths to the segmentations')
	parser.add_argument('--out', dest='out', required=True, type=Path, help='output directory')

	parser.add_argument('--nn-threshold', dest='nn_threshold', type=float, default=Features.nn_threshold, help='nearest neighbour threshold to assign cell neighbours, in px')
	parser.add_argument('--scale-length', dest='scale_length', type=float, default=Features.scale_length, help='units of the segmentation pixels, in [length unit]/px, used for feature extraction (passed to ``Features``)')
	parser.add_argument('--scale-time', dest='scale_time', type=float, default=Features.scale_time, help='units of the segmentation frame separation, in [time unit]/frame, used for feature extraction (passed to ``Features``)')
	
	args = parser.parse_args()
	args_dict = vars(args)
	args.fp_segmentations = [ Path(fp) for fp in sorted(glob(args.fp_segmentations)) ]

	for fp_segmentation in tqdm(args.fp_segmentations, desc='segmentation'):
		seg = Segmentation.from_h5(fp_segmentation)
		feat = Features(seg, nn_threshold=args.nn_threshold, scale_length=args.scale_length, scale_time=args.scale_time)
		name = fp_segmentation.stem
		os.makedirs(args.out / name, exist_ok=True)

		for time_id in tqdm(range(len(seg)), desc='frame', leave=False):
			graph = tracking.build_cellgraph(feat, time_id=time_id)
			with open(args.out / name / f'cellgraph__{time_id:03d}.pkl', 'wb') as file:
				pickle.dump(graph, file)

	with open(args.out / 'metadata.txt', 'w') as file:
		file.write(f'Generated on {datetime.datetime.now()} with arguments {sys.argv}\n\n{args_dict}')