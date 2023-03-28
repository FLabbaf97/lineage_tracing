from ._extraction import *
from bread.data import Features, Segmentation, Lineage
import torch

if __name__ == '__main__':
	import argparse
	from pathlib import Path
	from typing import List
	from glob import glob

	parser = argparse.ArgumentParser('bread.algo.lineage.nn.preprocess', description='extract training data from segmentations and lineage', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument('--segmentations', dest='fp_segmentations', type=str, default='data/colony00[12345]_segmentation.h5', help='filepaths to the segmentations (in order)')
	parser.add_argument('--lineages', dest='fp_lineages', type=str, default='data/colony00[12345]_lineage.csv', help='filepaths to the lineage (in order)')
	parser.add_argument('--out', dest='out', type=Path, default=Path('data.pt'), help='output filepath')

	parser.add_argument('--filter-nan', dest='filter_nan', action='store_true', help='filter out data samples containing nan')
	parser.add_argument('--filter-incomplete-expspeed', dest='filter_incomplete_expspeed', action='store_true', help='filter out data samples where the full number of frames required for expspeed was not available, due to being at the end of the movie')
	
	parser.add_argument('--budding-frames', dest='budding_frames', type=int, default=Features.budding_time, help='how long each budding event of the lineage should be extended to last, in number of frames')
	parser.add_argument('--nn-threshold-px', dest='nn_threshold', type=int, default=Features.nn_threshold, help='threshold to consider two cells as neighbors, in pixels')
	
	parser.add_argument('--scale-length', dest='scale_length', type=float, default=Features.scale_length, help='units of the segmentation pixels, in [length unit]/px, used for feature extraction (passed to ``Features``)')
	parser.add_argument('--scale-time', dest='scale_time', type=float, default=Features.scale_time, help='units of the segmentation frame separation, in [time unit]/frame, used for feature extraction (passed to ``Features``)')
	
	parser.add_argument('--num-processes', dest='num_processes', type=int, default=None, help='number of worker processes to use')

	args = parser.parse_args()
	args_dict = vars(args)
	args.fp_segmentations = [ Path(fp) for fp in sorted(glob(args.fp_segmentations)) ]
	args.fp_lineages = [ Path(fp) for fp in sorted(glob(args.fp_lineages)) ]

	import logging
	logging.basicConfig()
	logger = logging.getLogger()
	logger.setLevel(logging.INFO)

	import time
	class Timer:
		def __init__(self, name = ''):
			self.name = name

		def __enter__(self):
			self.time_start = time.perf_counter()
			return self

		def __exit__(self, *exc_info):
			print(f'"{self.name}" finished in {time.perf_counter()-self.time_start:.2f}s')
	
	def process(fps):
		fp_seg, fp_lin = fps

		with Timer(f'loading data `{fp_seg}` and `{fp_lin}`'):
			seg = Segmentation.from_h5(fp_seg)
			lin, dts = Lineage.from_csv(fp_lin).only_budding_events().extended_budding_events(args.budding_frames, len(seg)-1, return_dts=True)
			feat = Features(
				seg,
				args.scale_length, args.scale_time,
				budding_time=args.budding_frames,
				nn_threshold=args.nn_threshold
			)

		time_ids = []
		bud_ids = []
		candidate_ids = []
		is_budding = []
		num_nn = []
		time_since_budding = []

		with Timer(f'compute neighbouring cell pairs `{fp_seg}`'):
			for (parent_id, bud_id, time_id), dt in zip(lin, dts):
				if bud_id not in seg.cell_ids(time_id): continue
				nn_ids = feat._nearest_neighbours_of(time_id, bud_id)
				for nn_id in nn_ids:
					time_ids.append(time_id)
					bud_ids.append(bud_id)
					candidate_ids.append(nn_id)
					is_budding.append(parent_id == nn_id)
					num_nn.append(len(nn_ids))
					time_since_budding.append(dt)
			# if not args.all_pairs:
			# 	# include only pairs containing buds
			# else:
			# 	# include all pairs
			# 	for time_id in range(len(seg)):
			# 		for cell_id in seg.cell_ids(time_id):
			# 			nn_ids = feat._nearest_neighbours_of(time_id, cell_id)
			# 			for nn_id in feat._nearest_neighbours_of(time_id, cell_id):
			# 				time_ids.append(time_id)
			# 				bud_ids.append(cell_id)
			# 				candidate_ids.append(nn_id)
			# 				is_budding.append((nn_id, cell_id, time_id) in lin)
			# 				num_nn.append(len(nn_ids))

		N_pairs = len(is_budding)
		x = torch.zeros((N_pairs, NUM_FEATURES), dtype=torch.float)
		y = torch.tensor(is_budding, dtype=torch.long)

		with Timer(f'compute cell and pair features (num datapoints : {N_pairs}) (`{fp_seg}`)'):
			for i, (time_id, bud_id, candidate_id) in enumerate(zip(time_ids, bud_ids, candidate_ids)):
				x[i, :] = extract_features(feat, time_id, bud_id, candidate_id)

		time_ids = torch.tensor(time_ids, dtype=torch.long)
		bud_ids = torch.tensor(bud_ids, dtype=torch.long)
		candidate_ids = torch.tensor(candidate_ids, dtype=torch.long)
		num_nn = torch.tensor(num_nn, dtype=torch.long)
		time_since_budding = torch.tensor(time_since_budding, dtype=torch.long)

		with Timer(f'filter data (`{fp_seg}`)'):
			mask_discard = torch.zeros(N_pairs, dtype=bool)
			if args.filter_nan:
				mask_discard |= torch.any(x.isnan(), dim=1)
			if args.filter_incomplete_expspeed:
				mask_discard |= time_ids > (len(seg)-args.budding_frames)
			x = x[~mask_discard]
			y = y[~mask_discard]
			time_ids = time_ids[~mask_discard]
			bud_ids = bud_ids[~mask_discard]
			candidate_ids = candidate_ids[~mask_discard]
			num_nn = num_nn[~mask_discard]
			time_since_budding = time_since_budding[~mask_discard]

		return dict(
			fp_segmentation=fp_seg, fp_lineage=fp_lin,
			time_id=time_ids, bud_id=bud_ids, candidate_id=candidate_ids,
			num_nn=num_nn, time_since_budding=time_since_budding,
			x=x, y=y,
		)

	# for a in zip(args.fp_segmentations, args.fp_lineages):
	# 	process(a)

	from multiprocessing import Pool
	dats = []
	with Pool(args.num_processes) as pool:
		for dat in pool.imap_unordered(process, zip(args.fp_segmentations, args.fp_lineages)):
			dats.append(dat)
	
	torch.save(dict(data=dats, meta=args_dict, feat_names=FEATURE_NAMES), args.out)
	

# if __name__ == '__main__':
# 	# $ python -m bread.algo.lineage.nn.preprocess --filter-nan True --filter-after-time 176
# 	# TODO : unify this with nn._lineage feat extraction ? 

# 	from pathlib import Path
# 	import argparse

# 	parser = argparse.ArgumentParser('bread.algo.lineage.nn.preprocess', description='convert cell statistics (feat_cells.npz, feat_pairs.npz) into pytorch training data', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# 	parser.add_argument('--fp-feat-cells', dest='fp_feat_cells', type=Path, default=Path('docs/source/examples/outputs/feat_cells.npz'), help='path to feat_cells.npz')
# 	parser.add_argument('--fp-feat-pairs', dest='fp_feat_pairs', type=Path, default=Path('docs/source/examples/outputs/feat_pairs.npz'), help='path to feat_pairs.npz')
# 	parser.add_argument('--filter-nan', dest='filter_nan', type=bool, default=True, help='filter out data samples containing nan')
# 	parser.add_argument('--filter-after-time', dest='filter_after_time', type=int, default=-1, help='filter out data samples where `time_id` is greater than specified (useful when we want to exclude expspeed features where a small number of frames was used for estimation). set to `-1` to disable')
# 	parser.add_argument('--fp-out', dest='fp_out', type=Path, default=Path('data.pt'), help='output filepath')
# 	args = parser.parse_args()
	
# 	import torch
# 	import numpy as np
# 	import logging
# 	logging.basicConfig()
# 	logger = logging.getLogger()
# 	logger.setLevel(logging.INFO)

# 	feat_cells_names = [
# 		'colony_id', 'time_id', 'cell_id',  # metadata
# 		'area', 'r_equiv', 'ecc', 'maj_x', 'maj_y', 'maj_arg'  # features
# 	]
# 	feat_pairs_names = [
# 		'colony_id', 'time_id', 'bud_id', 'candidate_id',  # metadata
# 		'is_budding',  # train target (y)
# 		'majmaj_angle', 'majbudpt_angle', 'dist', 'expspeed',  # features
# 		'cmtocm_x', 'cmtocm_y', 'cmtocm_arg', 'cmtocm_len',
# 		'budpt_x', 'budpt_y', 'budpt_arg', 'budpt_len'
# 	]

# 	logger.info('loading npz files...')

# 	with np.load(args.fp_feat_cells) as file_feat_cells:
# 		N_cells = len(file_feat_cells['time_id'])
# 		feat_cells = torch.zeros((N_cells, len(feat_cells_names)))
# 		for d, name in enumerate(feat_cells_names):
# 			feat_cells[:, d] = torch.from_numpy(file_feat_cells[name])

# 	with np.load(args.fp_feat_pairs) as file_feat_pairs:
# 		N_pairs = len(file_feat_pairs['is_budding'])  # number of pairs
# 		feat_pairs = torch.zeros((N_pairs, len(feat_pairs_names)))
# 		for d, name in enumerate(feat_pairs_names):
# 			feat_pairs[:, d] = torch.from_numpy(file_feat_pairs[name])

# 	logger.info('building training targets (y)...')

# 	# 1 if is budding, 0 else
# 	# we convert to torch.long dtype, so that it is understood as a class index
# 	# see https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
# 	y = feat_pairs[:, feat_pairs_names.index('is_budding')].type(torch.long)

# 	def get_cell_mask(colony_id, time_id, cell_id):
# 		mask = \
# 			(feat_cells[:, feat_cells_names.index('colony_id')] == colony_id) &\
# 			(feat_cells[:, feat_cells_names.index('time_id')] == time_id) &\
# 			(feat_cells[:, feat_cells_names.index('cell_id')] == cell_id)
# 		num_indices = mask.sum()
# 		assert num_indices > 0, f'no such cell exists colony_id={colony_id}, time_id={time_id}, cell_id={cell_id}'
# 		assert num_indices == 1, f'more than one cell matches colony_id={colony_id}, time_id={time_id}, cell_id={cell_id}'
# 		return mask

# 	# 24 features
# 	x = torch.zeros((N_pairs, 24), dtype=torch.float)

# 	logger.info('building bud feature data (x[0:6])...')

# 	# complete the bud features d=0..6 (6 total)
# 	for i, (colony_id, time_id, bud_id) in enumerate(feat_pairs[:, [feat_pairs_names.index('colony_id'), feat_pairs_names.index('time_id'), feat_pairs_names.index('bud_id')]]):
# 		mask = get_cell_mask(colony_id, time_id, bud_id)
# 		x[i, 0:6] = feat_cells[mask, 3:9]

# 	logger.info('building candidate feature data (x[6:12])...')

# 	# complete the candidate features d=6..12 (6 total)
# 	for i, (colony_id, time_id, candidate_id) in enumerate(feat_pairs[:, [feat_pairs_names.index('colony_id'), feat_pairs_names.index('time_id'), feat_pairs_names.index('candidate_id')]]):
# 		mask = get_cell_mask(colony_id, time_id, candidate_id)
# 		x[i, 6:12] = feat_cells[mask, 3:9]

# 	logger.info('building pair feature data (x[12:24])...')

# 	# complete the pair features d=12..24 (12 total)
# 	x[:, 12] = feat_pairs[:, feat_pairs_names.index('majmaj_angle')]
# 	x[:, 13] = feat_pairs[:, feat_pairs_names.index('majbudpt_angle')]
# 	x[:, 14] = feat_pairs[:, feat_pairs_names.index('dist')]
# 	x[:, 15] = feat_pairs[:, feat_pairs_names.index('expspeed')]
# 	x[:, 16] = feat_pairs[:, feat_pairs_names.index('cmtocm_x')]
# 	x[:, 17] = feat_pairs[:, feat_pairs_names.index('cmtocm_y')]
# 	x[:, 18] = feat_pairs[:, feat_pairs_names.index('cmtocm_len')]
# 	x[:, 19] = feat_pairs[:, feat_pairs_names.index('cmtocm_arg')]
# 	x[:, 20] = feat_pairs[:, feat_pairs_names.index('budpt_x')]
# 	x[:, 21] = feat_pairs[:, feat_pairs_names.index('budpt_y')]
# 	x[:, 22] = feat_pairs[:, feat_pairs_names.index('budpt_len')]
# 	x[:, 23] = feat_pairs[:, feat_pairs_names.index('budpt_arg')]

# 	mask_discard = torch.zeros(N_pairs, dtype=bool)
# 	if args.filter_nan:
# 		logger.info('filtering out NaN values')
# 		mask_discard |= torch.any(x.isnan(), dim=1)
# 	if args.filter_after_time != -1:
# 		logger.info('filtering out late frames')
# 		mask_discard |= feat_pairs[:, feat_pairs_names.index('time_id')] > args.filter_after_time
# 	x = x[~mask_discard]
# 	y = y[~mask_discard]

# 	logger.info('saving...')

# 	torch.save(dict(x=x, y=y), args.fp_out)

# 	logger.info('done !')