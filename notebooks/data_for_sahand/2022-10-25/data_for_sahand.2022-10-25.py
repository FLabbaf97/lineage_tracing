from bread.data import Features, Segmentation, Lineage
import pandas as pd
import numpy as np

if __name__ == '__main__':
	import argparse
	from pathlib import Path
	from typing import List
	from glob import glob

	import sys
	sys.path.append('../..')
	from utils import Timer
	import re

	parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument('--segmentations', dest='fp_segmentations', type=str, default='data/colony00[12345]_segmentation.h5', help='filepaths to the segmentations (in order)')
	parser.add_argument('--lineages', dest='fp_lineages', type=str, default='data/colony00[12345]_lineage.csv', help='filepaths to the lineage (in order)')
	parser.add_argument('--out', dest='out', type=Path, default=Path('data.pt'), help='output filepath')

	parser.add_argument('--budding-frames', dest='budding_frames', type=int, default=Features.budding_time, help='how long each budding event of the lineage should be extended to last, in number of frames')
	parser.add_argument('--num-nn', dest='num_nn', type=int, default=4, help='maximum number of nearest neighbours')
	
	parser.add_argument('--scale-length', dest='scale_length', type=float, default=Features.scale_length, help='units of the segmentation pixels, in [length unit]/px, used for feature extraction (passed to ``Features``)')
	parser.add_argument('--scale-time', dest='scale_time', type=float, default=Features.scale_time, help='units of the segmentation frame separation, in [time unit]/frame, used for feature extraction (passed to ``Features``)')
	
	parser.add_argument('--num-processes', dest='num_processes', type=int, default=None, help='number of worker processes to use')

	args = parser.parse_args()
	args_dict = vars(args)
	args.fp_segmentations = [ Path(fp) for fp in sorted(glob(args.fp_segmentations)) ]
	args.fp_lineages = [ Path(fp) for fp in sorted(glob(args.fp_lineages)) ]

	print(vars(args))

	def process(fps):
		fp_seg, fp_lin = fps

		with Timer(f'loading data `{fp_seg}` and `{fp_lin}`'):
			seg = Segmentation.from_h5(fp_seg)
			lin, dts = Lineage.from_csv(fp_lin).only_budding_events().extended_budding_events(args.budding_frames, len(seg)-1, return_dts=True)
			feat = Features(
				seg,
				args.scale_length, args.scale_time,
				budding_time=args.budding_frames,
				nn_threshold=np.inf
			)

		# unique identifiers inside the colony
		time_ids = []
		bud_ids = []
		candidate_ids = []
		# groud truth
		is_budding = []
		# metadata
		num_nn = []
		time_since_budding = []
		# features
		dists = []
		expansion_vectors = []
		budcm_to_budpts = []
		budcm_to_candidatecms = []
		budmajs = []
		candidatemajs = []
		budmins = []
		candidatemins = []

		with Timer(f'compute neighbouring cell pairs `{fp_seg}`'):
			for (parent_id, bud_id, time_id), dt in zip(lin, dts):
				if bud_id not in seg.cell_ids(time_id): continue
				nn_ids = feat._nearest_neighbours_of(time_id, bud_id)[:args.num_nn]  # extract the num_nn closest neighbours
				for nn_id in nn_ids:
					time_ids.append(time_id)
					bud_ids.append(bud_id)
					candidate_ids.append(nn_id)
					is_budding.append(parent_id == nn_id)
					num_nn.append(len(nn_ids))
					time_since_budding.append(dt)

		with Timer(f'compute features `{fp_seg}`'):
			for i, (time_id, bud_id, candidate_id) in enumerate(zip(time_ids, bud_ids, candidate_ids)):
				budcm_to_candidatecm = feat.pair_cmtocm(time_id, candidate_id, bud_id)
				candidatecm_to_budpt = feat.pair_budpt(time_id, bud_id, candidate_id)
				bud_el = feat._ellipse(bud_id, time_id)
				candidate_el = feat._ellipse(candidate_id, time_id)

				dists.append(feat.pair_dist(time_id, bud_id, candidate_id))
				expansion_vectors.append(feat._expansion_vector(bud_id, candidate_id, time_id))
				budcm_to_budpts.append(budcm_to_candidatecm + candidatecm_to_budpt)
				budcm_to_candidatecms.append(budcm_to_candidatecm)
				budmajs.append(np.array([ np.cos(bud_el.angle), np.sin(bud_el.angle) ]) * bud_el.r_maj)
				candidatemajs.append(np.array([ np.cos(candidate_el.angle), np.sin(candidate_el.angle) ]) * candidate_el.r_maj)
				budmins.append(np.array([ np.cos(bud_el.angle+np.pi/2), np.sin(bud_el.angle+np.pi/2) ]) * bud_el.r_min)
				candidatemins.append(np.array([ np.cos(candidate_el.angle+np.pi/2), np.sin(candidate_el.angle+np.pi/2) ]) * candidate_el.r_min)

		return fps, pd.DataFrame(dict(
			# unique identifiers inside the colony
			time_id=time_ids,
			bud_id=bud_ids,
			candidate_id=candidate_ids,
			# colony_id=[int(re.findall('(?:.*colony)(\d\d\d)(?:_.*)', str(fp_seg))[0])]*len(time_ids),
			colony_id=[str(fp_seg)]*len(time_ids),
			# metadata
			num_nn=num_nn,
			time_since_budding=time_since_budding,
			# features
			dist=dists,
			expansion_vector=expansion_vectors,
			budcm_to_budpt=budcm_to_budpts,
			budcm_to_candidatecm=budcm_to_candidatecms,
			budmaj=budmajs,
			candidatemaj=candidatemajs,
			budmin=budmins,
			candidatemin=candidatemins,
			# ground truth
			is_budding=is_budding,
		))

	
	from multiprocessing import Pool
	dats = []
	with Pool(args.num_processes) as pool:
		for dat in pool.imap_unordered(process, zip(args.fp_segmentations, args.fp_lineages)):
			dats.append(dat)

	df = pd.concat([ dat[1] for dat in dats ])
	df.to_pickle(args.out)