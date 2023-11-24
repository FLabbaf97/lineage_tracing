from bread.data import Features, Segmentation, Lineage
from math import sin, cos, sqrt, atan2
from typing import List
import torch

__all__ = ['extract_features', 'NUM_FEATURES', 'FEATURE_NAMES']

FEATURE_NAMES: List[str] = [
	# bud features, idx 0..6
	'bud_area', 'bud_r_equiv', 'bud_ecc', 'bud_maj_x', 'bud_maj_y', 'bud_maj_arg',
	# candidate features, idx 6..12
	'candidate_area', 'candidate_r_equiv', 'candidate_ecc', 'candidate_maj_x', 'candidate_maj_y', 'candidate_maj_arg',
	# pair features, idx 12..25
	'majmaj_angle', 'majbudpt_angle', 'cmtocm_budmaj_angle',
	'dist', 'expspeed',
	'cmtocm_x', 'cmtocm_y', 'cmtocm_arg', 'cmtocm_len',
	'budpt_x', 'budpt_y', 'budpt_arg', 'budpt_len'
]
NUM_FEATURES: int = len(FEATURE_NAMES)

def extract_features(feat: Features, time_id: int, bud_id: int, candidate_id: int, feat_options: dict = {}) -> torch.Tensor:
	"""Extract features for a given pair of (bud, candidate) at a given time

	Parameters
	----------
	feat : Features
	time_id : int
	bud_id : int
	candidate_id : int

	Returns
	-------
	torch.Tensor : shape=25, dtype=torch.float
		0..6 bud features : area, r_equiv, ecc, maj_x, maj_y, maj_arg
		6..12 candidate features : area, r_equiv, ecc, maj_x, maj_y, maj_arg
		12..25 pair features : 'majmaj_angle', 'majbudpt_angle', 'cmtocm_budmaj_angle', 'dist', 'expspeed',
		'cmtocm_x', 'cmtocm_y', 'cmtocm_arg', 'cmtocm_len',
		'budpt_x', 'budpt_y', 'budpt_arg', 'budpt_len'
	"""

	x = torch.zeros(NUM_FEATURES, dtype=torch.float)

	# bud features
	x[0] = feat.cell_area(time_id, bud_id)
	x[1] = feat.cell_r_equiv(time_id, bud_id)
	x[2] = feat.cell_ecc(time_id, bud_id)
	x[3] = cos(feat.cell_alpha(time_id, bud_id))
	x[4] = sin(feat.cell_alpha(time_id, bud_id))
	x[5] = feat.cell_alpha(time_id, bud_id)

	# candidate features
	x[6] = feat.cell_area(time_id, candidate_id)
	x[7] = feat.cell_r_equiv(time_id, candidate_id)
	x[8] = feat.cell_ecc(time_id, candidate_id)
	x[9] = cos(feat.cell_alpha(time_id, candidate_id))
	x[10] = sin(feat.cell_alpha(time_id, candidate_id))
	x[11] = feat.cell_alpha(time_id, candidate_id)

	# pair features
	x[12] = feat.pair_majmaj_angle(time_id, bud_id, candidate_id)
	x[13] = feat.pair_majbudpt_angle(time_id, bud_id, candidate_id)
	x[14] = feat.pair_cmtocm_budmaj_angle(time_id, bud_id, candidate_id)
	x[15] = feat.pair_dist(time_id, bud_id, candidate_id)
	x[16] = feat.pair_expspeed(time_id, bud_id, candidate_id)
	cmtocm = feat.pair_cmtocm(time_id, bud_id, candidate_id)
	x[17] = cmtocm[0]
	x[18] = cmtocm[1]
	x[19] = sqrt(cmtocm[0]**2 + cmtocm[1]**2)
	x[20] = atan2(cmtocm[1], cmtocm[0])
	budpt = feat.pair_budpt(time_id, bud_id, candidate_id)
	x[21] = budpt[0]
	x[22] = budpt[1]
	x[23] = sqrt(budpt[0]**2 + budpt[1]**2)
	x[24] = atan2(budpt[1], budpt[0])

	return x