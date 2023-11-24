from bread.data import Features, Segmentation, Lineage, BreadException, BreadWarning
from .._lineage import LineageGuesser
from ._model import LineageClassifierDeep
from ._extraction import extract_features, NUM_FEATURES
from dataclasses import dataclass, field
from typing import List
from pathlib import Path
import torch
import warnings

__all__ = ['LineageGuesserDeepNN']

@dataclass
class LineageGuesserDeepNN(LineageGuesser):
	"""Guess lineage relations by extraction cell and pair features, then passing them as a classification task to a pretrained neural network

	Note : it is important here to set ``scale_length`` and ``scale_time``, since the neural network has been trained on real units (microns/pixel and minutes/frame respectively).

	Parameters
	----------
	seg : Segmentation
	nn_threshold : float, optional
		Cell masks separated by less than this threshold are considered neighbours. by default 8.0.
	flexible_threshold : bool, optional
		If no nearest neighbours are found within the given threshold, try to find the closest one, by default False.
	num_frames_refractory : int, optional
		After a parent cell has budded, exclude it from the parent pool in the next frames.
		It is recommended to set it to a low estimate, as high values will cause mistakes to propagate in time.
		A value of 0 corresponds to no refractory period.
		by default 0.
	scale_length : float, optional
		Units of the segmentation pixels, in [length unit]/px, used for feature extraction (passed to ``Features``).
		by default 1.
	scale_time : float, optional
		Units of the segmentation frame separation, in [time unit]/frame, used for feature extraction (passed to ``Features``).
		by default 1.
	path_models : Path | List[Path], optional
		Path to the model checkpoint. If multiple checkpoints are given, the models are combined using ensemble voting to minimize bias.
		by default ``[Path('data/generated/models/best/deepnn.ckpt')]``
	"""

	path_models: List[Path] = Path('data/generated/models/best/deepnn.ckpt')

	_models: torch.nn.ModuleList = field(init=False, repr=False)

	def __post_init__(self):
		LineageGuesser.__post_init__(self)
		if not isinstance(self.path_models, list):
			# convert to list
			self.path_models = [self.path_models]
		self._models = torch.nn.ModuleList([LineageClassifierDeep.load_from_checkpoint(p) for p in self.path_models])

	class AllNanFeatures(BreadException):
		def __init__(self, bud_id, candidate_ids, time_id):
			super().__init__(f'Unable to determine parent for bud #{bud_id} at frame #{time_id}. All candidates {candidate_ids} gave (at least one) NaN feature(s)')

	def guess_parent(self, bud_id: int, time_id: int) -> int:
		"""Guess the parent associated to a bud using the classification NN

		Parameters
		----------
		bud_id : int
			id of the bud in the segmentation
		time_id : int
			frame index in the movie

		Returns
		-------
		parent_id : int
			guessed parent id
		"""
		
		candidate_ids = self._candidate_parents(time_id, excluded_ids=[bud_id], nearest_neighbours_of=bud_id)

		# no candidates
		if len(candidate_ids) == 0:
			return Lineage.SpecialParentIDs.PARENT_OF_EXTERNAL.value

		batch = dict(x=torch.zeros((len(candidate_ids), NUM_FEATURES), dtype=torch.float))

		with warnings.catch_warnings():
			warnings.filterwarnings('ignore', category=BreadWarning)
			for i, candidate_id in enumerate(candidate_ids):
				batch['x'][i, :] = extract_features(self._features, time_id, bud_id, candidate_id)

		# remove candidates which have nan features
		mask_discard = torch.any(batch['x'].isnan(), dim=1)
		if mask_discard.all():
			raise LineageGuesserDeepNN.AllNanFeatures(bud_id, candidate_ids, time_id)
		batch['x'] = batch['x'][~mask_discard]
		candidate_ids = candidate_ids[~mask_discard.numpy()]

		# run the models with ensemble averaging
		self._models.eval()
		with torch.no_grad():
			scores = torch.stack([m.scores(batch) for m in self._models]).mean(0)  # shape=(num_candidates)

		# maximize the score
		idx_parent = int(scores.argmax())
		parent_id = candidate_ids[idx_parent]
		
		return parent_id


if __name__ == '__main__':
	from glob import glob
	from bread.algo.lineage import accuracy
	import warnings
	with warnings.catch_warnings():
		warnings.filterwarnings('ignore', category=BreadWarning)
		for fp_seg, fp_lin in zip(sorted(glob('data/raw/colony00[12345]_segmentation.h5')), sorted(glob('data/raw/colony00[12345]_lineage.csv'))):
			seg = Segmentation.from_h5(fp_seg)
			lineage = Lineage.from_csv(fp_lin)
			guesser = LineageGuesserDeepNN(
				segmentation=seg,
				scale_length=1/9.2308,
				scale_time=5,
				# path_models=glob('data/generated/models/best/kfolds/model*.pt')
				path_models=glob('data/generated/models/best/kfolds_min_val_loss/*.ckpt')
				# path_models=Path('data/generated/models/demo/deepnn.2.ckpt')
			)

			lin_pred = guesser.guess_lineage()

			print(f'{fp_seg} | acc={accuracy(lineage, lin_pred, strict=False):.4f}')

	### USING ALL_PAIRS

	# python -m bread.algo.lineage.nn.train --batch-size=2048 --num-epochs=1000 --size-hidden=256 --dropout=0.1 --num-hidden=5
	# data/colony001_segmentation.h5 | acc=0.7500
	# data/colony002_segmentation.h5 | acc=0.7793
	# data/colony003_segmentation.h5 | acc=0.7312
	# data/colony004_segmentation.h5 | acc=0.7167
	# data/colony005_segmentation.h5 | acc=0.8056

	# python -m bread.algo.lineage.nn.train --batch-size=2048 --num-epochs=1000 --size-hidden=512 --dropout=0.1 --num-hidden=4
	# data/colony001_segmentation.h5 | acc=0.7500
	# data/colony002_segmentation.h5 | acc=0.8069
	# data/colony003_segmentation.h5 | acc=0.7527
	# data/colony004_segmentation.h5 | acc=0.7167
	# data/colony005_segmentation.h5 | acc=0.8333

	# python -m bread.algo.lineage.nn.train --batch-size=2048 --num-epochs=1000 --size-hidden=512 --dropout=0.05 --num-hidden=4
	# data/colony001_segmentation.h5 | acc=0.7250
	# data/colony002_segmentation.h5 | acc=0.7862
	# data/colony003_segmentation.h5 | acc=0.7527
	# data/colony004_segmentation.h5 | acc=0.7333
	# data/colony005_segmentation.h5 | acc=0.8056

	# python -m bread.algo.lineage.nn.train --batch-size=2048 --num-epochs=1000 --size-hidden=512 --dropout=0.02 --num-hidden=4         
	# using cmtocm - expvec angle
	# data/colony001_segmentation.h5 | acc=0.7500
	# data/colony002_segmentation.h5 | acc=0.7862
	# data/colony003_segmentation.h5 | acc=0.7527
	# data/colony004_segmentation.h5 | acc=0.7667
	# data/colony005_segmentation.h5 | acc=0.8333

	# python -m bread.algo.lineage.nn.train --batch-size=2048 --num-epochs=1000 --size-hidden=512 --dropout=0.02 --num-hidden=4
	# using cmtocm - bud maj (@ max ecc) angle
	# data/colony001_segmentation.h5 | acc=0.7250
	# data/colony002_segmentation.h5 | acc=0.7517
	# data/colony003_segmentation.h5 | acc=0.7312
	# data/colony004_segmentation.h5 | acc=0.7417
	# data/colony005_segmentation.h5 | acc=0.8611

	# python -m bread.algo.lineage.nn.train --batch-size=2048 --num-epochs=1000 --size-hidden=512 --dropout=0.02 --num-hidden=4
	# using cmtocm - bud maj (@ bud appearance) angle
	# data/colony001_segmentation.h5 | acc=0.7625
	# data/colony002_segmentation.h5 | acc=0.7586
	# data/colony003_segmentation.h5 | acc=0.7419
	# data/colony004_segmentation.h5 | acc=0.7417
	# data/colony005_segmentation.h5 | acc=0.8333

	# python -m bread.algo.lineage.nn.train --model deep --num-epochs=300 --batch-size=1024 --size-hidden=512 --dropout=0.02 --num-hidden=4 --num-folds=5
	# only one model (the last one)
	# data/raw/colony001_segmentation.h5 | acc=0.7750
	# data/raw/colony002_segmentation.h5 | acc=0.7793
	# data/raw/colony003_segmentation.h5 | acc=0.752
	# data/raw/colony004_segmentation.h5 | acc=0.7500
	# data/raw/colony005_segmentation.h5 | acc=0.8889
	# ensemble averaging (very good overall !)
	# data/raw/colony001_segmentation.h5 | acc=0.7625
	# data/raw/colony002_segmentation.h5 | acc=0.7586
	# data/raw/colony003_segmentation.h5 | acc=0.7849
	# data/raw/colony004_segmentation.h5 | acc=0.7417
	# data/raw/colony005_segmentation.h5 | acc=0.8333

	# python -m bread.algo.lineage.nn.train --model deep --num-epochs=1 --batch-size=1024 --size-hidden=512 --dropout=0.02 --num-hidden=4 --num-folds=0
	# (sanity check)
	# data/raw/colony001_segmentation.h5 | acc=0.7000
	# data/raw/colony002_segmentation.h5 | acc=0.7103
	# data/raw/colony003_segmentation.h5 | acc=0.6559
	# data/raw/colony004_segmentation.h5 | acc=0.6917
	# data/raw/colony005_segmentation.h5 | acc=0.7500

	### USING ONLY PAIRS WITH BUDS

	# python -m bread.algo.lineage.nn.train --model deep --num-epochs=150 --batch-size=2048 --size-hidden=512 --dropout=0.02 --num-hidden=4 --num-folds=5
	# no early stopping (we see some overfitting)
	# data/raw/colony001_segmentation.h5 | acc=0.8000
	# data/raw/colony002_segmentation.h5 | acc=0.8483
	# data/raw/colony003_segmentation.h5 | acc=0.8172
	# data/raw/colony004_segmentation.h5 | acc=0.8083
	# data/raw/colony005_segmentation.h5 | acc=0.9167
	# early stopping at minimum of validation loss
	# data/raw/colony001_segmentation.h5 | acc=0.7750
	# data/raw/colony002_segmentation.h5 | acc=0.8621
	# data/raw/colony003_segmentation.h5 | acc=0.8065
	# data/raw/colony004_segmentation.h5 | acc=0.7917
	# data/raw/colony005_segmentation.h5 | acc=0.9444

	# ==> this is great, because we are beating nearest neighbours, using only bud-candidate pairs,
	# i.e. we are missing the information about the context !!