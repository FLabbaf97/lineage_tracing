import numpy as np
import warnings
from bread.data import Lineage, BreadWarning

__all__ = ['align_lineages', 'accuracy']

# TODO : a function to validate a Lineage
# i.e. does each bud_id appear once, do ids follow etc

def align_lineages(lineage_truth: Lineage, lineage_pred: Lineage):
	parent_ids_pred = np.empty_like(lineage_truth.parent_ids, dtype=int)

	for irow, (parent_id_truth, bud_id) in enumerate(zip(lineage_truth.parent_ids, lineage_truth.bud_ids)):
		i_bud_pred = np.where(lineage_pred.bud_ids == bud_id)[0]

		if len(i_bud_pred) == 0:
			parent_id_pred = Lineage.SpecialParentIDs.NO_GUESS.value  # bud does not exist in prediction (!!)
			warnings.warn(BreadWarning(f'bud {bud_id} does not appear in the guessed lineage'))
		else:
			parent_id_pred = lineage_pred.parent_ids[i_bud_pred[0]]

		parent_ids_pred[irow] = parent_id_pred

	return lineage_truth.parent_ids, parent_ids_pred, lineage_truth.bud_ids, lineage_truth.time_ids


def accuracy(lineage_truth: Lineage, lineage_pred: Lineage, strict: bool = True) -> float:
	"""Compute the accuracy of a guessed lineage

	Parameters
	----------
	lineage_truth : Lineage
		ground truth lineage
	lineage_pred : Lineage
		guessed lineage (see ``bread.algo.lineage.LineageGuesser``)
	strict : bool, optional
		penalize abscence of a guess, by default True

	Returns
	-------
	score : float
	"""
	
	parent_ids_truth, parent_ids_pred, *_ = align_lineages(lineage_truth, lineage_pred)
	mask_exclude = parent_ids_truth == Lineage.SpecialParentIDs.PARENT_OF_ROOT.value
	if not strict:
		mask_exclude |= parent_ids_pred == Lineage.SpecialParentIDs.NO_GUESS.value
	score = np.mean(~(parent_ids_pred[~mask_exclude] - parent_ids_truth[~mask_exclude]).astype(bool))
	return score
