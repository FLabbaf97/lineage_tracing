from bread.data import Segmentation
import numpy as np
import itertools

__all__ = ['assignment_from_segmentation']

def assignment_from_segmentation(seg: Segmentation, idt1: int, idt2: int):
	cellids1 = seg.cell_ids(idt1)
	cellids2 = seg.cell_ids(idt2)
	
	ass = np.empty((len(cellids1), len(cellids2)), dtype=int)
	
	for idx1, idx2 in itertools.product(range(len(cellids1)), range(len(cellids2))):
		ass[idx1, idx2] = cellids1[idx1] == cellids2[idx2]
		
	return ass