
from pathlib import Path
from typing import List
import torch
from torch_geometric.data import Dataset, Data
import os, json

__all__ = ['AssignmentDataset', 'InMemoryAssignmentDataset']

def normalize(x: torch.Tensor, eps=1e-6) -> torch.Tensor:
	"""Standardize data such that it has zero mean and unit variance"""
	return (x - x.mean(axis=0)) / (x.std(axis=0, unbiased=True) + eps)


class AssignmentDataset(Dataset):
	def __init__(self, filepaths: List[Path]):
		"""Assignment graph dataset, in memory. Data is loaded, features normalized, and stored into a temporary file.

		Parameters
		----------
		filepaths : List[Path]
			List of paths to the assignment graphs in .pt format. See ``build_assgraphs.py``
		"""

		self.filepaths = filepaths
		self.filenames = [ Path(filepath).name for filepath in self.filepaths ]
		root = os.path.dirname(self.filepaths[0])
		with open(Path(root) / 'extra.json') as file:
			extra = json.load(file)
		self.num_class_positive: int = extra['num_class_positive']  # class balance
		self.num_class_negative: int = extra['num_class_negative']
		self.node_attr = extra['node_attr']
		self.edge_attr = extra['edge_attr']
		super().__init__(root, transform=None, pre_transform=None, pre_filter=None)

	@property
	def raw_file_names(self):
		return self.filenames

	@property
	def processed_file_names(self):
		return self.filenames

	def len(self) -> int:
		return len(self.processed_file_names)

	def get(self, idx: int) -> Data:
		return torch.load(self.filepaths[idx])

class InMemoryAssignmentDataset(Dataset):
	def __init__(self, datalist: List[Data]):
		self.datalist = datalist
		super().__init__(None, transform=None, pre_transform=None, pre_filter=None)

	def len(self) -> int:
		return len(self.datalist)

	def get(self, idx: int) -> Data:
		return self.datalist[idx]