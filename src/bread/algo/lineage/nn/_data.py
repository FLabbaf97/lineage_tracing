from ._kfold import BaseKFoldDataModule, kfold_masks
from pathlib import Path
from typing import Callable, List, Tuple
import torch
from torch.utils.data import Dataset, DataLoader, Subset


class LineageData(Dataset):
	def __init__(self, dict_data: dict):
		self.dict_data = dict_data
		# self.transforms: List[Callable[['dict[str, torch.Tensor]'], 'dict[str, torch.Tensor]']] = []
		# if equalize:
		# 	self.transforms.append(EqualizeTransform())
		self.x = torch.cat([ data['x'] for data in self.dict_data['data'] ])
		self.y = torch.cat([ data['y'] for data in self.dict_data['data'] ])
		assert len(self.x) == len(self.y), 'data and labels must have the same length'

	def __len__(self):
		return len(self.x)

	def __getitem__(self, index) -> 'dict[str, torch.Tensor]':
		ret = dict(x=self.x[index], y=self.y[index])
		# for fn in self.transforms:
		# 	ret = fn(ret)
		return ret

	def class_weights(self) -> 'torch.Tensor[torch.float]':
		"""Compute the normalize class weights for ``torch.CrossEntropyLoss``,
		as the inverse of the frequency of each class in the dataset
		
		Since :math:`w_0 L_0 + w_1 L_1 = l_0 + l_1`, where :math:`L_0 = N_0 l_0`, we impose :math:`w_0 \\propto 1/N_0`
		"""
		n = self.y.size()[0]
		_, counts = torch.unique(self.y, sorted=True, return_counts=True)
		freqs = counts/n
		weights = 1.0/freqs
		weights /= weights.sum()
		return weights


class LineageDataModule(BaseKFoldDataModule):
	def __init__(self, fp_data_train: Path, fp_data_test: Path, frac_train: float=0.9, batch_size: int=32, num_workers=12):
		super().__init__()

		self.frac_train = frac_train
		self.fp_data_train = fp_data_train
		self.fp_data_test = fp_data_test
		self.batch_size = batch_size
		self.num_workers = num_workers

	def setup(self, stage: str):
		self.train_dataset = LineageData(torch.load(self.fp_data_train))
		self.test_dataset = LineageData(torch.load(self.fp_data_test))

	def setup_folds(self, num_folds: int) -> None:
		if num_folds == 0:
			# no kfold validation
			self.num_folds = 0
			return

		self.num_folds = num_folds
		self.kfold_masks = kfold_masks(self.num_folds, len(self.train_dataset))

	def setup_fold_index(self, fold_index: int) -> None:
		if self.num_folds == 0:
			# no kfold validation
			self.train_fold = self.train_dataset
			self.val_fold = Subset(self.train_dataset, [])
			return

		train_mask = self.kfold_masks[fold_index]
		train_indices = train_mask.nonzero(as_tuple=True)[0]
		val_indices = (~train_mask).nonzero(as_tuple=True)[0]
		self.train_fold = Subset(self.train_dataset, train_indices)
		self.val_fold = Subset(self.train_dataset, val_indices)

	def train_dataloader(self) -> DataLoader:
		return DataLoader(self.train_fold,
			batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)

	def val_dataloader(self) -> DataLoader:
		return DataLoader(self.val_fold,
			batch_size=self.batch_size, num_workers=self.num_workers)

	def test_dataloader(self) -> DataLoader:
		return DataLoader(self.test_dataset,
			batch_size=self.batch_size, num_workers=self.num_workers)


# class EqualizeTransform:
# 	"""Implements equalization of class labels in a batch"""

# 	def __call__(self, batch: 'dict[str, torch.Tensor]') -> 'dict[str, torch.Tensor]':
# 		x, y = batch['x'], batch['y']

# 		if x.ndim == 1:
# 			return dict(x=x, y=y)

# 		y_uniques, y_counts = torch.unique(y, return_counts=True)
# 		y_count_thresholder = torch.min(y_counts)

# 		mask_points = torch.ones(x.size(), dtype=bool)
# 		for y_unique, y_count in zip(y_uniques, y_counts):
# 			nb_rm = abs(y_count - y_count_thresholder)  # number of points to remove
# 			mmask = torch.ones(y_count, dtype=bool)
# 			mmask[:nb_rm] = False
# 			mask_points[y == y_unique] = mmask[torch.randperm(mmask.size()[0])]

# 		ret = dict(x=x[mask_points, :], y=y[mask_points])
# 		return ret