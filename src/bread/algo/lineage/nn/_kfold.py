"""KFold validation, adapted from `https://github.com/Lightning-AI/lightning/blob/master/examples/pl_loops/kfold.py`_"""

from abc import ABC, abstractmethod
from copy import deepcopy
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple, Type
from pytorch_lightning import LightningDataModule, LightningModule
from pytorch_lightning.loops.loop import Loop
from pytorch_lightning.loops.fit_loop import FitLoop
from pytorch_lightning.trainer.states import TrainerFn
import torch
from torchmetrics import Accuracy

__all__ = ['BaseKFoldDataModule', 'EnsembleVotingModel', 'KFoldLoop', 'kfold_masks']


class BaseKFoldDataModule(LightningDataModule, ABC):
	@abstractmethod
	def setup_folds(self, num_folds: int) -> None:
		pass

	@abstractmethod
	def setup_fold_index(self, fold_index: int) -> None:
		pass


class EnsembleVotingModel(LightningModule):
	def __init__(self, model_cls: Type[LightningModule], checkpoint_paths: List[str]) -> None:
		super().__init__()
		self.save_hyperparameters()
		# Create `num_folds` models with their associated fold weights
		self.models = torch.nn.ModuleList([model_cls.load_from_checkpoint(p) for p in checkpoint_paths])
		self.test_acc = Accuracy()

	def test_step(self, batch, batch_idx: int, dataloader_idx: int = 0) -> None:
		# Compute the averaged predictions over the `num_folds` models.
		scores = torch.stack([m.scores(batch) for m in self.models]).mean(0)
		acc = self.test_acc(scores, batch['y'])
		self.log('test_err', 1-acc)
		self.log('hp_metric', 1-acc)


class KFoldLoop(Loop):
	def __init__(self, num_folds: int, export_path: str) -> None:
		super().__init__()
		self.num_folds = num_folds
		self.current_fold: int = 0
		self.export_path = export_path

	@property
	def done(self) -> bool:
		return self.current_fold >= self.num_folds

	def connect(self, fit_loop: FitLoop) -> None:
		self.fit_loop = fit_loop

	def reset(self) -> None:
		"""Nothing to reset in this loop."""

	def on_run_start(self, *args: Any, **kwargs: Any) -> None:
		"""Used to call `setup_folds` from the `BaseKFoldDataModule` instance and store the original weights of the
		model."""
		assert isinstance(self.trainer.datamodule, BaseKFoldDataModule)
		self.trainer.datamodule.setup_folds(self.num_folds)
		self.lightning_module_state_dict = deepcopy(self.trainer.lightning_module.state_dict())
		self.callback_state_dicts = [ deepcopy(callback.state_dict()) for callback in self.trainer.callbacks ]

	def on_advance_start(self, *args: Any, **kwargs: Any) -> None:
		"""Used to call `setup_fold_index` from the `BaseKFoldDataModule` instance."""
		print(f"STARTING FOLD {self.current_fold}")
		assert isinstance(self.trainer.datamodule, BaseKFoldDataModule)
		self.trainer.datamodule.setup_fold_index(self.current_fold)

	def advance(self, *args: Any, **kwargs: Any) -> None:
		"""Used to the run a fitting and testing on the current hold."""

		self._reset_fitting()  # requires to reset the tracking stage.
		self.fit_loop.run()

		self._reset_testing()  # requires to reset the tracking stage.
		self.trainer.test_loop.run()

		self.current_fold += 1  # increment fold tracking number.

	def on_advance_end(self) -> None:
		"""Used to save the weights of the current fold and reset the LightningModule and its optimizers."""
		self.trainer.save_checkpoint(os.path.join(self.export_path, f"model.{self.current_fold}.ckpt"))

		# restore the original weights + optimizers and schedulers.
		self.trainer.lightning_module.load_state_dict(self.lightning_module_state_dict)

		old_state = self.trainer.state.fn  # HACK
		self.trainer.state.fn = TrainerFn.FITTING  
		self.trainer.strategy.setup_optimizers(self.trainer)  # https://github.com/Lightning-AI/lightning/issues/12409
		self.trainer.state.fn = old_state

		# restore the callbacks
		for callback, state_dict in zip(self.trainer.callbacks, self.callback_state_dicts):
			callback.load_state_dict(deepcopy(state_dict))  # deepcopy because ``ModelCheckpoint`` assigns without copying

		self.replace(fit_loop=FitLoop)

	def on_run_end(self) -> None:
		"""Used to compute the performance of the ensemble model on the test set."""
		checkpoint_paths = [os.path.join(self.export_path, f"model.{f_idx + 1}.ckpt") for f_idx in range(self.num_folds)]
		voting_model = EnsembleVotingModel(type(self.trainer.lightning_module), checkpoint_paths)
		voting_model.trainer = self.trainer
		# This requires to connect the new model and move it the right device.
		self.trainer.strategy.connect(voting_model)
		self.trainer.strategy.model_to_device()
		self.trainer.test_loop.run()
		# self.trainer.save_checkpoint(os.path.join(self.export_path, 'votingmodel.pt'))

	def on_save_checkpoint(self) -> Dict[str, int]:
		return {"current_fold": self.current_fold}

	def on_load_checkpoint(self, state_dict: Dict) -> None:
		self.current_fold = state_dict["current_fold"]

	def _reset_fitting(self) -> None:
		self.trainer.reset_train_dataloader()
		self.trainer.reset_val_dataloader()
		self.trainer.state.fn = TrainerFn.FITTING
		self.trainer.training = True

	def _reset_testing(self) -> None:
		self.trainer.reset_test_dataloader()
		self.trainer.state.fn = TrainerFn.TESTING
		self.trainer.testing = True

	def __getattr__(self, key) -> Any:
		# requires to be overridden as attributes of the wrapped loop are being accessed.
		if key not in self.__dict__:
			return getattr(self.fit_loop, key)
		return self.__dict__[key]

	def __setstate__(self, state: Dict[str, Any]) -> None:
		self.__dict__.update(state)


def kfold_masks(K: int, N: int) -> torch.Tensor:
	"""Generates masks for kfold validation

	Parameters
	----------
	K : int
		number of folds
	N : int
		length of the dataset

	Returns
	-------
	torch.Tensor[dtype=torch.bool, size=(K, N)]
		training masks (use ``~mask`` to get validation masks)
	"""

	masks = torch.empty((K, N), dtype=torch.bool)
	permutation = torch.randperm(N)
	for k in range(K):
		mask = torch.ones(N, dtype=torch.bool)
		mask[int(k*N/K):int((k+1)*N/K)] = 0
		masks[k, :] = mask[permutation]
	return masks