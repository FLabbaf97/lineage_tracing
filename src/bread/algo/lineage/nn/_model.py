from abc import abstractmethod
import torch, torch.optim, torch.nn as nn
import pytorch_lightning as pl
import torchmetrics

__all__ = ['LineageClassifierDeep', 'LineageClassifierLogistic']

class LineageClassifierNN(pl.LightningModule):
	def __init__(self):
		super().__init__()

	def configure_optimizers(self):
		optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
		lr_scheduler = { 'scheduler': torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=self.lr_decay), }
		return [optimizer], [lr_scheduler]

	def training_step(self, batch: 'dict[str, torch.Tensor]', batch_idx):
		l = self.compute_loss(batch)
		a = self.compute_accuracy(batch)
		self.log('train_err', 1-a)
		self.log('train_loss', l)
		return l

	def validation_step(self, batch: 'dict[str, torch.Tensor]', batch_idx):
		l = self.compute_loss(batch)
		a = self.compute_accuracy(batch)
		self.log('val_err', 1-a)
		self.log('val_loss', l)
		return l

	def test_step(self, batch: 'dict[str, torch.Tensor]', batch_idx):
		l = self.compute_loss(batch)
		a = self.compute_accuracy(batch)
		self.log('test_err', 1-a)
		self.log('test_loss', l)

	def forward(self, batch: 'dict[str, torch.Tensor]') -> torch.Tensor:
		return self.layers(batch['x'])

	def compute_loss(self, batch: 'dict[str, torch.Tensor]'):
		return self.loss(self.forward(batch), batch['y'])

	def compute_accuracy(self, batch: 'dict[str, torch.Tensor]'):
		return torchmetrics.functional.accuracy(self.scores(batch), batch['y'], num_classes=1)

	def predict_step(self, batch: 'dict[str, torch.Tensor]'):
		"""Rounds scores for binary classification"""
		z = self.scores(batch)
		yhat = torch.round(z)
		return yhat

	@abstractmethod
	def scores(self, batch: 'dict[str, torch.Tensor]') -> torch.Tensor:
		"""Compute binary classification probability :math:`z = Pr(Y = 0)`"""
		raise NotImplementedError()


class Layer(nn.Module):
	def __init__(self, input_dim: int, output_dim: int, dropout: float):
		super().__init__()
		self.layer = nn.Sequential(
			nn.Linear(input_dim, output_dim),
			nn.BatchNorm1d(output_dim, track_running_stats=True),
			nn.ELU(),
			nn.Dropout1d(p=dropout),
		)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		return self.layer(x)

class LineageClassifierDeep(LineageClassifierNN):
	def __init__(self, config: dict):
		super().__init__()
		self.save_hyperparameters(config)

		self.lr = config['lr']
		self.lr_decay = config['lr_decay']
		# class weights help train on an unbalanced dataset
		self.loss = nn.CrossEntropyLoss(weight=torch.tensor(config['class_weights']))
		# ELU trains faster (I found SELU too tricky to correctly initialize, and we have BatchNorm1d layers anyways)
		# deeper and shallow networks often are better
		# batch normalization helps exploit nonlinearities
		# dropout helps reduce overfitting and interdependencies in the neurons of a layer
		layers = []
		layers += [ Layer(config['input_dim'], config['size_hidden'], config['dropout']) ]
		layers += [ Layer(config['size_hidden'], config['size_hidden'], config['dropout']) ] * (config['num_hidden']-1)
		layers += [  # output layer
			nn.Linear(config['size_hidden'], 2),  # one-hot encoding of classes y=(0, 1)
		]
		self.layers = nn.Sequential(*layers)

	def scores(self, batch: 'dict[str, torch.Tensor]'):
		# CrossEntropy applies SoftMax, then Log, then NLLLoss
		# 2-class SoftMax reduces to binary Sigmoid when we take the diff
		# sigmoid : :math:`\frac{1}{1-e^{-a}}`
		# softmax : :math:`\frac{1}{1-e^{x_0-x_1}}`
		# so :math:`a=x_1-x_0`
		return torch.sigmoid(self.forward(batch).detach().diff().squeeze())


class LineageClassifierLogistic(LineageClassifierNN):
	def __init__(self, config: dict):
		super().__init__()
		self.save_hyperparameters(config, ignore=['input_dim', 'num_hidden', 'size_hidden', 'dropout'])

		self.lr = config['lr']
		self.lr_decay = config['lr_decay']
		self.loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(config['class_weights'][1]))
		self.layers = nn.Sequential(
			nn.BatchNorm1d(25),
			nn.Linear(25, 1),
		)

	def scores(self, batch: 'dict[str, torch.Tensor]'):
		return torch.sigmoid(self.forward(batch).detach().squeeze())

	def compute_loss(self, batch: 'dict[str, torch.Tensor]'):
		z = self.forward(batch)
		return self.loss(z, batch['y'][:, None].type(torch.float))  # z (unnormalized prediction) should match y