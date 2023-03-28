from pathlib import Path
import json, os
from skorch.callbacks import Checkpoint
import torch, numpy as np, random

__all__ = ['SaveHyperParams', 'seed']

class SaveHyperParams(Checkpoint):
	def on_train_begin(self, net, X=None, y=None, **kwargs):
		os.makedirs(self.dirname, exist_ok=True)
		d = {}
		d.update(net.get_params_for('module'))
		d.update(net.get_params_for('optimizer'))
		d.update(net.get_params_for('callbacks__LRScheduler'))
		with open(Path(self.dirname) / 'hyperparams.json', 'w') as file:
			json.dump(d, file)


def seed(s: int):
	# TODO : settings these does not seem to make results totally reproducible
	torch.manual_seed(s)
	np.random.seed(s)
	random.seed(s)