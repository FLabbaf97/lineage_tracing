from bread.algo.tracking import AssignmentDataset
from skorch import NeuralNetClassifier
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
import numpy as np, scipy.optimize, scipy.sparse
from typing import List, Union

__all__ = ['AssignmentClassifier', 'GraphLoader']

class AssignmentClassifier(NeuralNetClassifier):
	def predict_assignment(self, data: Union[AssignmentDataset, Data]) -> Union[List[np.ndarray], np.ndarray]:
		if not isinstance(data, Data):
			return [self.predict_assignment(graph) for graph in data ]

		graph = data
		n1, n2 = len(graph.cell_ids1), len(graph.cell_ids2)
		z = self.evaluation_step((graph, None), training=False).cpu().numpy()  # perform raw forward pass
		scores = z.reshape((n1, n2))
		yx = scipy.optimize.linear_sum_assignment(scores, maximize=True)
		assignment = scipy.sparse.bsr_matrix(([1] * len(yx[0]), yx), shape=(n1, n2), dtype=bool).toarray().astype(int)
		
		return assignment

class GraphLoader(DataLoader):
	# we need this class to load the graph data into the training loop, because graphs are dynamically sized and can't be stored as normal numpy arrays
	# https://github.com/skorch-dev/skorch/blob/8db8a0d4d23e696c54cc96494b54a83f5ac55d69/notebooks/CORA-geometric.ipynb
	def __iter__(self):
		it = super().__iter__()
		for graph in it:
			yield graph, graph.y.float()