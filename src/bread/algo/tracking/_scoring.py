from ._classifier import AssignmentClassifier
from ._dataset import AssignmentDataset
from torch.utils.data import Subset

__all__ = ['accuracy_assignment', "f1_assignment"]

def accuracy_assignment(net: AssignmentClassifier, subset: Subset, y=None) -> float:
	tp, tn, fp, fn = 0, 0, 0, 0

	for graph in subset:
		yhat = net.predict_assignment(graph)
		y = graph.y.cpu().numpy().reshape(yhat.shape)
		tp += ((yhat == 1) & (y == 1)).sum()
		fp += ((yhat == 1) & (y == 0)).sum()
		tn += ((yhat == 0) & (y == 0)).sum()
		fn += ((yhat == 0) & (y == 1)).sum()

	return (tp + tn) / (tp + tn + fp + fn)

def f1_assignment(net: AssignmentClassifier, subset: Subset, y=None) -> float:
	tp, tn, fp, fn = 0, 0, 0, 0

	for graph in subset:
		yhat = net.predict_assignment(graph)
		y = graph.y.cpu().numpy().reshape(yhat.shape)
		tp += ((yhat == 1) & (y == 1)).sum()
		fp += ((yhat == 1) & (y == 0)).sum()
		tn += ((yhat == 0) & (y == 0)).sum()
		fn += ((yhat == 0) & (y == 1)).sum()

	return (tp*2) / (2*tp + fp + fn)