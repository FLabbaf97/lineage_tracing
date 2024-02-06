from bread.algo.tracking import AssignmentDataset
from skorch import NeuralNetClassifier
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
import numpy as np, scipy.optimize, scipy.sparse
from typing import List, Union

__all__ = ['AssignmentClassifier', 'GraphLoader','seed_torch']

def seed_torch(seed=42):
    import random
    import os
    import numpy as np
    import torch
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

class AssignmentClassifier(NeuralNetClassifier):
    def predict_assignment(self, data: Union[AssignmentDataset, Data], assignment_method: str = 'hungarian') -> Union[List[np.ndarray], np.ndarray]:
        if not isinstance(data, Data):
            return [self.predict_assignment(graph) for graph in data ]

        graph = data
        cell_ids1 = set([ i for i, a in list(graph.cell_ids) ])
        cell_ids2 = set([ a for i, a in list(graph.cell_ids)])
        n1, n2 = len(cell_ids1), len(cell_ids2)
        z = self.evaluation_step((graph, None), training=False).cpu().numpy()  # perform raw forward pass
        scores = z.reshape((n1, n2))
        if(assignment_method == 'hungarian'):
            assignment = self.hungarian(scores,n1,n2)
        elif(assignment_method == 'modified_hungarian'):
            assignment = self.modified_hungarian(scores,n1,n2)
        elif(assignment_method == 'Jonker_Volgenant'):
            assignment = self.Jonker_Volgenant(scores,n1,n2)
        elif(assignment_method == 'squared_hungarian'):
            assignment = self.squared_hungarian(scores,n1,n2)
        elif(assignment_method == 'modified_hungarian2'):
            assignment = self.modified_hungarian2(scores,n1,n2)
        elif(assignment_method == 'percentage_hungarian'):
            assignment = self.percentage_hungarian(scores,n1,n2)
        else:
            raise ValueError(f'assignment_method {assignment_method} not recognized')
        return assignment
    
    def predict_raw(self, data: Union[AssignmentDataset, Data]) -> Union[List[np.ndarray], np.ndarray]:
        if not isinstance(data, Data):
            return [self.predict_assignment(graph) for graph in data ]

        graph = data
        cell_ids1 = set([ i for i, a in list(graph.cell_ids) ])
        cell_ids2 = set([ a for i, a in list(graph.cell_ids)])
        n1, n2 = len(cell_ids1), len(cell_ids2)
        z = self.evaluation_step((graph, None), training=False).cpu().numpy()  # perform raw forward pass
        scores = z.reshape((n1, n2))
        return scores

    def percentage_hungarian(self, scores: np.ndarray, n1: int ,n2: int) -> np.ndarray:
        # Create a copy of the scores matrix
        scores_copy = scores.copy()

        # Solving the LAP using the Jonker-Volgenant algorithm
        row_ind, col_ind = scipy.optimize.linear_sum_assignment(scores_copy, maximize=True)  # maximizing the total assignment score

        # 'row_ind' and 'col_ind' are the row and column indices of the optimal assignments.
        # Create the binary assignment matrix
        binary_assignment_matrix = np.zeros_like(scores, dtype=int)
        binary_assignment_matrix[row_ind, col_ind] = 1

        # Calculate the percentage of assignment scores that are higher for each assigned cell
        percentage_higher_scores = []
        threshold = 20  # Adjust the threshold as needed
        for i, j in zip(row_ind, col_ind):
            if binary_assignment_matrix[i, j] == 1:
                count_higher_scores = np.sum(scores_copy[:, j] > scores[i, j])
                percentage = (count_higher_scores / (scores.shape[0] - 1)) * 100  # excluding the current cell
                percentage_higher_scores.append(percentage)
                if percentage > threshold:
                    binary_assignment_matrix[:, j] = 0  # Set all assignments to this cell to 0
        print(percentage_higher_scores)
        print("min_percentage_higher_scores: ", np.min(percentage_higher_scores), "max_percentage_higher_scores: ", np.max(percentage_higher_scores), "average_percentage_higher_scores: ", np.mean(percentage_higher_scores))
        return binary_assignment_matrix
    
    # def percentage_hungarian(self, scores: np.ndarray, n1: int ,n2: int, threshold: float=20) -> np.ndarray:
    #     # Create a copy of the scores matrix
    #     scores_copy = scores.copy()

    #     # Solving the LAP using the Jonker-Volgenant algorithm
    #     row_ind, col_ind = scipy.optimize.linear_sum_assignment(scores_copy, maximize=True)  # maximizing the total assignment score

    #     # Create the binary assignment matrix
    #     binary_assignment_matrix = np.zeros_like(scores, dtype=int)
    #     binary_assignment_matrix[row_ind, col_ind] = 1

    #     # Calculate the percentage of assignment scores that are higher for each assigned cell
    #     comparison_matrix = scores_copy > scores[np.newaxis, col_ind, :]
    #     percentage_higher_scores = (np.sum(comparison_matrix, axis=0) / (scores.shape[0] - 1)) * 100

    #     # Apply the threshold to the binary assignment matrix
    #     exceeding_threshold = percentage_higher_scores > threshold
    #     binary_assignment_matrix[:, exceeding_threshold] = 0

    #     print("percentage_higher_scores:", percentage_higher_scores)
    #     print("min_percentage_higher_scores:", np.min(percentage_higher_scores), "max_percentage_higher_scores:", np.max(percentage_higher_scores), "average_percentage_higher_scores:", np.mean(percentage_higher_scores))

    #     return binary_assignment_matrix

    def hungarian(self, scores: np.ndarray, n1: int ,n2: int) -> np.ndarray:
        yx = scipy.optimize.linear_sum_assignment(scores, maximize=True)
        assignment = scipy.sparse.bsr_matrix(([1] * len(yx[0]), yx), shape=(n1, n2), dtype=bool).toarray().astype(int)
        return assignment

    def modified_hungarian(self, scores: np.ndarray, n1: int ,n2: int, threshold: float = 0.01) -> np.ndarray:
        scores_modified = scores.copy()
        scores_modified = 1 / (1 + np.exp(-scores_modified))  # apply sigmoid function
        scores_modified[scores_modified < threshold] = 0
        yx = scipy.optimize.linear_sum_assignment(scores_modified, maximize=True)
        assignment = scipy.sparse.bsr_matrix(([1] * len(yx[0]), yx), shape=(n1, n2), dtype=bool).toarray().astype(int)
        # create a boolean mask of elements that are zero in scores_modified
        mask = scores_modified == 0
        # set the corresponding elements in assignment to zero
        assignment[mask] = 0
        return assignment

    def sigmoid_hungarian(self, scores: np.ndarray, n1: int ,n2: int,) -> np.ndarray:
        scores_modified = scores.copy()
        scores_modified = 1 / (1 + np.exp(-scores_modified))  # apply sigmoid function
        yx = scipy.optimize.linear_sum_assignment(scores_modified, maximize=True)
        assignment = scipy.sparse.bsr_matrix(([1] * len(yx[0]), yx), shape=(n1, n2), dtype=bool).toarray().astype(int)
        return assignment

    def squared_hungarian(self, scores: np.ndarray, n1: int ,n2: int) -> np.ndarray:
        scores_modified = scores.copy()
        scores_modified = 1 / (1 + np.exp(-scores_modified))  # apply sigmoid function
        scores = scores_modified**2
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