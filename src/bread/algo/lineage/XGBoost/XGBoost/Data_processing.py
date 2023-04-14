# -*- coding: utf-8 -*-
"""
Created on Tue Jan 17 14:57:10 2023

@author: gligorov
"""
import numpy as np
import itertools

def generate_data(n_samples=1000, n_features=5, imbalance=0.5):
    """
    Generates random binary classification data
    Imbalance sets how many samples have which label)
    """
    np.random.seed(0)
    X = np.random.rand(n_samples, n_features)
    y = np.random.binomial(1, imbalance, size=n_samples)
    return X, y

def flatten_3d_array(arr):
    """
    Flattens a 3-dimensional numpy array while keeping the first dimension unchanged
    """
    shape = arr.shape
    new_shape = (shape[0], np.prod(shape[1:]))
    return arr.reshape(new_shape)

def permute_matrix(matrix, row_id):
    """
    Generates all possible permutations of a matrix  rows
    it takes row_index as input, which is a one-hot encoded label for the classification 
    and outputs the one-hot encoded labels of the permutated matrices
    """
    # Get the number of rows in the matrix
    rows = len(matrix)
    
    # Get all possible permutations of the row indices
    permutations = list(itertools.permutations(range(rows)))

    # Use list comprehension to create a list of all permuted matrices
    permuted_matrices = [np.array([matrix[i] for i in permutation]) for permutation in permutations]

    # Use list comprehension to find the index of the specified row in each permuted matrix
    row_indices = [list(permutation).index(row_id) for permutation in permutations]
    
    
    return permuted_matrices, row_indices

def generate_all_permutations(data, labels):
    """
    Generates all posible permutations for matrices in data 
    and the corresponding labels
    Labels should be integers
    """
    
    permuted_matrices_list = []
    permuted_labels_list = []
    
    for matrix, label in zip(data, labels):
        permuted_matrices, permuted_labels = permute_matrix(matrix, label)
        permuted_matrices_list.extend(permuted_matrices)
        permuted_labels_list.extend(permuted_labels)
        
    return np.array(permuted_matrices_list), np.array(permuted_labels_list)
        
def count_classes(y):
    """
    Counts number of zeros and ones in binary classification dataset
    """
    n_zeros = (y == 0).sum()
    n_ones = (y == 1).sum()
    return n_zeros, n_ones          

def keep_features(matrices, feature_columns = [0,1,2,3]):
    """
    For a list of bud matrices keep only certain features
    """
    new_matrices = np.zeros((matrices.shape[0], matrices.shape[1], len(feature_columns)))
    for i, fid in enumerate(feature_columns):
        new_matrices[:,:,i] = matrices[:,:,fid]
    return new_matrices