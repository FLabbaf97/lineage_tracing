# -*- coding: utf-8 -*-
"""
Created on Thu Jan 19 12:23:50 2023

@author: gligorov
"""
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def perform_PCA(matrices, n_components):
    """
    Perform PCA on the dataset and return feature importance
    """
    #Extract rows from all matrices
    X = [row for matrix in matrices for row in matrix]
    
    # Subtract mean from data
    X = X - np.mean(X, axis=0)
    
    # Compute covariance matrix
    cov_matrix = np.cov(X, rowvar=False)
    
    # Compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
    
    # Compute feature importance
    feature_importance = eigenvectors[:, :n_components]
    
    # Compute principal components
    principal_components = np.dot(X, feature_importance)
    
    # Sort eigenvalues and eigenvectors in descending order
    sort_indices = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[sort_indices]
    eigenvectors = eigenvectors[:, sort_indices]
    principal_components = np.dot(X, eigenvectors[:, :n_components])
    
    # Compute explained variance
    explained_variance = eigenvalues[:n_components] / np.sum(eigenvalues)
    
    return principal_components, feature_importance, explained_variance

def visualize_feature_importance(feature_importance):
    """
    Visualize the feature importance of PCA
    """
    sns.heatmap(feature_importance, cmap='bwr', annot=True)
    plt.xlabel('PC')
    plt.ylabel('Features')
    plt.title('Loadings of the Principal Components')
    plt.show()

def visualize_PCA(principal_components):
    """
    Visualize the results of PCA by taking into account first two components
    """
    x = principal_components[:, 0]
    y = principal_components[:, 1]
    plt.scatter(x, y)
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('PCA Visualization')
    plt.show() 
    
#Load data
folder = '../data/'
data_full = np.load(folder + 'out_full.npy')
data_small = np.load(folder + 'out_small.npy')
labels = np.load(folder + 'out_GT.npy')

#Changing axes order 
data_full = np.moveaxis(data_full, [0, 1, 2], [1, 2, 0])
data_small = np.moveaxis(data_small, [0, 1, 2], [1, 2, 0])

n_components = 3
principal_components, feature_importance, explained_variance = perform_PCA(data_small, n_components)
print(explained_variance)
visualize_feature_importance(feature_importance)
visualize_PCA(principal_components)
