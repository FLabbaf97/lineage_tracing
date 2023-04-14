# -*- coding: utf-8 -*-
"""
Created on Tue Jan 17 14:57:10 2023

@author: gligorov
"""
import numpy as np
import itertools

def generate_random_data(n_instances, num_classes, num_features, predictable = False):
    """
    Generates random data with a structure that corresponds the one that will be used for training
    if predictable set to True, then the NN can learn to predict. Otherwise, labels are random 
    and expected accuracy should be around 100%/num_classes
    """
    data = []
    labels = []
    
    for i in range(n_instances):
        #Number of vectors is anywhere between 3 and number of classes
        n_vectors = np.random.randint(3, num_classes+1)
        
        #We create a matrix with n_vectors rows and num_features columns
        
        instance = np.random.rand(n_vectors, num_features)
        
        #If predictable is set to True, the classification can get more than 90% accurate
        if predictable:
            #if you want to test on a permutationally invariant network then you should give a permutationally invariant task
            #for example, predict the value of the sine of the index of the largest column
            label =  int(np.argmax(np.sin(np.sum(instance, axis = 1)))) #or the one that is not: int(np.ceil(instance[0,0] + instance[0,1] + instance[0,2]))
        else: #label is randomly drawn from 0 to num_classes-1 
            label = np.random.randint(0, num_classes)
        
        data.append(instance)
        labels.append(label)
    return np.array(data), np.array(labels)

def sort_matrix(matrix):
    """
    Sorts the padded input data by the value of the first column
    """
    # Create an index array to sort the matrix by the first column
    index_array = np.lexsort((matrix[:, 0],))
    # Sort the matrix using the index array
    sorted_matrix = matrix[index_array]
    # Get the indices of rows that have -1 in the first element
    indices_of_minus_one = np.where(sorted_matrix[:, 0] == -1)[0]
    # Split the matrix into two parts, one containing rows with -1 and one without
    minus_one_rows = sorted_matrix[indices_of_minus_one]
    other_rows = np.delete(sorted_matrix, indices_of_minus_one, axis=0)
    # Concatenate the two parts of the matrix, putting the rows with -1 at the bottom
    final_matrix = np.concatenate((other_rows, minus_one_rows))
    return final_matrix	
	
def data_generator(data, labels, shuffle = False, force_n_classes = 0):
    """
    Prepares the data to be compatible with the expected NN input
    Encodes the labels into one-hot labels
    """
    # data_size is the number of instances in the data
    data_size = len(data)
    
    if(shuffle == True):
        # create an array of indices for the data
        indices = np.arange(data_size)
        
        # shuffle the indices randomly
        np.random.shuffle(indices)
        
        # using the shuffled indices, shuffle the data and labels
        data = [data[i] for i in indices]
        labels = [labels[i] for i in indices]
        
    #If we don't specify max number of classes, then it will be correspond to the number of motehers for a cell with max number of mothers
    if(force_n_classes == 0):
        # find the maximum number of vectors in any instance
        max_vectors = max([len(x) for x in data])
        # number of classes corresponds to maximal number of vectors
        num_classes = max_vectors
    else:
        num_classes = force_n_classes

    print('Number of classes is ', str(num_classes))

    # pad all instances with -1 vectors so that every instance has force_n_classes number of vectors
    data = [np.pad(x, [(0, num_classes - len(x)), (0, 0)], mode='constant', constant_values=-1) for x in data]
    
    # sort the vectors in each instance by the first element of the vector
    for x in data:
        x = sort_matrix(x)
        x = x[:num_classes, :] #we keep only first num_classses rows
        
    # one-hot encode the labels
    #This means we should later use categorical crossentropy as the loss function
    labels = np.array([np.eye(num_classes)[i] for i in labels])
    
    # return the sets and their labels
    return data, labels

def keep_features(matrices, feature_columns = [0,1,2,3]):
    """
    For a list of bud matrices keep only certain features
    """
    new_matrices = np.zeros((matrices.shape[0], matrices.shape[1], len(feature_columns)))
    for i, fid in enumerate(feature_columns):
        new_matrices[:,:,i] = matrices[:,:,fid]
    return new_matrices

def one_hot_encode(label, num_classes):
    """
    Encode a label into a one-hot encoding.
    :param label: The label to encode (int)
    :param num_classes: The number of classes in the dataset
    :return: The one-hot encoded label (numpy array)
    """
    one_hot = np.zeros(num_classes)
    one_hot[label] = 1
    return one_hot

def one_hot_decode(one_hot):
    """
    Decode a one-hot encoded label into a single label.
    :param one_hot: The one-hot encoded label (numpy array)
    :return: The decoded label (int)
    """
    return np.argmax(one_hot)

def permute_matrix(matrix, row_index):
    """
    Generate all possible permutations of a matrix rows
    it takes row_index as input, which is a one-hot encoded label for the classification 
    and outputs the one-hot encoded labels of the permuted matrices
    """
    
    # Get the number of rows in the matrix
    rows = len(matrix)
    row_id = one_hot_decode(row_index)
    num_classes = len(row_index)
    # Get all possible permutations of the row indices
    permutations = list(itertools.permutations(range(rows)))

    # Use list comprehension to create a list of all permuted matrices
    permuted_matrices = [np.array([matrix[i] for i in permutation]) for permutation in permutations]

    # Use list comprehension to find the index of the specified row in each permuted matrix
    row_indices = [list(permutation).index(row_id) for permutation in permutations]
    
    #One-hot encode the row indices
    encoded_row_indices = [one_hot_encode(row_id_, num_classes) for row_id_ in row_indices]
    
    return permuted_matrices, encoded_row_indices

def generate_all_permutations(data, labels):
    """
    Generate all permutation for a given matrix
    output also the label which correspond to the row_id in the permuted matrix
    """
    
    permuted_matrices_list = []
    permuted_labels_list = []
    
    for matrix, label in zip(data, labels):
        permuted_matrices, permuted_labels = permute_matrix(matrix, label)
        permuted_matrices_list.extend(permuted_matrices)
        permuted_labels_list.extend(permuted_labels)
    return np.array(permuted_matrices_list), np.array(permuted_labels_list)


        
        