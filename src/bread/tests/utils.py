import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import xgboost as xgb
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from bread.algo.lineage import LineageGuesserML, LineageGuesserNearestCell
from bread.data import Segmentation, Lineage, SegmentationFile
import itertools



def extract_features_for_dist_threshold(segmentation_path, args):
    candidate_features = pd.DataFrame(columns=['bud_id', 'candid_id', 'time_id', 'feature1', 'feature2',
                                      'feature3', 'feature4', 'feature5', 'feature6', 'feature7', 'feature8', 'feature9', 'feature10'])
    segmentation = SegmentationFile.from_h5(
        segmentation_path).get_segmentation('FOV0')
    guesser = LineageGuesserML(
        segmentation=segmentation,
        nn_threshold=args["nn_threshold"],
        flexible_nn_threshold=args["flexible_nn_threshold"],
        num_frames_refractory=args["num_frames_refractory"],
        num_frames=args["num_frames"],
        bud_distance_max=args["bud_distance_max"]
    )
    bud_ids, time_ids = segmentation.find_buds(
    ).bud_ids, segmentation.find_buds().time_ids
    for i, (bud_id, time_id) in enumerate(zip(bud_ids, time_ids)):
        frame_range = guesser.segmentation.request_frame_range(
            time_id, time_id + guesser.num_frames)
        num_frames_available = guesser.num_frames
        if len(frame_range) < 2:
            # raise NotEnoughFramesException(bud_id, time_id, guesser.num_frames, len(frame_range))
            print("Not enough frames for bud {} at time {}. Only {} frames available.".format(
                bud_id, time_id, len(frame_range)))
            continue

        if len(frame_range) < guesser.num_frames:
            num_frames_available = len(frame_range)
            # warnings.warn(NotEnoughFramesWarning(bud_id, time_id, guesser.num_frames, len(frame_range)))
            # print("Not enough frames for bud {} at time {}. Only {} frames available.".format(
            #     bud_id, time_id, len(frame_range)))
        # check the bud still exists !
        for time_id_ in frame_range:
            if bud_id not in guesser.segmentation.cell_ids(time_id_):
                # raise LineageGuesserExpansionSpeed.BudVanishedException(bud_id, time_id_)
                print("Bud {} vanished at time {}".format(bud_id, time_id_))
        selected_times = [i for i in range(
            time_id, time_id + num_frames_available)]
        candidate_parents = guesser._candidate_parents(
            time_id, nearest_neighbours_of=bud_id)
        summary_features = np.zeros(
            (len(candidate_parents), guesser.number_of_features), dtype=np.float64)
        for c_id, candidate in enumerate(candidate_parents):
            try:
                features, _ = guesser._get_ml_features(
                    bud_id, candidate, time_id, selected_times)
                new_row = {'bud_id': bud_id, 'candid_id': candidate, 'time_id': time_id, 'feature1': features[0], 'feature2': features[1], 'feature3': features[2], 'feature4': features[
                    3], 'feature5': features[4], 'feature6': features[5], 'feature7': features[6], 'feature8': features[7], 'feature9': features[8], 'feature10': features[9]}
                new_df = pd.DataFrame(new_row, index=[0])
                candidate_features = pd.concat([candidate_features, new_df])
            except Exception as e:
                print("Error for bud {} at time {} with candidate {}: {}".format(
                    bud_id, time_id, candidate, e))
    return candidate_features


def extract_features_for_count_threshold(segmentation_path, num_nn = 6, num_frames=4, filling_features=[100,100,0,0,0,-1,1,-1,-1,-1]):
    candidate_features = pd.DataFrame(columns=['bud_id', 'candid_id', 'time_id', 'feature1', 'feature2',
                                      'feature3', 'feature4', 'feature5', 'feature6', 'feature7', 'feature8', 'feature9', 'feature10'])
    segmentation = SegmentationFile.from_h5(
        segmentation_path).get_segmentation('FOV0')
    guesser = LineageGuesserML(
        segmentation=segmentation,
        nn_threshold=100.0,
        flexible_nn_threshold=False,
        num_frames_refractory=0,
        num_frames=num_frames,
        bud_distance_max=100.0
    )
    bud_ids, time_ids = segmentation.find_buds(
    ).bud_ids, segmentation.find_buds().time_ids
    for i, (bud_id, time_id) in enumerate(zip(bud_ids, time_ids)):
        frame_range = guesser.segmentation.request_frame_range(
            time_id, time_id + guesser.num_frames)
        num_frames_available = guesser.num_frames
        if len(frame_range) < 2:
            # raise NotEnoughFramesException(bud_id, time_id, guesser.num_frames, len(frame_range))
            print("Not enough frames for bud {} at time {}. Only {} frames available.".format(
                bud_id, time_id, len(frame_range)))
            continue
        if len(frame_range) < guesser.num_frames:
            num_frames_available = len(frame_range)
            # warnings.warn(NotEnoughFramesWarning(bud_id, time_id, guesser.num_frames, len(frame_range)))
            # print("Not enough frames for bud {} at time {}. Only {} frames available.".format(
            #     bud_id, time_id, len(frame_range)))
            
        # check the bud still exists !
        for time_id_ in frame_range:
            if bud_id not in guesser.segmentation.cell_ids(time_id_):
                # raise LineageGuesserExpansionSpeed.BudVanishedException(bud_id, time_id_)
                print("Bud {} vanished at time {}".format(bud_id, time_id_))
        selected_times = [i for i in range(
            time_id, time_id + num_frames_available)]
        candidate_parents = guesser._candidate_parents(
            time_id, nearest_neighbours_of=bud_id, num_nn=num_nn, threshold_mode='count')
        # summary_features = np.zeros(
        #     (len(candidate_parents), guesser.number_of_features), dtype=np.float64)
        # initialize the features with the filling features
        num_rows = len(candidate_parents)
        num_cols = guesser.number_of_features
        summary_features = np.full((num_rows, num_cols), filling_features, dtype=np.float64)

        for c_id, candidate in enumerate(candidate_parents):
            try:
                features, _ = guesser._get_ml_features(
                    bud_id, candidate, time_id, selected_times)
                new_row = {'bud_id': bud_id, 'candid_id': candidate, 'time_id': time_id, 'feature1': features[0], 'feature2': features[1], 'feature3': features[2], 'feature4': features[
                    3], 'feature5': features[4], 'feature6': features[5], 'feature7': features[6], 'feature8': features[7], 'feature9': features[8], 'feature10': features[9]}
                new_df = pd.DataFrame(new_row, index=[0])
                candidate_features = pd.concat([candidate_features, new_df])
            except Exception as e:
                print("Error for bud {} at time {} with candidate {}: {}".format(
                    bud_id, time_id, candidate, e))
    return candidate_features

def extract_all_features_for_count_threshold(segmentation_path, num_nn = 6, num_frames=4, threshold=100):
    candidate_features = pd.DataFrame()
    segmentation = SegmentationFile.from_h5(
        segmentation_path).get_segmentation('FOV0')
    guesser = LineageGuesserML(
        segmentation=segmentation,
        nn_threshold=threshold,
        flexible_nn_threshold=False,
        num_frames_refractory=0,
        num_frames=num_frames,
        bud_distance_max=threshold
    )
    bud_ids, time_ids = segmentation.find_buds(
    ).bud_ids, segmentation.find_buds().time_ids
    f_list = []
    for i, (bud_id, time_id) in enumerate(zip(bud_ids, time_ids)):
        frame_range = guesser.segmentation.request_frame_range(
            time_id, time_id + guesser.num_frames)
        num_frames_available = guesser.num_frames
        if len(frame_range) < 2:
            # raise NotEnoughFramesException(bud_id, time_id, guesser.num_frames, len(frame_range))
            print("Not enough frames for bud {} at time {}. Only {} frames available.".format(
                bud_id, time_id, len(frame_range)))
            continue
        if len(frame_range) < guesser.num_frames:
            num_frames_available = len(frame_range)

        # check the bud still exists !
        for time_id_ in frame_range:
            if bud_id not in guesser.segmentation.cell_ids(time_id_):
                # raise LineageGuesserExpansionSpeed.BudVanishedException(bud_id, time_id_)
                print("Bud {} vanished at time {}".format(bud_id, time_id_))
        selected_times = [i for i in range(
            time_id, time_id + num_frames_available)]
        candidate_parents = guesser._candidate_parents(
            time_id, nearest_neighbours_of=bud_id, num_nn=num_nn, threshold_mode='count')
        for c_id, candidate in enumerate(candidate_parents):
            try:
                features, f_list = guesser._get_all_possible_features(
                    bud_id, candidate, time_id, selected_times)
                new_row = {'bud_id': bud_id, 'candid_id': candidate, 'time_id': time_id}
                new_row.update(features)
                new_df= pd.DataFrame(new_row, index=[0])
                candidate_features = pd.concat([candidate_features, new_df])
            except Exception as e:
                print("Error for bud {} at time {} with candidate {}: {}".format(
                    bud_id, time_id, candidate, e))
    return candidate_features, f_list


def find_nearest_neighbors(segmentation_path, args):
    segmentation = SegmentationFile.from_h5(
        segmentation_path).get_segmentation('FOV0')
    guesser = LineageGuesserNearestCell(
        segmentation=segmentation,
        bud_distance_max=100.0
    )
    bud_ids, time_ids = segmentation.find_buds(
    ).bud_ids, segmentation.find_buds().time_ids
    nn_df = pd.DataFrame(columns=['bud_id', 'time_id', 'nearest_cell'])
    nn_df['bud_id'] = bud_ids
    nn_df['time_id'] = time_ids
    nearest_cells = []
    for i, (bud_id, time_id) in enumerate(zip(bud_ids, time_ids)):
        # find neighbors
        parent, dist = guesser.guess_parent(bud_id, time_id, num_nn=4)
        nearest_cells.append(parent)
    nn_df['nearest_cell'] = nearest_cells
    return nn_df

def get_matrix_features(features_all, lineage_gt):
    # Generate np array of feature sets for each bud
    df1 = lineage_gt.copy()
    # remove the rows with parent_GT = -1 (no parent) and the rows with candid_GT = -2 (disappearing buds)
    df1 = df1.loc[df1.parent_GT != -1]
    df1 = df1.loc[df1.parent_GT != -2]
    df2 = features_all.copy()

    features_list = []
    parent_index_list = []
    candidate_list = []
    for bud, colony in df1[['bud_id', 'colony']].values:
        bud_data = df2.loc[(df2['bud_id'] == bud) & (df2['colony'] == colony)]
        candidates = bud_data['candid_id'].to_numpy()
        if candidates.shape[0] < 4:
            candidates = np.pad(
                candidates, ((0, 4 - candidates.shape[0])), mode='constant', constant_values=-3)
        features = bud_data[['feature1', 'feature2', 'feature3', 'feature4', 'feature5',
                             'feature6', 'feature7', 'feature8', 'feature9', 'feature10']].to_numpy()
        if features.shape[0] < 4:
            features = np.pad(features, ((
                0, 4 - features.shape[0]), (0, 0)), mode='constant', constant_values=-1)
        if features.shape[0] > 4:
            sorted_indices = np.argsort(features[:, 0])
            print('more than 4 candidates', bud, colony, candidates, sorted_indices, int(df1.loc[(df1['bud_id'] == bud) & (
                df1['colony'] == colony), 'parent_GT']))
            # slice the top 4 rows
            k = 4
            features = features[sorted_indices[:k]]
            candidates = candidates[sorted_indices[:k]]

        # parent = int(df1.loc[(df1['bud_id'] == bud) & (
        #     df1['colony'] == colony), 'parent_GT'])
        parent = int(df1.loc[(df1['bud_id'] == bud) & (df1['colony'] == colony), 'parent_GT'].iloc[0])
        # print(bud, colony, parent)
        # print(candidates)
        if(parent not in candidates):
            print('parent not in candidates', bud, colony, candidates, parent)
            # remove this from the df
            df1.drop(df1.loc[(df1['bud_id'] == bud) & (df1.colony == colony)].index,
                     inplace=True)
            continue
        else:
            parent_index = np.where(candidates == parent)[0][0]

        parent_index_list.append(parent_index)
        features_list.append(features)
        candidate_list.append(candidates)
    df1['features'] = features_list
    df1['candidates'] = candidate_list
    df1['parent_index_in_candidates'] = parent_index_list
    return df1

def get_custom_matrix_features(features_all, lineage_gt, feature_list, filling_features=[0 for i in range(100)]):
    # Generate np array of feature sets for each bud
    df1 = lineage_gt.copy()
    # remove the rows with parent_GT = -1 (no parent) and the rows with candid_GT = -2 (disappearing buds)
    df1 = df1.loc[df1.parent_GT != -1]
    df1 = df1.loc[df1.parent_GT != -2]
    df2 = features_all.copy()
    features_list = []
    filling_features = filling_features[:len(feature_list)]
    parent_index_list = []
    candidate_list = []
    for bud, colony in df1[['bud_id', 'colony']].values:
        bud_data = df2.loc[(df2['bud_id'] == bud) & (df2['colony'] == colony)]
        candidates = bud_data['candid_id'].to_numpy()
        if candidates.shape[0] < 4:
            candidates = np.pad(
                candidates, ((0, 4 - candidates.shape[0])), mode='constant', constant_values=-3)
        features = bud_data[feature_list].to_numpy()
        if features.shape[0] < 4:
            n_rows = 4 - features.shape[0]
            rows_to_add = np.array([filling_features for i in range(n_rows)])
            features = np.concatenate((features, rows_to_add), axis=0)
        if features.shape[0] > 4:
            sorted_indices = np.argsort(features[:, 0])
            print('more than 4 candidates', bud, colony, candidates, sorted_indices, int(df1.loc[(df1['bud_id'] == bud) & (
                df1['colony'] == colony), 'parent_GT']))
            # slice the top 4 rows
            k = 4
            features = features[sorted_indices[:k]]
            candidates = candidates[sorted_indices[:k]]

        parent = int(df1.loc[(df1['bud_id'] == bud) & (df1['colony'] == colony), 'parent_GT'].iloc[0])
        if(parent not in candidates):
            df1.drop(df1.loc[(df1['bud_id'] == bud) & (df1.colony == colony)].index,
                     inplace=True)
            continue
        else:
            parent_index = np.where(candidates == parent)[0][0]

        parent_index_list.append(parent_index)
        features_list.append(features)
        candidate_list.append(candidates)
    df1['features'] = features_list
    df1['candidates'] = candidate_list
    df1['parent_index_in_candidates'] = parent_index_list
    return df1




def get_age_related_features(candidate_features, lineage_gt):
    times = lineage_gt['time_index'].to_numpy()
    bud_ids = lineage_gt['bud_id'].to_numpy()
    colony_ids = lineage_gt['colony'].to_numpy()
    parent_ids = lineage_gt['parent_GT'].to_numpy()
    for index, row in candidate_features.iterrows():
        bud_id = row['bud_id']
        colony_id = row['colony']
        time_id = row['time_index']
        parent_id = row['parent_GT']
        # find age of candidate
        bud_candidates = candidate_features.loc[(candidate_features['bud_id'] == bud_id) & (
            candidate_features['colony'] == colony_id) & (candidate_features['time_index'] == time_id)]
        for row in bud_candidates:
            candidate_id = row['candid_id']
            candidate_birth_time = lineage_gt.loc[(lineage_gt['bud_id'] == candidate_id) & (
                lineage_gt['colony'] == colony_id) , 'time_index'].values[0]
            candidate_age = row['time_index'] - candidate_birth_time
            candidate_divisions = lineage_gt.loc[(lineage_gt['parent_GT'] == candidate_id) & (lineage_gt['colony'] == colony_id)].values
            candidate_last_division = 0
            if candidate_divisions.shape[0] == 0:
                candidate_last_division = candidate_birth_time
            else:
                candidate_last_division = np.max(candidate_last_division)


def plot_history(history, what_to_plot = 'all'):
    """
    Plot history of training
    what_to_plot can be set to 'acc', 'loss' or 'lr' (for learning rate)
    or 'all' for all three
    """
    if(what_to_plot == 'all' or what_to_plot == 'acc'):
        #Visualizing the results of training
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('Model accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['train', 'validation'], loc='upper left')
        plt.show()
    
    if(what_to_plot == 'all' or what_to_plot == 'loss'):
        # "Loss"
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['train', 'validation'], loc='upper left')
        plt.show()
    
    if(what_to_plot == 'all' or what_to_plot == 'lr'):
        # "Learning rate
        plt.plot(history.history['lr'])
        plt.title('Learning rate')
        plt.ylabel('lr')
        plt.xlabel('\Epoch')
        plt.show()
        
def print_matrix(s):
    """"
    Print a nice matrix into the console
    """
    # Do heading
    print("     ", end="")
    for j in range(len(s[0])):
        print("%5d " % j, end="")
    print()
    print("     ", end="")
    for j in range(len(s[0])):
        print("------", end="")
    print()
    # Matrix contents
    for i in range(len(s)):
        print("%3d |" % (i), end="") # Row nums
        for j in range(len(s[0])):
            print("%.3f " % (s[i][j]), end="")
        print()  
        
def show_classification_results(x, y, ypred, N = 3):

    """
    Plot N randomly chosen wrongly classified x and N randomly chosen correctly classified x
    """
    #Converting from one-hot encoded labels to class indices
    y = y.argmax(axis = 1)
    ypred = ypred.argmax(axis = 1)
    
    #Finding indices where clssification is correct
    results = y==ypred
    correct_i = np.where(results == True)
    wrong_i = np.where(results == False)
    
    #Chosing random indices for plotting
    correct_chosen = np.ceil(np.random.uniform(0, len(correct_i[0]), N))
    wrongly_chosen = np.ceil(np.random.uniform(0, len(correct_i[0]), N))
    
    #Printing chosen correctly classified cells
    for ind in correct_chosen:
        print('Data instance n. ' + str(int(ind)) + ' is correctly classified.')
        print_matrix(x[int(ind)])
    
    #Printing chosen wrongly classified cells        
    for ind in wrongly_chosen:
        print('Data instance n. ' + str(int(ind)) + ' is wrongly classified.')
        print_matrix(x[int(ind)])
            
    #print('Wrongly classified data instances are ' + str(list(wrong_i[0])))
    return correct_i, wrong_i

def grid_train_xgboost(X_train, y_train, X_test, y_test, max_depths=[5, 10], min_child_weights=[1, 2, 5, 10], early_stopping_rounds=10, num_boost_round=100):
    # This funciton performs grid search through parameters that are influencing the probability of overfitting
    # It returns the values for AUC-ROC for the best model and the values used for training

    best_params = None
    best_acc = 0
    for max_d in max_depths:
        for min_child in min_child_weights:
            params = {'objective': 'multi:softmax', 'num_class': 4, 'max_depth': max_d,
                      'min_child_weight': min_child, 'eval_metric': 'merror'}
            dtrain = xgb.DMatrix(X_train, label=y_train)
            dtest = xgb.DMatrix(X_test, label=y_test)
            model = xgb.train(params, dtrain, evals=[(dtest, 'test')],
                              early_stopping_rounds=early_stopping_rounds, verbose_eval=False, num_boost_round=num_boost_round)
            preds = model.predict(dtest)
            acc = accuracy_score(y_test, preds)
            # print('Accuracy with ', params, 'is ', "%.4f " % (acc))
            if acc > best_acc:
                best_acc = acc
                best_params = params

    # print("Best ACC:", best_acc)
    # print("Best params:", best_params)
    return best_params


def flatten_3d_array(arr):
    """
    Flattens a 3-dimensional numpy array while keeping the first dimension unchanged
    """
    if arr.ndim == 1:
        arr2 = np.array([arr[i] for i in range(len(arr))])
        arr = np.stack(arr2)
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
    permuted_matrices = [np.array([matrix[i] for i in permutation])
                         for permutation in permutations]

    # Use list comprehension to find the index of the specified row in each permuted matrix
    row_indices = [list(permutation).index(row_id)
                   for permutation in permutations]

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


def keep_features(matrices, feature_columns=[0, 1, 2, 3]):
    """
    For a list of bud matrices keep only certain features
    """
    new_matrices = np.zeros(
        (matrices.shape[0], matrices.shape[1], len(feature_columns)))
    for i, fid in enumerate(feature_columns):
        new_matrices[:, :, i] = matrices[:, :, fid]
    return new_matrices

