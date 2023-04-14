# -*- coding: utf-8 -*-
"""
Created on Wed Jan 18 23:12:09 2023

@author: gligorov
"""


# -*- coding: utf-8 -*-
"""
Created on Wed Jan 18 14:12:39 2023

@author: gligorov
"""
import xgboost as xgb
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from utils import plot_confusion_matrix, plot_feature_importance, plot_eval_metrics
from Data_processing import keep_features, count_classes, generate_data

def get_vectors_with_labels(matrices, row_ids, exclude_neg = True):
    """
    Returns a list of tuples containing the vectors and labels of the matrices
    Used for converting the input data into data suitable for binary classification
    (for a given set of potential mother features, decide whether it's a bud or not)
    
    if exclude_neg = True, it skips the rows with -1
    """
    vectors = []
    labels = []
    for matrix, row_id in zip(matrices, row_ids):
        for i, row in enumerate(matrix):
            if(exclude_neg and not row[0] == -1):
                label = 1 if i == row_id else 0
                vectors.append(row)
                labels.append(label)
            elif(not exclude_neg):
                label = 1 if i == row_id else 0
                vectors.append(row)
                labels.append(label)  
    return np.array(vectors), np.array(labels)

def run_prediction_on_buds(matrices, row_ids, model, exclude_neg = True):
    """
    For a given budding data in matrices, calculates a score for each mother
    and finds the one with the highest score
    
    if exclude_neg = True, it skips the rows with -1 for prediction
    """
    y_pred = []
    for matrix, row_id in zip(matrices, row_ids):
        vectors = []
        for row in matrix:
            if(exclude_neg and not row[0] == -1):
                vectors.append(row)
            elif(not exclude_neg):
                vectors.append(row)        
        dtest = xgb.DMatrix(vectors)
        preds = model.predict(dtest)
        y_pred.append(np.argmax(preds))
    return y_pred

def grid_train_xgboost(X_train, y_train, X_test, y_test, max_depths = [2,3,4,5,10], min_child_weights = [0,1,2,5,10], early_stopping_rounds=10, num_boost_round = 100):
    #This funciton performs grid search through parameters that are influencing the probability of overfitting
    #It returns the values for AUC-ROC for the best model and the values used for training
    
    best_params = None
    best_acc = 0
    num_zeros, num_ones = count_classes(y_train)
    for max_d in max_depths:
        for min_child in min_child_weights:
            params = {'objective': 'binary:logistic', 'eval_metric': 'error', 'max_depth': max_d, 'min_child_weight': min_child, 'scale_pos_weight': (num_zeros / num_ones)}
            dtrain = xgb.DMatrix(X_train, label=y_train)
            dtest = xgb.DMatrix(X_test, label=y_test)
            model = xgb.train(params, dtrain, evals=[(dtest, 'test')], 
                          early_stopping_rounds=early_stopping_rounds, verbose_eval=False)
            preds = model.predict(dtest)
            preds = np.round(preds)
            preds = [int(pred) for pred in preds]
            acc = accuracy_score (y_test, preds)
            print('Accuracy with ', params, 'is ', "%.4f " % (acc))
            if acc > best_acc:
                best_acc = acc
                best_params = params

    print("Best ACC:", best_acc)
    print("Best params:", best_params)
    return best_params 
   
# Generate some example data to test the model
#N_data = 100000
#X, y = generate_data(N_data, 100, 0.1)

#Load data
folder = '../data/'
data_full = np.load(folder + 'out_full.npy')
data_small = np.load(folder + 'out_small.npy')
labels = np.load(folder + 'out_GT.npy')
num_classes = 4

#Changing axes order so that they match the expected input into the network
data_full = np.moveaxis(data_full, [0, 1, 2], [1, 2, 0])
data_small = np.moveaxis(data_small, [0, 1, 2], [1, 2, 0])

#Keep only certain features
#Be carefull that data is incomplete and an error is going to be generated if you leave only features
# that have no value for at least one bud
#For example, frequently picked top 4 features are 5,6,7 and 11 but there are buds for which we miss all of these values
data_small = keep_features(data_small, feature_columns = [0, 5, 6, 7, 11])

#We have to get rid of the inner brackets in the labels data
new_labels = []
for i in labels:
    new_labels.append(i[0])
new_labels = np.array(new_labels)
  
# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(data_small, new_labels, test_size=0.2, shuffle = True)

#Convert the data into vectors of potential mothers that can be binary-classified
X_train, y_train = get_vectors_with_labels(X_train, y_train, exclude_neg = True)
X_test, y_test = get_vectors_with_labels(X_test, y_test, exclude_neg = True)

# Train the XGBoost model with the default parameters
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)
#Here we set the weights so that they corerspond to the ratio of zeros and ones in the input data
#This is supposed to help when dealing with imbalance of classes - we here have more zeros (fake mothers) than ones (true mothers)

num_zeros, num_ones = count_classes(labels)
params = {'objective': 'binary:logistic', 'eval_metric': 'error', 'scale_pos_weight': (num_zeros / num_ones)} 
bst = xgb.train(params, dtrain, num_boost_round = 100)

# Make predictions and evaluate the model
preds = bst.predict(dtest)
preds = np.round(preds)
preds = [int(pred) for pred in preds]
print("Accuracy of the default model:", accuracy_score(y_test, preds))
#plot_confusion_matrix(y_test, preds)

#Searching for optimal tree depth and child weights 
best_params = grid_train_xgboost(X_train, y_train, X_test, y_test, early_stopping_rounds = 10, num_boost_round = 100)

#Retrain the model with best params
#Here, boosting for more than 20 rounds helps
evals_result = {}
bst = xgb.train(best_params, dtrain, num_boost_round = 200, early_stopping_rounds = 10, evals=[(dtest, 'test'), (dtrain, 'train')], verbose_eval = False, evals_result=evals_result)
plot_eval_metrics(evals_result)

# Make predictions and evaluate the model
preds = bst.predict(dtest)
preds = np.round(preds)
print("Accuracy of the best model:", "%.4f " % accuracy_score(y_test, preds))
plot_confusion_matrix(y_test, preds)

# Plot feature importance of the best model
plot_feature_importance(bst)

#Now we use the best tree to run predictions on the bud data
preds = run_prediction_on_buds(data_small, new_labels, bst, exclude_neg = True)
print("Final accuracy for bud assignment:", "%.4f " % accuracy_score(new_labels, preds))
bst.save_model('best_model_for_binary_data.json')
