# -*- coding: utf-8 -*-
"""
Created on Wed Jan 18 14:12:39 2023

@author: gligorov
"""
import xgboost as xgb
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from Data_processing import keep_features, generate_all_permutations, flatten_3d_array
from utils import plot_feature_importance, plot_eval_metrics



def grid_train_xgboost(X_train, y_train, X_test, y_test, max_depths = [2,3,4,5,10], min_child_weights = [0,1,2,5,10], early_stopping_rounds=10, num_boost_round = 100):
    #This funciton performs grid search through parameters that are influencing the probability of overfitting
    #It returns the values for AUC-ROC for the best model and the values used for training
    
    best_params = None
    best_acc = 0
    for max_d in max_depths:
        for min_child in min_child_weights:
            params = {'objective': 'multi:softmax', 'num_class': 4, 'max_depth': max_d, 'min_child_weight': min_child, 'eval_metric': 'merror'}
            dtrain = xgb.DMatrix(X_train, label=y_train)
            dtest = xgb.DMatrix(X_test, label=y_test)
            model = xgb.train(params, dtrain, evals=[(dtest, 'test')],
                          early_stopping_rounds=early_stopping_rounds, verbose_eval = False, num_boost_round = num_boost_round)
            preds = model.predict(dtest)
            acc = accuracy_score (y_test, preds)
            print('Accuracy with ', params, 'is ', "%.4f " % (acc))
            if acc > best_acc:
                best_acc = acc
                best_params = params

    print("Best ACC:", best_acc)
    print("Best params:", best_params)
    return best_params 

if __name__ == '__main__':

    #Load data
    folder = '../data/'
    data_full = np.load(folder + 'out_full.npy')
    data_small = np.load(folder + 'out_small.npy')
    labels = np.load(folder + 'out_GT.npy')

    num_classes = 4

    #Changing axes order so that they match the expected input into the network
    data_full = np.moveaxis(data_full, [0, 1, 2], [1, 2, 0])
    data_small = np.moveaxis(data_small, [0, 1, 2], [1, 2, 0])

    #Keep only some featyres
    data_small = keep_features(data_small, feature_columns = [0, 6, 7, 11])
    X = data_small #X is the set that is going to be used

    #We have to get rid of the inner brackets in the labels data
    new_labels = []
    for i in labels:
        new_labels.append(i[0])
    new_labels = np.array(new_labels)
    y = new_labels #y is the set of labels that is going to be used

    augment = True
    if(augment):
        X, y = generate_all_permutations(data_small, labels)
        #returning labels into integers

    #Flattening the data into 1D  - required by XGBoost
    X = flatten_3d_array(X)

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle = True)

    # Train the default XGBoost model
    dtrain = xgb.DMatrix(X_train, label=y_train)
    params = {'objective': 'multi:softmax', 'num_class': 4, 'eval_metric': 'merror'} 
    bst = xgb.train(params, dtrain, num_boost_round = 100)

    # Make predictions and evaluate the model
    dtest = xgb.DMatrix(X_test, label=y_test)
    preds = bst.predict(dtest)
    preds = np.round(preds)
    print("Accuracy of the default model:", "%.4f " % accuracy_score(y_test, preds))

    #Searching for optimal tree depth to prevent overfitting
    #early stopping rounds specidies the number of epochs after which the training will stop if there is no improvement in accuracy
    best_params = grid_train_xgboost(X_train, y_train, X_test, y_test, early_stopping_rounds = 10, num_boost_round = 100)

    #Make predictions with best_params
    evals_result = {}
    bst = xgb.train(best_params, dtrain, num_boost_round = 200, early_stopping_rounds= 10, evals=[(dtest, 'test'), (dtrain, 'train')], verbose_eval = False, evals_result=evals_result)

    # Make predictions and evaluate the best model
    preds = bst.predict(dtest)
    preds = np.round(preds)
    print("Accuracy of the best model:", "%.4f " % accuracy_score(y_test, preds))
    plot_eval_metrics(evals_result)

    #plot_confusion_matrix(y_test, preds) #doesn't make much sense here to look at the confusion matrix
    plot_feature_importance(bst) 
    bst.save_model('best_model_for_matrix_data.json')


