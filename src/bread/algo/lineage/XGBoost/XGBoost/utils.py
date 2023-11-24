# -*- coding: utf-8 -*-
"""
Created on Tue Jan 17 14:58:25 2023

@author: gligorov
"""

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

def plot_feature_importance(bst, figsize=(10,5)):
    """
    Plots the feature importances of an XGBoost model
    """
    plt.figure(figsize=figsize)
    feature_importance = bst.get_score(importance_type='weight')
    feature_importance = {k: v for k, v in sorted(feature_importance.items(), key=lambda item: item[1], reverse=True)}
    plt.bar(range(len(feature_importance)), list(feature_importance.values()), align='center')
    plt.xticks(range(len(feature_importance)), list(feature_importance.keys()), rotation=90)
    plt.xlabel('Features')
    plt.ylabel('Feature importance score')
    plt.title('Feature importance')
    plt.show()

def plot_eval_metrics(evals_result):
    """
    Plot evaluation metrics from an XGBoost model
    """
    plt.figure(figsize=(10,5))
    plt.subplot(1, 2, 1)
    
    try:
        plt.plot(evals_result['train']['merror'], label='Train')
        plt.plot(evals_result['test']['merror'], label='Validation')
        
    except: #For binary classification, the loss is 'error'
        plt.plot(evals_result['train']['error'], label='Train')
        plt.plot(evals_result['test']['error'], label='Validation')
        
    plt.xlabel('Number of Boosting Rounds')
    plt.ylabel('Error')
    plt.title('Error vs Number of Boosting Rounds')
    plt.legend()
    
def plot_confusion_matrix(y_true, y_pred):
    plt.figure()
    plt.imshow(confusion_matrix(y_true, y_pred), cmap='Blues')
    plt.title('Confusion matrix')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
        
def print_matrix(s):
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
        