# -*- coding: utf-8 -*-
"""
Created on Tue Jan 17 14:58:25 2023

@author: gligorov
"""

import matplotlib.pyplot as plt
import numpy as np

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