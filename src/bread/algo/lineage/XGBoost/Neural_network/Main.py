# -*- coding: utf-8 -*-
"""
Created on Tue Jan 17 12:06:44 2023

@author: gligorov
"""
import tensorflow as tf
import numpy as np
from Model import create_model, find_optimal_parameters
from Data_processing import generate_all_permutations, keep_features
from utils import plot_history, show_classification_results
from sklearn.model_selection import train_test_split

# Set parameters of training
batch_s = 32
num_classes = 4
n_epochs = 100
use_conv = False  #Convolutional layer is optional
learning_rate = 1e-2

# generate random data to test the code. If predictable = True, it will be possible to learn to predict the labels with >90% high accuracy
# This can be used for testing the model
#data, labels = generate_random_data(n_data_points, num_classes, num_features, predictable = True)
# Processs the data in a way that can be used for training
#x, y = data_generator(data, labels)

#Loading mother-daughter data
folder = '../data/'
data_full = np.load(folder + 'out_full.npy')
data_small = np.load(folder + 'out_small.npy')
labels = np.load(folder + 'out_GT.npy')

#Changing axes order so that they match the expected input into the network
#(n_data_points, n_potential_mothes, n_features)
data_full = np.moveaxis(data_full, [0, 1, 2], [1, 2, 0])
data_small = np.moveaxis(data_small, [0, 1, 2], [1, 2, 0])

#Keep only a subset of features
data_small = keep_features(data_small, feature_columns = [0, 1, 2, 3, 4, 5])
num_features = data_small.shape[2]

#Get rid of the inner brackets in the labels data
new_labels = []
for i in labels:
    new_labels.append(i[0])
new_labels = np.array(new_labels)
  
#One-hot encoding the labels
labels = np.array([np.eye(num_classes)[i] for i in new_labels])

#Augmenting data by generating all possible permutations of rows of each matrix
#The factor of augmentaions is going to be num_rows! (in our case, 4! = 24)
#This boosts the prediction accuracy by few percents
x, y = generate_all_permutations(data_small, labels)

#Split into test set and train set (later we split train set ono validation and train)
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle = True)

# Create a model
model = create_model((num_classes, num_features), num_classes, dense_layers = [128, 64, 32, 16], conv_layers = use_conv)
model.summary()

print('Training model on ' + str(len(x)) + ' data points')

# Define the learning rate scheduler
lr_schedule = tf.keras.callbacks.LearningRateScheduler(
    lambda epoch: learning_rate / 10**(epoch /100))

#Checkpoint for saving the weights
checkpoint = tf.keras.callbacks.ModelCheckpoint(
    filepath='weights.{epoch:02d}-{val_loss:.2f}.hdf5',
    monitor='val_loss',
    save_best_only=True,
    save_weights_only=True,
    mode='min',
    verbose=1)

if(use_conv):
    X_train = np.expand_dims(X_train, 3)
    print('Adjusted the shape of x to ' + str(X_train.shape) + 'to be compatible with expected input for a convolutional network')

# Train the model
history  = model.fit(X_train, y_train, epochs=n_epochs, batch_size = batch_s, validation_split = 0.2, shuffle = True, callbacks=[lr_schedule])#, checkpoint])
plot_history(history, 'acc')

ypred = model.predict(x)
#N = 3
# Next function plots N randomly chosen accurately classified and N randomly chosen wrongly classified data instances
#show_classification_results(x, y, ypred, 3)

#Evaluate on the test set
if(use_conv): 
    X_test = np.expand_dims(X_test, 3)
    
# Test the model
test_loss, test_acc = model.evaluate(X_test, y_test)
print('Test accuracy:', "%.4f " % test_acc)

#Do a grid scan over batch sizes and learning rates to find the optimal ones
#batch_sizes = [10, 100]
#learning_rates = [1e-2, 1e-3]
## Do a scan through batch sizes and learning rates to find the best combination
#(highest_acc, best_param) = find_optimal_parameters(batch_sizes, learning_rates, X_train, y_train, n_epochs, num_classes, num_features, use_conv = False, verbose = False)
#print('Highest classification accuracy on a validation set is ' + f'{highest_acc:.3f}' + ' and is achieved using batch size ' + str(best_param[0]) + ' and learning rate ' + str(best_param[1]) + '.')