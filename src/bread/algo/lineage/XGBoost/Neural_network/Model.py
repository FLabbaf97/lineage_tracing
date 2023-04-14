# -*- coding: utf-8 -*-
"""
Created on Tue Jan 17 14:56:29 2023

@author: gligorov
"""
import tensorflow as tf
import numpy as np
from utils import plot_history

def create_model(input_shape, num_classes, dense_layers = [128, 64, 32, 16], conv_layers = True, conv_filters=32, conv_kernel_size= 3, activation='relu', last_activation = 'softmax'):
    
    """
    Create a NN model with specified parameters
    
    input_shape: The shape of the input data, in the form of a tuple
    num_classes: The number of output classes
    dense_layers: A list of integers representing the number of units in each dense layer. By default, the function creates four dense layers with 128, 64, 32 and 16 units respectively.
    conv_layers: A Boolean value indicating whether or not to include convolutional layers in the model. If set to True, the function creates two convolutional layers with by default 32 filters and a kernel size of 3, and a max pooling layer after each convolutional layer.
    conv_filters: Number of filters in the convolutional layers.
    conv_kernel_size: Size of the kernel in the convolutional layers.
    activation: The activation function to use in the dense layers. By default, the function uses the ReLU activation function.
    last_activation: The last activation function to use in the output layer. By default, the function uses the softmax activation function.
    """

    # Define optional convolutional layers
    if conv_layers:
        input_shape = (input_shape[0], input_shape[1], 1) #we have to expand input shape for since convolution data with one additional dimension for channels
        input_layer = tf.keras.layers.Input(shape=input_shape)
        x = tf.keras.layers.Conv2D(conv_filters, conv_kernel_size, activation='relu', padding='same')(input_layer)
        x = tf.keras.layers.MaxPooling2D()(x)
        x = tf.keras.layers.Conv2D(conv_filters, conv_kernel_size, activation='relu', padding='same')(x)
        x = tf.keras.layers.MaxPooling2D()(x)
    else:
        input_layer = tf.keras.layers.Input(shape=input_shape)
        x = input_layer
    
    x = tf.keras.layers.Flatten()(x)
    
    for layer in dense_layers:
        x = tf.keras.layers.Dense(layer, activation=activation)(x)

    # Define output layer
    output_layer = tf.keras.layers.Dense(num_classes, activation='softmax')(x)

    # Create model
    model = tf.keras.models.Model(inputs=input_layer, outputs=output_layer)

    # Compile model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model

def find_optimal_parameters(batch_sizes, learning_rates, x, y, n_epochs, num_classes, num_features, use_conv = False, verbose = False):
    
    """
    Searches throuh all batch_sizes and all learning_rates from the input list
    To find the ones that give highest accuracy of prediction
    """
    
    #Initialization of highest accuracy prediction
    highest_acc = 0
    best_param = None
    
    for b in batch_sizes:
        for lr in learning_rates:
            
            # Define the learning rate scheduler
            lr_schedule = tf.keras.callbacks.LearningRateScheduler(
                    lambda epoch: lr / 10**(epoch /100))
            
            #Train a new model
            print("Training new model with Learning rate " + str(lr) + ", batch size is " + str(b))
            model = create_model((num_classes, num_features), num_classes, dense_layers = [128, 64, 32, 16], conv_layers = use_conv)
            if(use_conv):
                x = np.expand_dims(x, 3)
            history = model.fit(x, y, epochs = n_epochs, batch_size = b, validation_split = 0.2, shuffle = True, callbacks = [lr_schedule], verbose = verbose)
            plot_history(history, 'acc')
            
            #Remember if its the best one so far
            if(np.max(history.history['val_accuracy']) > highest_acc):
                highest_acc = np.max(history.history['val_accuracy'])
                best_param = [b, lr]
                
    return (highest_acc, best_param)
