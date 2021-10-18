"""
Residual network model (stage_1_2)

@authors: Joaquin Rives, Mojtaba Jafaritadi, Ph.D.
@email: joaquin.rives01@gmail.com
"""

from sklearn.model_selection import StratifiedKFold
import os
from utils import encode_labels, custom_multiclass_f1, multiclass_f1
import shutil
import ntpath
from collections import Counter
import numpy as np
from keras.initializers import glorot_uniform, he_normal
from preprocess_and_segmentation import load_data, segment_all_dict_data, reshape_segmented_arrays
from preprocessor import preprocess_input_data
from utils import encode_labels
import tensorflow.keras as keras
import pandas as pd


def identity_block(X, f, filters, stage, block):
    """
    Implementation of the identity block for RESNET

    Arguments:
    X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
    f -- integer, specifying the shape of the middle CONV's window for the main path
    filters -- python list of integers, defining the number of filters in the CONV layers of the main path
    stage -- integer, used to name the layers, depending on their position in the network
    block -- string/character, used to name the layers, depending on their position in the network

    Returns:
    X -- output of the identity block, tensor of shape (m, n_W, n_C)
    """

    # defining name basis
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    # Retrieve Filters
    F1, F2, F3 = filters
    # n_timesteps, n_features = X.shape[1], X.shape[2]
    # Save the input value. You'll need this later to add back to the main path.
    X_shortcut = X
    n_timesteps, n_features = X.shape[1], X.shape[2]

    # First component of main path
    X = keras.layers.Conv1D(filters=F1, kernel_size=1, strides=1, padding='same',
                            input_shape=(None, n_timesteps, n_features),
                            name=conv_name_base + '2a')(X)
    X = keras.layers.BatchNormalization(name=bn_name_base + '2a')(X)
    X = keras.layers.Activation('relu')(X)
    X = keras.layers.Dropout(0.2)(X)
    # Second component of main path (≈3 lines)
    X = keras.layers.Conv1D(filters=F2, kernel_size=f, strides=1, padding='same', name=conv_name_base + '2b')(X)
    X = keras.layers.BatchNormalization(name=bn_name_base + '2b')(X)
    X = keras.layers.Activation('relu')(X)
    X = keras.layers.Dropout(0.2)(X)
    # Third component of main path (≈2 lines)
    X = keras.layers.Conv1D(filters=F3, kernel_size=1, strides=1, padding='same', name=conv_name_base + '2c')(X)
    X = keras.layers.BatchNormalization(name=bn_name_base + '2c')(X)

    # Final step: Add shortcut value to main path, and pass it through a RELU activation (≈2 lines)
    X = keras.layers.add([X, X_shortcut])
    X = keras.layers.Activation('relu')(X)

    ### END CODE HERE ###

    return X


def maxpool_block_1(X, f, filters, s, stage, block):
    """
    Implementation of the identity block for RESNET

    Arguments:
    X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
    f -- integer, specifying the shape of the middle CONV's window for the main path
    filters -- python list of integers, defining the number of filters in the CONV layers of the main path
    stage -- integer, used to name the layers, depending on their position in the network
    block -- string/character, used to name the layers, depending on their position in the network

    Returns:
    X -- output of the identity block, tensor of shape (m, n_W, n_C)
    """

    # defining name basis
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    mp_name_base = 'mp' + str(stage) + block + '_branch'

    # Retrieve Filters
    F1, F2 = filters
    # n_timesteps, n_features = X.shape[1], X.shape[2]
    # Save the input value. You'll need this later to add back to the main path.
    X_shortcut = X
    n_timesteps, n_features = X.shape[1], X.shape[2]

    # First component of main path
    X = keras.layers.Conv1D(filters=F1, kernel_size=f, strides=s, padding='same',
                            input_shape=(None, n_timesteps, n_features),
                            name=conv_name_base + '2a', kernel_initializer=he_normal(seed=0))(X)
    X = keras.layers.BatchNormalization(name=bn_name_base + '2a')(X)
    X = keras.layers.Activation('relu')(X)
    X = keras.layers.Dropout(0.2)(X)

    # Second component of main path ()
    X = keras.layers.Conv1D(filters=F2, kernel_size=1, strides=1, name=conv_name_base + '2b',
                            kernel_initializer=he_normal(seed=0))(
        X)

    X_shortcut = keras.layers.Conv1D(filters=F2, kernel_size=1, strides=1, name=conv_name_base + '1',
                                     kernel_initializer=he_normal(seed=0))(X_shortcut)

    X_shortcut = keras.layers.MaxPooling1D(pool_size=s, padding='same', name=mp_name_base + '1')(X_shortcut)

    # Final step: Add shortcut value to main path, and pass it through a RELU activation (≈2 lines)
    X = keras.layers.add([X, X_shortcut])
    X = keras.layers.Activation('relu')(X)

    ### END CODE HERE ###

    return X


def maxpool_block_2(X, f, filters, s, stage, block):
    """
    Implementation of the identity block for RESNET

    Arguments:
    X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
    f -- integer, specifying the shape of the middle CONV's window for the main path
    filters -- python list of integers, defining the number of filters in the CONV layers of the main path
    stage -- integer, used to name the layers, depending on their position in the network
    block -- string/character, used to name the layers, depending on their position in the network

    Returns:
    X -- output of the identity block, tensor of shape (m, n_W, n_C)
    """

    # defining name basis
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    mp_name_base = 'mp' + str(stage) + block + '_branch'
    # Retrieve Filters
    F1, F2 = filters
    # n_timesteps, n_features = X.shape[1], X.shape[2]
    # Save the input value. You'll need this later to add back to the main path.
    X_shortcut = X
    n_timesteps, n_features = X.shape[1], X.shape[2]

    # First component of main path
    X = keras.layers.BatchNormalization(name=bn_name_base + '2a')(X)
    X = keras.layers.Activation('relu')(X)
    # Second component of main path (≈3 lines)
    X = keras.layers.Conv1D(filters=F1, kernel_size=f, strides=s, padding='same',
                            input_shape=(None, n_timesteps, n_features),
                            name=conv_name_base + '2b', kernel_initializer=he_normal(seed=0))(X)
    X = keras.layers.BatchNormalization(name=bn_name_base + '2b')(X)
    X = keras.layers.Activation('relu')(X)
    X = keras.layers.Dropout(0.2)(X)
    # Third component of main path (≈2 lines)
    X = keras.layers.Conv1D(filters=F2, kernel_size=1, strides=1, padding='same', name=conv_name_base + '2c',
                            kernel_initializer=he_normal(seed=0))(X)

    # Final step: Add shortcut value to main path, and pass it through a RELU activation (≈2 lines)
    ##### SHORTCUT PATH #### (≈2 lines)
    X_shortcut = keras.layers.Conv1D(filters=F2, kernel_size=1, strides=1, name=conv_name_base + '1',
                                     kernel_initializer=he_normal(seed=0))(X_shortcut)

    X_shortcut = keras.layers.MaxPooling1D(pool_size=s, padding='same', name=mp_name_base + '1')(X_shortcut)
    # print(X_shortcut.shape)

    # Final step: Add shortcut value to main path, and pass it through a RELU activation (≈2 lines)
    X = keras.layers.add([X, X_shortcut])
    X = keras.layers.Activation('relu')(X)

    return X


def convolutional_block(X, f, filters, stage, block, s=2):
    """
    Implementation of the convolutional block as defined in Figure 4

    Arguments:
    X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
    f -- integer, specifying the shape of the middle CONV's window for the main path
    filters -- python list of integers, defining the number of filters in the CONV layers of the main path
    stage -- integer, used to name the layers, depending on their position in the network
    block -- string/character, used to name the layers, depending on their position in the network
    s -- Integer, specifying the stride to be used

    Returns:
    X -- output of the convolutional block, tensor of shape (m, n_W, n_C)
    """

    # defining name basis
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    # Retrieve Filters
    F1, F2, F3 = filters
    # n_timesteps, n_features = X.shape[1], X.shape[2]
    # Save the input value
    X_shortcut = X
    n_timesteps, n_features = X.shape[1], X.shape[2]

    ##### MAIN PATH #####
    # First component of main path
    X = keras.layers.Conv1D(filters=F1, kernel_size=3, strides=s, padding='same',
                            input_shape=(None, n_timesteps, n_features),
                            name=conv_name_base + '2a', kernel_initializer=he_normal(seed=0))(X)
    X = keras.layers.BatchNormalization(name=bn_name_base + '2a')(X)
    X = keras.layers.Activation('relu')(X)
    X = keras.layers.Dropout(0.2)(X)

    # Second component of main path (≈3 lines)
    X = keras.layers.Conv1D(filters=F2, kernel_size=24, strides=1, padding='same', name=conv_name_base + '2b',
                            kernel_initializer=he_normal(seed=0))(X)
    X = keras.layers.BatchNormalization(name=bn_name_base + '2b')(X)
    X = keras.layers.Activation('relu')(X)
    X = keras.layers.Dropout(0.2)(X)

    # Third component of main path (≈2 lines)
    X = keras.layers.Conv1D(filters=F3, kernel_size=1, strides=1, padding='same', name=conv_name_base + '2c',
                            kernel_initializer=he_normal(seed=0))(X)
    X = keras.layers.BatchNormalization(name=bn_name_base + '2c')(X)
    X = keras.layers.Activation('relu')(X)
    X = keras.layers.Dropout(0.2)(X)

    ##### SHORTCUT PATH #### (≈2 lines)
    X_shortcut = keras.layers.Conv1D(filters=F3, kernel_size=1, strides=s, padding='same', name=conv_name_base + '1',
                                     kernel_initializer=he_normal(seed=0))(X_shortcut)
    # print(X_shortcut.shape)
    X_shortcut = keras.layers.BatchNormalization(name=bn_name_base + '1')(X_shortcut)
    # X_shortcut = Dropout(0.25)(X_shortcut)

    # Final step: Add shortcut value to main path, and pass it through a RELU activation (≈2 lines)
    X = keras.layers.add([X, X_shortcut])
    X = keras.layers.Activation('relu')(X)

    return X


def ResNet1D(input_shape=(2000, 12), classes=9):
    """
    Implementation of the popular ResNet50 the following architecture:
    CONV2D -> BATCHNORM -> RELU -> MAXPOOL -> CONVBLOCK -> IDBLOCK*2 -> CONVBLOCK -> IDBLOCK*3
    -> CONVBLOCK -> IDBLOCK*5 -> CONVBLOCK -> IDBLOCK*2 -> AVGPOOL -> TOPLAYER

    Arguments:
    input_shape -- shape of the images of the dataset
    classes -- integer, number of classes

    Returns:
    model -- a Model() instance in Keras
    """

    # Define the input as a tensor with shape input_shape
    X_input = keras.layers.Input(input_shape)
    # Zero-Padding
    X = keras.layers.ZeroPadding1D(0)(X_input)

    # Stage 1
    X = keras.layers.Conv1D(filters=16, kernel_size=7, padding='same', name='conv1',
                            kernel_initializer=he_normal(seed=0))(X)
    X = keras.layers.BatchNormalization(name='bn_conv1')(X)
    X = keras.layers.Activation('relu')(X)

    # Stage 2

    X = maxpool_block_1(X, f=5, filters=[16, 32], s=1, stage=2, block='a')
    X = identity_block(X, 7, [16, 16, 32], stage=2, block='b')
    X = identity_block(X, 7, [16, 16, 32], stage=2, block='c')
    X = convolutional_block(X, f=2, filters=[16, 16, 32], s=1, stage=2, block='d')

    # Stage 3

    X = maxpool_block_2(X, 5, filters=[16, 32], s=2, stage=3, block='a')
    X = maxpool_block_2(X, 13, filters=[16, 32], s=1, stage=3, block='b')
    X = maxpool_block_2(X, 13, filters=[16, 32], s=2, stage=3, block='d')
    X = identity_block(X, 7, [16, 16, 32], stage=3, block='e')
    X = identity_block(X, 7, [16, 16, 32], stage=3, block='f')
    X = convolutional_block(X, f=2, filters=[16, 16, 32], s=1, stage=3, block='g')

    # Stage 4
    X = maxpool_block_2(X, 5, filters=[32, 64], s=1, stage=4, block='a')
    X = maxpool_block_2(X, 13, filters=[32, 64], s=2, stage=4, block='b')
    X = maxpool_block_2(X, 13, filters=[32, 64], s=2, stage=4, block='d')
    X = identity_block(X, 7, [32, 32, 64], stage=4, block='e')
    X = identity_block(X, 7, [32, 32, 64], stage=4, block='f')
    X = convolutional_block(X, f=2, filters=[32, 32, 64], s=1, stage=4, block='g')

    # Stage 5
    X = maxpool_block_2(X, 5, filters=[64, 128], s=1, stage=5, block='a')
    X = maxpool_block_2(X, 13, filters=[64, 128], s=2, stage=5, block='b')
    X = maxpool_block_2(X, 13, filters=[64, 128], s=2, stage=5, block='d')
    X = identity_block(X, 7, [64, 64, 128], stage=5, block='e')
    X = identity_block(X, 7, [64, 64, 128], stage=5, block='f')
    X = convolutional_block(X, f=2, filters=[64, 64, 128], s=1, stage=5, block='g')

    # Stage 6
    X = maxpool_block_2(X, 5, filters=[128, 256], s=1, stage=6, block='a')
    X = maxpool_block_2(X, 13, filters=[128, 256], s=2, stage=6, block='b')
    X = maxpool_block_2(X, 13, filters=[128, 256], s=2, stage=6, block='d')
    X = identity_block(X, 7, [128, 128, 256], stage=6, block='e')
    X = identity_block(X, 7, [128, 128, 256], stage=6, block='f')
    X = convolutional_block(X, f=2, filters=[128, 128, 256], s=1, stage=6, block='g')

    X = keras.layers.BatchNormalization(name='bn_final')(X)
    X = keras.layers.Activation('relu')(X)

    # X=LSTM(50, return_sequences=True,input_shape=(X.shape[1],1))(X)

    # X=LSTM(20)(X)
    # X = MaxPooling1D(pool_size=2, name='max_pool')(X)
    # X = TimeDistributed(Flatten())(X)
    X = keras.layers.Flatten()(X)
    # X = LSTM(100)(X)
    # X = CuDNNLSTM(1000)(X)

    #
    # X=Dense(10, activation='relu',activity_regularizer=l1(0.0001), kernel_regularizer=regularizers.l2(0.001))(X)
    X = keras.layers.Dense(300, activation='relu')(X)
    X = keras.layers.Dropout(0.3)(X)

    X = keras.layers.Dense(classes, activation='softmax', name='fc' + str(classes))(X)

    # Create model
    model = keras.models.Model(inputs=X_input, outputs=X, name='ResNet1D')

    return model