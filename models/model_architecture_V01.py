# -*- coding: utf-8 -*-
"""ResNet_module.ipynb

Created on Wed Nov 13 16:11:12 2019

@author: Mojtaba Jafaritadi, Ph.D.
"""

import tensorflow.keras as keras


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
    #n_timesteps, n_features = X.shape[1], X.shape[2]
    # Save the input value. You'll need this later to add back to the main path. 
    X_shortcut = X
    n_timesteps, n_features = X.shape[1], X.shape[2]

    # First component of main path
    X = keras.layers.Conv1D(filters=F1, kernel_size=1,strides =1,padding='same',input_shape=(None,n_timesteps,n_features), name = conv_name_base + '2a')(X)
    X = keras.layers.BatchNormalization(name = bn_name_base + '2a')(X)
    X = keras.layers.Activation('relu')(X)
    X = keras.layers.Dropout(0.25)(X)
    # Second component of main path (≈3 lines)
    X = keras.layers.Conv1D(filters=F2, kernel_size=f,strides =1,padding='same', name = conv_name_base + '2b')(X)
    X = keras.layers.BatchNormalization(name = bn_name_base + '2b')(X)
    X = keras.layers.Activation('relu')(X)
    X = keras.layers.Dropout(0.25)(X)
    # Third component of main path (≈2 lines)
    X = keras.layers.Conv1D(filters=F3, kernel_size=1, strides =1,padding='same', name = conv_name_base + '2c')(X)
    X = keras.layers.BatchNormalization(name = bn_name_base + '2c')(X)

    # Final step: Add shortcut value to main path, and pass it through a RELU activation (≈2 lines)
    X = keras.layers.Add()([X, X_shortcut])
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
    #n_timesteps, n_features = X.shape[1], X.shape[2]
    # Save the input value. You'll need this later to add back to the main path. 
    X_shortcut = X
    n_timesteps, n_features = X.shape[1], X.shape[2]

    # First component of main path
    X = keras.layers.Conv1D(filters=F1, kernel_size=f,strides =s,padding='same',input_shape=(None,n_timesteps,n_features), name = conv_name_base + '2a', kernel_initializer = keras.initializers.he_normal(seed=0))(X)
    X = keras.layers.BatchNormalization(name = bn_name_base + '2a')(X)
    X = keras.layers.Activation('relu')(X)
    X = keras.layers.Dropout(0.15)(X)
    # Second component of main path ()
    X = keras.layers.Conv1D(filters=F2, kernel_size=1, strides=1, name = conv_name_base + '2b', kernel_initializer = keras.initializers.he_normal(seed=0))(X)
    
    X_shortcut = keras.layers.Conv1D(filters=F2, kernel_size=1, strides=1, name = conv_name_base + '1', kernel_initializer = keras.initializers.he_normal(seed=0))(X_shortcut)

    X_shortcut = keras.layers.MaxPooling1D(pool_size=s, padding='same',name = mp_name_base + '1')(X_shortcut)

    

    # Final step: Add shortcut value to main path, and pass it through a RELU activation (≈2 lines)
    X = keras.layers.Add()([X, X_shortcut])
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
    #n_timesteps, n_features = X.shape[1], X.shape[2]
    # Save the input value. You'll need this later to add back to the main path. 
    X_shortcut = X
    n_timesteps, n_features = X.shape[1], X.shape[2]

    # First component of main path
    X = keras.layers.BatchNormalization(name = bn_name_base + '2a')(X)
    X = keras.layers.Activation('relu')(X)
#    X = LeakyReLU(alpha=0.3)(X)    
    # Second component of main path (≈3 lines)
    X = keras.layers.Conv1D(filters=F1, kernel_size=f,strides =s,padding='same',input_shape=(None,n_timesteps,n_features), name = conv_name_base + '2b', kernel_initializer = keras.initializers.he_normal(seed=0))(X)
    X = keras.layers.BatchNormalization(name = bn_name_base + '2b')(X)
    X = keras.layers.Activation('relu')(X)
    #X = LeakyReLU(alpha=0.3)(X)
    X = keras.layers.Dropout(0.15)(X)
    # Third component of main path (≈2 lines)
    X = keras.layers.Conv1D(filters=F2, kernel_size=1, strides =1,padding='same', name = conv_name_base + '2c', kernel_initializer = keras.initializers.he_normal(seed=0))(X)
   

    # Final step: Add shortcut value to main path, and pass it through a RELU activation (≈2 lines)
    ##### SHORTCUT PATH #### (≈2 lines)
    X_shortcut = keras.layers.Conv1D(filters=F2, kernel_size=1, strides=1, name = conv_name_base + '1', kernel_initializer = keras.initializers.he_normal(seed=0))(X_shortcut)

    X_shortcut = keras.layers.MaxPooling1D(pool_size=s, padding='same',name = mp_name_base + '1')(X_shortcut)

   # print(X_shortcut.shape)

    # Final step: Add shortcut value to main path, and pass it through a RELU activation (≈2 lines)
    X = keras.layers.Add()([X, X_shortcut,])
    X = keras.layers.Activation('relu')(X)
    #X = LeakyReLU(alpha=0.3)(X)
    
    return X

def conv_inception_naive_resblock(X, f, filters, s, stage, block):
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
    F1, F2 , F3 = filters
    #n_timesteps, n_features = X.shape[1], X.shape[2]
    # Save the input value. You'll need this later to add back to the main path. 
    X_shortcut_a = X
    X_shortcut = X

    n_timesteps, n_features = X.shape[1], X.shape[2]

    # First component of main path
    X = keras.layers.Conv1D(filters=F1, kernel_size=1,strides =1,padding='same',input_shape=(None,n_timesteps,n_features), name = conv_name_base + '2a',kernel_initializer = keras.initializers.he_normal(seed=0))(X)
    X = keras.layers.BatchNormalization(name = bn_name_base + '2a')(X)
    X = keras.layers.Activation('relu')(X)
    X = keras.layers.Dropout(0.25)(X)
    X_shortcut_n=X 
    #X = LeakyReLU(alpha=0.3)(X)    
    # Second component of main path (≈3 lines)
    X = keras.layers.Conv1D(filters=F2, kernel_size=f,strides =s,padding='same', name = conv_name_base + '2b', kernel_initializer = keras.initializers.he_normal(seed=0))(X)
    X = keras.layers.BatchNormalization(name = bn_name_base + '2b')(X)
    X = keras.layers.Activation('relu')(X)
    X = keras.layers.Dropout(0.15)(X)

    #X_shortcut_c = X

    #X = LeakyReLU(alpha=0.3)(X)
    # Third component of main path (≈2 lines)
    X = keras.layers.Conv1D(filters=F3, kernel_size=1, strides =1,padding='same', name = conv_name_base + '2c', kernel_initializer = keras.initializers.he_normal(seed=0))(X)
    X = keras.layers.BatchNormalization(name = bn_name_base + '2c')(X)


    # Final step: Add shortcut value to main path, and pass it through a RELU activation (≈2 lines)
    ##### SHORTCUT PATH A #### (≈2 lines)
    X_shortcut_a = keras.layers.Conv1D(filters=F2, kernel_size=f, strides=1, padding='same', name = conv_name_base + '1', kernel_initializer = keras.initializers.he_normal(seed=0))(X_shortcut)
    X_shortcut_a = keras.layers.BatchNormalization(name = bn_name_base + '1')(X_shortcut_a)
    X_shortcut_a = keras.layers.Activation('relu')(X_shortcut_a)

    ##### SHORTCUT PATH B #### (≈2 lines)
    
    X_shortcut_b = keras.layers.Conv1D(filters=F2, kernel_size=f, strides=1,padding='same', name = conv_name_base + '2sha', kernel_initializer = keras.initializers.he_normal(seed=0))(X_shortcut_a)
    X_shortcut_b = keras.layers.BatchNormalization(name = bn_name_base + '2sha')(X_shortcut_b)
    X_shortcut_b = keras.layers.Activation('relu')(X_shortcut_b)
    X_shortcut_b= keras.layers.Dropout(0.15)(X_shortcut_b)
    X_shortcut_b = keras.layers.Conv1D(filters=F3, kernel_size=f, strides=1,padding='same', name = conv_name_base + '2shb', kernel_initializer = keras.initializers.he_normal(seed=0))(X_shortcut_b)
    X_shortcut_b = keras.layers.BatchNormalization(name = bn_name_base + '2shb')(X_shortcut_b)
    X_shortcut_b = keras.layers.Activation('relu')(X_shortcut_b)

    ##### SHORTCUT branch PATH M #### (≈2 lines)
    shortcut_m= keras.layers.Add()([X_shortcut_a, X_shortcut])

    shortcut_m = keras.layers.Conv1D(filters=F3, kernel_size=f, strides=1,padding='same', name = conv_name_base + '3sha', kernel_initializer = keras.initializers.he_normal(seed=0))(shortcut_m)
    shortcut_m = keras.layers.BatchNormalization(name = bn_name_base + '3sha')(shortcut_m)
    shortcut_m = keras.layers.Activation('relu')(shortcut_m)

    ##### SHORTCUT PATH D #### (≈2 lines)
    X_shortcut_d = keras.layers.Conv1D(filters=F2, kernel_size=f, strides=1,padding='same', name = conv_name_base + '4sha', kernel_initializer = keras.initializers.he_normal(seed=0))(X_shortcut_n)
    X_shortcut_d = keras.layers.BatchNormalization(name = bn_name_base + '4sha')(X_shortcut_d)
    X_shortcut_d = keras.layers.Activation('relu')(X_shortcut_d)
    X_shortcut_d= keras.layers.Dropout(0.15)(X_shortcut_d)
    X_shortcut_d = keras.layers.Conv1D(filters=F3, kernel_size=f, strides=1,padding='same', name = conv_name_base + '4shb', kernel_initializer = keras.initializers.he_normal(seed=0))(X_shortcut_d)
    X_shortcut_d = keras.layers.BatchNormalization(name = bn_name_base + '4shb')(X_shortcut_d)
    X_shortcut_d = keras.layers.Activation('relu')(X_shortcut_d)

    ##### SHORTCUT PATH E #### (≈2 lines)
    X_shortcut_e = keras.layers.Conv1D(filters=F3, kernel_size=f, strides=1,padding='same', name = conv_name_base + '5sha', kernel_initializer = keras.initializers.he_normal(seed=0))(X_shortcut)
    X_shortcut_e = keras.layers.BatchNormalization(name = bn_name_base + '5sha')(X_shortcut_d)
    X_shortcut_e = keras.layers.Activation('relu')(X_shortcut_e)
   # print(X_shortcut.shape)

    # Final step: Add shortcut value to main path, and pass it through a RELU activation (≈2 lines)
    X = keras.layers.Add()([X, shortcut_m, X_shortcut_b, X_shortcut_d, X_shortcut_e])
    X = keras.layers.Activation('relu')(X)
    
    #X = LeakyReLU(alpha=0.3)(X)
    
    return X

def convolutional_block(X, f, filters, stage, block,s=2):
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
    #n_timesteps, n_features = X.shape[1], X.shape[2]
    # Save the input value
    X_shortcut = X
    n_timesteps, n_features = X.shape[1], X.shape[2]


    ##### MAIN PATH #####
    # First component of main path 
    X = keras.layers.Conv1D(filters=F1, kernel_size=f,strides = s, padding='same', input_shape=(None,n_timesteps,n_features),name = conv_name_base + '2a', kernel_initializer = keras.initializers.he_normal(seed=0))(X)
    X = keras.layers.BatchNormalization(name = bn_name_base + '2a')(X)
    X = keras.layers.Activation('relu')(X)
    X = keras.layers.Dropout(0.3)(X)

    # Second component of main path (≈3 lines)
    X = keras.layers.Conv1D(filters=F2, kernel_size=f,strides = 1, padding='same', name = conv_name_base + '2b', kernel_initializer = keras.initializers.he_normal(seed=0))(X)
    #X = BatchNormalization(name = bn_name_base + '2b')(X)
    X = keras.layers.Activation('relu')(X)
    X = keras.layers.Dropout(0.3)(X)
  

    # Third component of main path (≈2 lines)
    X = keras.layers.Conv1D(filters=F3, kernel_size=1,strides = 1,padding='same', name = conv_name_base + '2c', kernel_initializer = keras.initializers.he_normal(seed=0))(X)
    X = keras.layers.BatchNormalization(name = bn_name_base + '2c')(X)
    X = keras.layers.Activation('relu')(X)

    X = keras.layers.Dropout(0.3)(X)


    ##### SHORTCUT PATH #### (≈2 lines)
    X_shortcut = keras.layers.Conv1D(filters=F3, kernel_size=1,strides = s,padding='same', name = conv_name_base + '1', kernel_initializer = keras.initializers.he_normal(seed=0))(X_shortcut)
   # print(X_shortcut.shape)
    X_shortcut = keras.layers.BatchNormalization(name = bn_name_base + '1')(X_shortcut)
    #X_shortcut = Dropout(0.25)(X_shortcut)

    # Final step: Add shortcut value to main path, and pass it through a RELU activation (≈2 lines)
    X = keras.layers.Add()([X, X_shortcut])
    X = keras.layers.Activation('relu')(X)
        
    return X

def DenseResNet1D(n_timesteps, n_features, classes = 9):
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
    #input1 = Input(input_shape)
    #input2 = Input(input_shape)
    #X_input = Concatenate()([input1, input2])
    X_input = keras.layers.Input(shape=(n_timesteps, n_features), dtype='float32')
    #X_input = Input(input_shape)
    # Zero-Padding
    X = keras.layers.ZeroPadding1D(0)(X_input)
    
    # Stage 0
    X = keras.layers.Conv1D(filters=4, kernel_size=3,padding='same', name = 'conv1', kernel_initializer = keras.initializers.he_normal(seed=0))(X)
    #X = BatchNormalization(name = 'bn_conv1')(X)
    X = keras.layers.Dropout(0.15)(X)
    X = keras.layers.LeakyReLU(alpha=0.3)(X)

   # X = Activation('relu')(X)
    
    # Stage 1
    stage=1
    block_conv='conv_dense_a'
    block_mp='mp_dense_a'
    X = maxpool_block_1(X, f = 3, filters = [4, 8], s=1,stage = 1, block='a')

    X_shortcut_1=X

    # Stage 2 
    stage=2
    block_conv='conv_dense_b'
    block_mp='mp_dense_b'
    X = maxpool_block_2(X, 3, filters = [8, 16], s=2, stage=2, block='a')
    X = maxpool_block_2(X, 3, filters = [8, 16], s=1, stage=2, block='b')
    X = maxpool_block_2(X, 3, filters = [8, 16], s=2, stage=2, block='c')

    X_shortcut_1 = keras.layers.Conv1D(filters=16, kernel_size=1, strides=1, name = block_conv + str(stage)  + '1', kernel_initializer = keras.initializers.he_normal(seed=0))(X_shortcut_1)
    X_shortcut_1 = keras.layers.MaxPooling1D(pool_size=4, padding='same',name = block_mp+str(stage) + '1')(X_shortcut_1)

    X = keras.layers.Add()([X, X_shortcut_1])
    X = keras.layers.Dropout(0.15)(X)


    X_shortcut_2=X

    # Stage 3
    stage=3
    block_conv='conv_dense_c'
    block_mp='mp_dense_c'
    X = maxpool_block_2(X, 3, filters = [16, 32],s=1, stage=3, block='a')
    X = maxpool_block_2(X, 3, filters = [16, 32],s=2, stage=3, block='b')
    X = maxpool_block_2(X, 3, filters = [16, 32],s=1, stage=3, block='c')

    X_shortcut_1 = keras.layers.Conv1D(filters=32, kernel_size=1, strides=1, name = block_conv + str(stage)  + '1', kernel_initializer = keras.initializers.he_normal(seed=0))(X_shortcut_1)
    X_shortcut_1 = keras.layers.MaxPooling1D(pool_size=2, padding='same',name = block_mp+str(stage) + '1')(X_shortcut_1)
    X_shortcut_2 = keras.layers.Conv1D(filters=32, kernel_size=1, strides=1, name = block_conv + str(stage)  + '2', kernel_initializer = keras.initializers.he_normal(seed=0))(X_shortcut_2)
    X_shortcut_2 = keras.layers.MaxPooling1D(pool_size=2, padding='same',name = block_mp+str(stage) + '2')(X_shortcut_2)

    X = keras.layers.Add()([X, X_shortcut_1,X_shortcut_2])
    X = keras.layers.Dropout(0.15)(X)

    X_shortcut_3=X

    # Stage 4
    stage=4
    block_conv='conv_dense_d'
    block_mp='mp_dense_d'    
    X = maxpool_block_2(X, 3, filters = [32, 64],s=2, stage=4, block='a')
    X = maxpool_block_2(X, 3, filters = [32, 64],s=1, stage=4, block='b')
    X = maxpool_block_2(X, 3, filters = [32, 64],s=2, stage=4, block='c')

    X_shortcut_1 = keras.layers.Conv1D(filters=64, kernel_size=1, strides=1, name = block_conv + str(stage)  + '1', kernel_initializer = keras.initializers.he_normal(seed=0))(X_shortcut_1)
    X_shortcut_1 = keras.layers.MaxPooling1D(pool_size=4, padding='same',name = block_mp+str(stage) + '1')(X_shortcut_1)

    X_shortcut_2 = keras.layers.Conv1D(filters=64, kernel_size=1, strides=1, name = block_conv + str(stage)  + '2', kernel_initializer = keras.initializers.he_normal(seed=0))(X_shortcut_2)
    X_shortcut_2 = keras.layers.MaxPooling1D(pool_size=4, padding='same',name = block_mp+str(stage) + '2')(X_shortcut_2)

    X_shortcut_3 = keras.layers.Conv1D(filters=64, kernel_size=1, strides=1, name = block_conv + str(stage)  + '3', kernel_initializer = keras.initializers.he_normal(seed=0))(X_shortcut_3)
    X_shortcut_3 = keras.layers.MaxPooling1D(pool_size=4, padding='same',name = block_mp+str(stage) + '3')(X_shortcut_3)

    X = keras.layers.Add()([X, X_shortcut_1,X_shortcut_2,X_shortcut_3])
    X = keras.layers.Dropout(0.15)(X)
    X_shortcut_4=X

    # Stage 5
    stage=5
    block_conv='conv_dense_e'
    block_mp='mp_dense_e'
    X = maxpool_block_2(X, 3, filters = [64, 128],s=1, stage=5, block='a')
    X = maxpool_block_2(X, 3, filters = [64, 128],s=2, stage=5, block='b')
    X = maxpool_block_2(X, 3, filters = [64, 128],s=1, stage=5, block='c')

    X_shortcut_1 = keras.layers.Conv1D(filters=128, kernel_size=1, strides=1, name = block_conv + str(stage)  + '1', kernel_initializer = keras.initializers.he_normal(seed=0))(X_shortcut_1)
    X_shortcut_1 = keras.layers.MaxPooling1D(pool_size=2, padding='same',name = block_mp+str(stage) + '1')(X_shortcut_1)

    X_shortcut_2 = keras.layers.Conv1D(filters=128, kernel_size=1, strides=1, name = block_conv + str(stage)  + '2', kernel_initializer = keras.initializers.he_normal(seed=0))(X_shortcut_2)
    X_shortcut_2 = keras.layers.MaxPooling1D(pool_size=2, padding='same',name = block_mp+str(stage) + '2')(X_shortcut_2)

    X_shortcut_3 = keras.layers.Conv1D(filters=128, kernel_size=1, strides=1, name = block_conv + str(stage)  + '3', kernel_initializer = keras.initializers.he_normal(seed=0))(X_shortcut_3)
    X_shortcut_3 = keras.layers.MaxPooling1D(pool_size=2, padding='same',name = block_mp+str(stage) + '3')(X_shortcut_3)

    X_shortcut_4 = keras.layers.Conv1D(filters=128, kernel_size=1, strides=1, name = block_conv + str(stage)  + '4', kernel_initializer = keras.initializers.he_normal(seed=0))(X_shortcut_4)
    X_shortcut_4 = keras.layers.MaxPooling1D(pool_size=2, padding='same',name = block_mp+str(stage) + '4')(X_shortcut_4)

    X = keras.layers.Add()([X, X_shortcut_1,X_shortcut_2,X_shortcut_3,X_shortcut_4])
    X = keras.layers.vDropout(0.15)(X)
    X_shortcut_5=X

    # Stage 6
    stage=6
    block_conv='conv_dense_f'
    block_mp='mp_dense_f'
    X = maxpool_block_2(X, 3, filters = [128, 256],s=2, stage=6, block='a')
    X = maxpool_block_2(X, 3, filters = [128, 256],s=1, stage=6, block='b')
    X = maxpool_block_2(X, 3, filters = [128, 256],s=2, stage=6, block='c')

    X_shortcut_1 = keras.layers.Conv1D(filters=256, kernel_size=1, strides=1, name = block_conv + str(stage)  + '1', kernel_initializer = keras.initializers.he_normal(seed=0))(X_shortcut_1)
    X_shortcut_1 = keras.layers.MaxPooling1D(pool_size=4, padding='same',name = block_mp+str(stage) + '1')(X_shortcut_1)

    X_shortcut_2 = keras.layers.Conv1D(filters=256, kernel_size=1, strides=1, name = block_conv + str(stage)  + '2', kernel_initializer = keras.initializers.he_normal(seed=0))(X_shortcut_2)
    X_shortcut_2 = keras.layers.MaxPooling1D(pool_size=4, padding='same',name = block_mp+str(stage) + '2')(X_shortcut_2)

    X_shortcut_3 = keras.layers.Conv1D(filters=256, kernel_size=1, strides=1, name = block_conv + str(stage)  + '3', kernel_initializer = keras.initializers.he_normal(seed=0))(X_shortcut_3)
    X_shortcut_3 = keras.layers.MaxPooling1D(pool_size=4, padding='same',name = block_mp+str(stage) + '3')(X_shortcut_3)

    X_shortcut_4 = keras.layers.Conv1D(filters=256, kernel_size=1, strides=1, name = block_conv + str(stage)  + '4', kernel_initializer = keras.initializers.he_normal(seed=0))(X_shortcut_4)
    X_shortcut_4 = keras.layers.MaxPooling1D(pool_size=4, padding='same',name = block_mp+str(stage) + '4')(X_shortcut_4)

    X_shortcut_5 = keras.layers.Conv1D(filters=256, kernel_size=1, strides=1, name = block_conv + str(stage)  + '5', kernel_initializer = keras.initializers.he_normal(seed=0))(X_shortcut_5)
    X_shortcut_5 = keras.layers.MaxPooling1D(pool_size=4, padding='same',name = block_mp+str(stage) + '5')(X_shortcut_5)

    X = keras.layers.Add()([X, X_shortcut_1,X_shortcut_2,X_shortcut_3,X_shortcut_4,X_shortcut_5])

    X = keras.layers.BatchNormalization(name = 'bn_final')(X)
    X = keras.layers.Dropout(0.5)(X)
    X = keras.layers.Activation('relu')(X)
    #X = MaxPooling1D(pool_size=4,padding='same', name='max_pool_final')(X)

### LSTM 
    X = keras.layers.LSTM(units=128, return_sequences=True)(X)
    X = keras.layers.BatchNormalization()(X)
    X = keras.layers.LeakyReLU(alpha=0.3)(X)
    X = keras.layers.LSTM(units=128)(X)
    X = keras.layers.BatchNormalization()(X)
    X = keras.layers.LeakyReLU(alpha=0.3)(X)

    X=keras.layers.Dense(128, activation='relu',kernel_regularizer=keras.regularizers.l2(0.001))(X)
    X = keras.layers.Dropout(0.15)(X)

    X=keras.layers.Dense(64, activation='relu',kernel_regularizer=keras.regularizers.l2(0.001))(X)
    X = keras.layers.Dropout(0.15)(X)

    X=keras.layers.Dense(32, activation='relu',kernel_regularizer=keras.regularizers.l2(0.001))(X)
    X = keras.layers.Dropout(0.15)(X)

    X=keras.layers.Dense(16, activation='relu',kernel_regularizer=keras.regularizers.l2(0.001))(X)
    X = keras.layers.Dropout(0.15)(X)
 
   
    X = keras.layers.Dense(classes, activation='sigmoid', name='fc' + str(classes))(X)
    
    
    # Create model
    model = keras.models.Model(inputs = X_input, outputs = X, name='DenseResNet1D')
    #model = Model(inputs=[input1, input2], outputs=X,name='DenseResNet1D')
    
    # try:
    #     model = keras.models. multi_gpu_model(model, gpus=4, cpu_relocation=True)
    #     print("Training on 4 GPUs")
    # except:
    #     print("Training on 1 GPU/CPU")

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['categorical_accuracy'])

    return model

def Dense_Inc_ResNet_LSTM(n_timesteps, n_features, classes = 9):
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
    X_input = keras.layers.Input(shape=(n_timesteps, n_features), dtype='float32')
    # Zero-Padding
    X = keras.layers.ZeroPadding1D(0)(X_input)

    kernel_lr_1=13
    kernel_lr_2=7
    kernel_lr_3=5
    kernel_lr_4=3

    # Stage 1
    stage=1
    block_conv='conv_dense_a'
    block_mp='mp_dense_a'
    X = keras.layers.Conv1D(filters=4, kernel_size=21,padding='same', name = 'conv1', kernel_initializer = keras.initializers.he_normal(seed=0))(X)
    X = keras.layers.BatchNormalization(name = 'bn_conv1')(X)
    X = keras.layers.Activation('relu')(X)

    # Stage 2
    stage=2
    block_conv='conv_dense_b'
    block_mp='mp_dense_b'
    X_shortcut_1=X
    X=conv_inception_naive_resblock(X, kernel_lr_1, filters = [4, 4, 8], s=1, stage=2, block='A')
    X = convolutional_block(X, f = kernel_lr_1, filters = [4, 4, 8], s=2,stage = 2, block='a')

    X_shortcut_1 = keras.layers.Conv1D(filters=8, kernel_size=1, strides=1, name = block_conv + str(stage)  + '1', kernel_initializer = keras.initializers.he_normal(seed=0))(X_shortcut_1)
    X_shortcut_1 = keras.layers.MaxPooling1D(pool_size=2, padding='same',name = block_mp+str(stage) + '1')(X_shortcut_1)

    X = keras.layers.Add()([X, X_shortcut_1])
    X = keras.layers.Dropout(0.15)(X)

    X_shortcut_2=X

    # Stage 3
    stage=3
    block_conv='conv_dense_c'
    block_mp='mp_dense_c'
    X=conv_inception_naive_resblock(X, kernel_lr_2, filters = [8, 8, 16], s=1, stage=3, block='A')
    X = convolutional_block(X, f = kernel_lr_2, filters = [8, 8, 16], s=2,stage = 3, block='a')
    X_shortcut_1 = keras.layers.Conv1D(filters=16, kernel_size=1, strides=1, name = block_conv + str(stage)  + '1', kernel_initializer = keras.initializers.he_normal(seed=0))(X_shortcut_1)
    X_shortcut_1 = keras.layers.MaxPooling1D(pool_size=2, padding='same',name = block_mp+str(stage) + '1')(X_shortcut_1)
    X_shortcut_2 = keras.layers.Conv1D(filters=16, kernel_size=1, strides=1, name = block_conv + str(stage)  + '2', kernel_initializer = keras.initializers.he_normal(seed=0))(X_shortcut_2)
    X_shortcut_2 = keras.layers.MaxPooling1D(pool_size=2, padding='same',name = block_mp+str(stage) + '2')(X_shortcut_2)

    X = keras.layers.Add()([X, X_shortcut_1,X_shortcut_2])
    X = keras.layers.Dropout(0.15)(X)

    X_shortcut_3=X

    # Stage 4
    stage=4
    block_conv='conv_dense_d'
    block_mp='mp_dense_d'   

    X=conv_inception_naive_resblock(X, kernel_lr_3, filters = [16, 16, 32], s=1, stage=4, block='A')
    X = convolutional_block(X, f =kernel_lr_3 , filters = [16, 16, 32], s=2,stage = 4, block='a')

    X_shortcut_1 = keras.layers.Conv1D(filters=32, kernel_size=1, strides=1, name = block_conv + str(stage)  + '1', kernel_initializer = keras.initializers.he_normal(seed=0))(X_shortcut_1)
    X_shortcut_1 = keras.layers.MaxPooling1D(pool_size=2, padding='same',name = block_mp+str(stage) + '1')(X_shortcut_1)

    X_shortcut_2 = keras.layers.Conv1D(filters=32, kernel_size=1, strides=1, name = block_conv + str(stage)  + '2', kernel_initializer = keras.initializers.he_normal(seed=0))(X_shortcut_2)
    X_shortcut_2 = keras.layers.MaxPooling1D(pool_size=2, padding='same',name = block_mp+str(stage) + '2')(X_shortcut_2)

    X_shortcut_3 = keras.layers.Conv1D(filters=32, kernel_size=1, strides=1, name = block_conv + str(stage)  + '3', kernel_initializer = keras.initializers.he_normal(seed=0))(X_shortcut_3)
    X_shortcut_3 = keras.layers.MaxPooling1D(pool_size=2, padding='same',name = block_mp+str(stage) + '3')(X_shortcut_3)

    X = keras.layers.Add()([X, X_shortcut_1,X_shortcut_2,X_shortcut_3])
    X = keras.layers.Dropout(0.15)(X)


    X_shortcut_4=X
    # Stage 5
    stage=5
    block_conv='conv_dense_e'
    block_mp='mp_dense_e'

    X=conv_inception_naive_resblock(X, kernel_lr_4, filters = [32, 32, 64], s=1, stage=5, block='A')
    X = convolutional_block(X, f = kernel_lr_4, filters = [32, 32, 64], s=2,stage = 5, block='a')

    X_shortcut_1 = keras.layers.Conv1D(filters=64, kernel_size=1, strides=1, name = block_conv + str(stage)  + '1', kernel_initializer = keras.initializers.he_normal(seed=0))(X_shortcut_1)
    X_shortcut_1 = keras.layers.MaxPooling1D(pool_size=2, padding='same',name = block_mp+str(stage) + '1')(X_shortcut_1)

    X_shortcut_2 = keras.layers.Conv1D(filters=64, kernel_size=1, strides=1, name = block_conv + str(stage)  + '2', kernel_initializer = keras.initializers.he_normal(seed=0))(X_shortcut_2)
    X_shortcut_2 = keras.layers.MaxPooling1D(pool_size=2, padding='same',name = block_mp+str(stage) + '2')(X_shortcut_2)

    X_shortcut_3 = keras.layers.Conv1D(filters=64, kernel_size=1, strides=1, name = block_conv + str(stage)  + '3', kernel_initializer = keras.initializers.he_normal(seed=0))(X_shortcut_3)
    X_shortcut_3 = keras.layers.MaxPooling1D(pool_size=2, padding='same',name = block_mp+str(stage) + '3')(X_shortcut_3)

    X_shortcut_4 = keras.layers.Conv1D(filters=64, kernel_size=1, strides=1, name = block_conv + str(stage)  + '4', kernel_initializer = keras.initializers.he_normal(seed=0))(X_shortcut_4)
    X_shortcut_4 = keras.layers.MaxPooling1D(pool_size=2, padding='same',name = block_mp+str(stage) + '4')(X_shortcut_4)

    X = keras.layers.Add()([X, X_shortcut_1,X_shortcut_2,X_shortcut_3,X_shortcut_4])
    X = keras.layers.Dropout(0.15)(X)

    X_shortcut_5=X

    # Stage 6
    stage=6
    block_conv='conv_dense_f'
    block_mp='mp_dense_f'

    X=conv_inception_naive_resblock(X, kernel_lr_4, filters = [64, 64, 128], s=1, stage=6, block='A')
    X = convolutional_block(X, f = kernel_lr_4, filters = [64, 64, 128], s=2,stage = 6, block='a')

    X_shortcut_1 = keras.layers.Conv1D(filters=128, kernel_size=1, strides=1, name = block_conv + str(stage)  + '1', kernel_initializer = keras.initializers.he_normal(seed=0))(X_shortcut_1)
    X_shortcut_1 = keras.layers.MaxPooling1D(pool_size=2, padding='same',name = block_mp+str(stage) + '1')(X_shortcut_1)

    X_shortcut_2 = keras.layers.Conv1D(filters=128, kernel_size=1, strides=1, name = block_conv + str(stage)  + '2', kernel_initializer = keras.initializers.he_normal(seed=0))(X_shortcut_2)
    X_shortcut_2 = keras.layers.MaxPooling1D(pool_size=2, padding='same',name = block_mp+str(stage) + '2')(X_shortcut_2)

    X_shortcut_3 = keras.layers.Conv1D(filters=128, kernel_size=1, strides=1, name = block_conv + str(stage)  + '3', kernel_initializer = keras.initializers.he_normal(seed=0))(X_shortcut_3)
    X_shortcut_3 = keras.layers.MaxPooling1D(pool_size=2, padding='same',name = block_mp+str(stage) + '3')(X_shortcut_3)

    X_shortcut_4 = keras.layers.Conv1D(filters=128, kernel_size=1, strides=1, name = block_conv + str(stage)  + '4', kernel_initializer = keras.initializers.he_normal(seed=0))(X_shortcut_4)
    X_shortcut_4 = keras.layers.MaxPooling1D(pool_size=2, padding='same',name = block_mp+str(stage) + '4')(X_shortcut_4)

    X_shortcut_5 = keras.layers.Conv1D(filters=128, kernel_size=1, strides=1, name = block_conv + str(stage)  + '5', kernel_initializer = keras.initializers.he_normal(seed=0))(X_shortcut_5)
    X_shortcut_5 = keras.layers.MaxPooling1D(pool_size=2, padding='same',name = block_mp+str(stage) + '5')(X_shortcut_5)

    X = keras.layers.Add()([X, X_shortcut_1,X_shortcut_2,X_shortcut_3,X_shortcut_4,X_shortcut_5])

    X_shortcut_6=X
    # Stage 7
    stage=7
    block_conv='conv_dense_g'
    block_mp='mp_dense_g'
    X=conv_inception_naive_resblock(X, kernel_lr_4, filters = [128, 128, 256], s=1, stage=7, block='A')
    X = convolutional_block(X, f = kernel_lr_4, filters = [128, 128, 256], s=2,stage = 7, block='a')

    X_shortcut_1 = keras.layers.Conv1D(filters=256, kernel_size=1, strides=1, name = block_conv + str(stage)  + '1', kernel_initializer = keras.initializers.he_normal(seed=0))(X_shortcut_1)
    X_shortcut_1 = keras.layers.MaxPooling1D(pool_size=2, padding='same',name = block_mp+str(stage) + '1')(X_shortcut_1)

    X_shortcut_2 = keras.layers.Conv1D(filters=256, kernel_size=1, strides=1, name = block_conv + str(stage)  + '2', kernel_initializer = keras.initializers.he_normal(seed=0))(X_shortcut_2)
    X_shortcut_2 = keras.layers.MaxPooling1D(pool_size=2, padding='same',name = block_mp+str(stage) + '2')(X_shortcut_2)

    X_shortcut_3 = keras.layers.Conv1D(filters=256, kernel_size=1, strides=1, name = block_conv + str(stage)  + '3', kernel_initializer = keras.initializers.he_normal(seed=0))(X_shortcut_3)
    X_shortcut_3 = keras.layers.MaxPooling1D(pool_size=2, padding='same',name = block_mp+str(stage) + '3')(X_shortcut_3)

    X_shortcut_4 = keras.layers.Conv1D(filters=256, kernel_size=1, strides=1, name = block_conv + str(stage)  + '4', kernel_initializer = keras.initializers.he_normal(seed=0))(X_shortcut_4)
    X_shortcut_4 = keras.layers.MaxPooling1D(pool_size=2, padding='same',name = block_mp+str(stage) + '4')(X_shortcut_4)

    X_shortcut_5 = keras.layers.Conv1D(filters=256, kernel_size=1, strides=1, name = block_conv + str(stage)  + '5', kernel_initializer = keras.initializers.he_normal(seed=0))(X_shortcut_5)
    X_shortcut_5 = keras.layers.MaxPooling1D(pool_size=2, padding='same',name = block_mp+str(stage) + '5')(X_shortcut_5)

    X_shortcut_6 = keras.layers.Conv1D(filters=256, kernel_size=1, strides=1, name = block_conv + str(stage)  + '6', kernel_initializer = keras.initializers.he_normal(seed=0))(X_shortcut_6)
    X_shortcut_6 = keras.layers.MaxPooling1D(pool_size=2, padding='same',name = block_mp+str(stage) + '6')(X_shortcut_6)

    X = keras.layers.Add()([X, X_shortcut_1,X_shortcut_2,X_shortcut_3,X_shortcut_4,X_shortcut_5, X_shortcut_6])

    #X = GlobalAveragePooling1D(data_format='channels_last')(X)

    #X = BatchNormalization(name = 'bn_final')(X)
    #X = Activation('relu')(X)
    #X = Dropout(0.2)(X)

    X = keras.layers.LSTM(units=256, return_sequences=False)(X)
    X = keras.layers.BatchNormalization()(X)
    X = keras.layers.Activation('relu')(X)
    #X = LSTM(units=256)(X)
   # X = BatchNormalization()(X)
  #  X = Activation('relu')(X)
    # X = Flatten()(X)
    X = keras.layers.Dense(classes, activation='sigmoid', name='fc' + str(classes))(X)
    
    
    # Create model
    model = keras.models.Model(inputs = X_input, outputs = X, name='Dense_ResNet_Inc_LSTM')
    
    # try:
    #     model = multi_gpu_model(model, gpus=4, cpu_relocation=True)
    #     print("Training on 4 GPUs")
    # except:
    #     print("Training on 1 GPU/CPU")

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['categorical_accuracy'])

    return model


##################################################### CONV_MAXPool only
def TinyResNet1D(n_timesteps, n_features, classes = 9):
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
    X_input = keras.layers.Input(shape=(n_timesteps, n_features), dtype='float32')
    # Zero-Padding
    X = keras.layers.ZeroPadding1D(0)(X_input)
    
    # Stage 1
    X = keras.layers.Conv1D(filters=2, kernel_size=31,padding='same', name = 'conv1', kernel_initializer = keras.initializers.he_normal(seed=0))(X)
    X = keras.layers.BatchNormalization(name = 'bn_conv1')(X)
    X = keras.layers.Activation('relu')(X)

    X = keras.layers.Conv1D(filters=4, kernel_size=31,padding='same', name = 'conv2', kernel_initializer = keras.initializers.he_normal(seed=0))(X)
    X = keras.layers.BatchNormalization(name = 'bn_conv2')(X)
    X = keras.layers.Activation('relu')(X)
    # Stage 2
    X = maxpool_block_1(X, f = 31, filters = [16, 32], s=1,stage = 2, block='a')
    
    # Stage 3 (≈4 lines)
    X = maxpool_block_2(X, 15, filters = [4, 8], s=2, stage=3, block='a')
    X = maxpool_block_2(X, 15, filters = [4, 8],s=1, stage=3, block='b')
    X = maxpool_block_2(X, 15, filters = [4, 8],s=2, stage=3, block='c')
    X = maxpool_block_2(X, 15, filters = [8, 16],s=1, stage=3, block='d')
    X = maxpool_block_2(X, 15, filters = [8, 16], s=2,stage=3, block='e')
    
    X = maxpool_block_2(X, 7, filters = [16, 32],s=1, stage=3, block='f')
    X = maxpool_block_2(X, 7, filters = [16, 32], s=2,stage=3, block='g')
    X = maxpool_block_2(X, 7, filters = [32, 64],s=1, stage=3, block='h')
    X = maxpool_block_2(X, 7, filters = [32, 64], s=2,stage=3, block='i')
    X = maxpool_block_2(X, 7, filters = [32, 64], s=1,stage=3, block='j')
    
    X = maxpool_block_2(X, 3, filters = [32, 64],s=2, stage=3, block='k')
    X = maxpool_block_2(X, 3, filters = [64, 128],s=1, stage=3, block='l')
    X = maxpool_block_2(X, 3, filters = [64, 128],s=2, stage=3, block='m')
    X = maxpool_block_2(X, 3, filters = [64, 128],s=1, stage=3, block='n')
    X = maxpool_block_2(X, 3, filters = [64, 128],s=2, stage=3, block='o')
    
    X = keras.layers.BatchNormalization(name = 'bn_final')(X)
    X = keras.layers.Activation('relu')(X)
    X = keras.layers.Dropout(0.2)(X)
    X = keras.layers.LSTM(units=128, return_sequences=True)(X)
    X = keras.layers.BatchNormalization()(X)
    X = keras.layers.LeakyReLU(alpha=0.3)(X)
    X = keras.layers.LSTM(units=128)(X)
    X = keras.layers.BatchNormalization()(X)
    X = keras.layers.LeakyReLU(alpha=0.3)(X)
 
    #X = BatchNormalization(name = 'bn_final')(X)
    #X = Activation('relu')(X)
    X = keras.layers.Dropout(0.6)(X)
    # X = Flatten()(X)
    X=keras.layers.Dense(128, activation='relu',kernel_regularizer=keras.regularizers.l2(0.001))(X)
    X = keras.layers.Dropout(0.6)(X)

    X=keras.layers.Dense(64, activation='relu',kernel_regularizer=keras.regularizers.l2(0.001))(X)
    X = keras.layers.Dropout(0.6)(X)

    X=keras.layers.Dense(32, activation='relu',kernel_regularizer=keras.regularizers.l2(0.001))(X)
    X = keras.layers.Dropout(0.6)(X)

    X=keras.layers.Dense(16, activation='relu',kernel_regularizer=keras.regularizers.l2(0.001))(X)
    X = keras.layers.Dropout(0.15)(X)

    X = keras.layers.Dense(classes, activation='sigmoid', name='fc' + str(classes))(X)
    
    
    # Create model
    model = keras.models.Model(inputs = X_input, outputs = X, name='TinyResNet1D')
    
    # try:
    #     model = multi_gpu_model(model, gpus=4, cpu_relocation=True)
    #     print("Training on 4 GPUs")
    # except:
    #     print("Training on 1 GPU/CPU")

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['categorical_accuracy'])

    return model

def Inc_ResNet_LSTM_v01(n_timesteps, n_features, classes = 9):
    """
    Arguments:
    input_shape -- shape of the signals of the dataset (window size= n_timesteps and nfeatures= number of channels)
    classes -- integer, number of classes

    Returns:
    model -- a Model() instance in Keras
    """
    
    # Define the input as a tensor with shape input_shape
    X_input = keras.layers.Input(shape=(n_timesteps, n_features), dtype='float32')
    # Zero-Padding
    X = keras.layers.ZeroPadding1D(0)(X_input)

    kernel_lr_1=13
    kernel_lr_2=7
    kernel_lr_3=5
    kernel_lr_4=3

    # Stage 1
    X = keras.layers.Conv1D(filters=16, kernel_size=kernel_lr_1,padding='same', name = 'conv1', kernel_initializer = keras.initializers.he_normal(seed=0))(X)
    X = keras.layers.BatchNormalization(name = 'bn_conv1')(X)
    X = keras.layers.Activation('relu')(X)

    # Stage 2

    X=conv_inception_naive_resblock(X, kernel_lr_1, filters = [16, 16, 32], s=1, stage=2, block='A')
    X = convolutional_block(X, f = kernel_lr_1, filters = [16, 16, 32], s=2,stage = 2, block='a')

    X=conv_inception_naive_resblock(X, kernel_lr_2, filters = [32, 32, 64], s=1, stage=3, block='A')
    X = convolutional_block(X, f = kernel_lr_2, filters = [32, 32, 64], s=2,stage = 3, block='a')

    X=conv_inception_naive_resblock(X, kernel_lr_3, filters = [64, 64, 128], s=1, stage=4, block='A')
    X = convolutional_block(X, f =kernel_lr_3 , filters = [64, 64, 128], s=2,stage = 4, block='a')

    X=conv_inception_naive_resblock(X, kernel_lr_4, filters = [128, 128, 256], s=1, stage=5, block='A')
    X = convolutional_block(X, f = kernel_lr_4, filters = [128, 128, 256], s=2,stage = 5, block='a')

    X=conv_inception_naive_resblock(X, kernel_lr_4, filters = [256, 256, 512], s=1, stage=6, block='A')
    X = convolutional_block(X, f = kernel_lr_4, filters = [256, 256, 512], s=2,stage = 6, block='a')

    X=conv_inception_naive_resblock(X, kernel_lr_4, filters = [512, 512, 1024], s=1, stage=7, block='A')
    X = convolutional_block(X, f = kernel_lr_4, filters = [512, 512, 1024], s=2,stage = 7, block='a')


    X = keras.layers.LSTM(units=256, return_sequences=False)(X)
    X = keras.layers.BatchNormalization()(X)
    X = keras.layers.Activation('relu')(X)

    X = keras.layers.Dense(classes, activation='sigmoid', name='fc' + str(classes))(X)
    
    
    # Create model
    model = keras.models.Model(inputs = X_input, outputs = X, name='ResNet_Inc_LSTM')
    
    # try:
    #     model = multi_gpu_model(model, gpus=4, cpu_relocation=True)
    #     print("Training on 4 GPUs")
    # except:
    #     print("Training on 1 GPU/CPU")

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['categorical_accuracy'])

    return model


def conv1d(x,numfilt,filtsz,strides=1,pad='same',act=True,name=None):
  x = keras.layers.Conv1D(filters=numfilt,kernel_size=filtsz,strides=strides,padding=pad,name=name+'conv1d',kernel_initializer = keras.initializers.he_normal(seed=0))(x)
  x = keras.layers.BatchNormalization(name=name+'conv1d'+'bn')(x)
  if act:
    x = keras.layers.Activation('relu',name=name+'conv1d'+'act')(x)
  return x


def Inc_ResNet_LSTM_v02(n_timesteps, n_features, classes = 9):
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
    X_input = keras.layers.Input(shape=(n_timesteps, n_features), dtype='float32')
    # Zero-Padding
    X = keras.layers.ZeroPadding1D(0)(X_input)
    kernel_lr_0=19
    kernel_lr_1=13
    kernel_lr_2=9
    kernel_lr_3=7
    kernel_lr_4=5
    kernel_lr_5=3

    # Stage 1
    X = conv1d(X_input,8,kernel_lr_0,2,'same',True,name='conv1')
    X = conv1d(X,8,kernel_lr_0,1,'same',True,name='conv2')
    X = conv1d(X,16,kernel_lr_0,1,'same',True,name='conv3')

    x_11 = keras.layers.MaxPooling1D(pool_size=3,strides=1,padding='same',name='stem_br_11'+'_maxpool_1d')(X)
    x_12 = conv1d(X,16,kernel_lr_0,1,'same',True,name='stem_br_12')
    X = keras.layers.Concatenate(name = 'stem_concat_1')([x_11,x_12])

    X = keras.layers.Conv1D(filters=16, kernel_size=kernel_lr_1,padding='same', name = 'conv1', kernel_initializer = keras.initializers.he_normal(seed=0))(X)
    X = keras.layers.BatchNormalization(name = 'bn_conv1')(X)
    X = keras.layers.Activation('relu')(X)

    X = keras.layers.MaxPooling1D(pool_size=3,strides=1,padding='same',name='stem_br_12'+'_maxpool_1d')(X)

    # Stage 2

    X=conv_inception_naive_resblock(X, kernel_lr_1, filters = [16, 16, 32], s=1, stage=2, block='A')
    X = convolutional_block(X, f = kernel_lr_1, filters = [16, 16, 32], s=2,stage = 2, block='a')

    X=conv_inception_naive_resblock(X, kernel_lr_2, filters = [32, 32, 64], s=1, stage=3, block='A')
    X = convolutional_block(X, f = kernel_lr_2, filters = [32, 32, 64], s=2,stage = 3, block='a')

    X=conv_inception_naive_resblock(X, kernel_lr_3, filters = [64, 64, 128], s=1, stage=4, block='A')
    X = convolutional_block(X, f =kernel_lr_3 , filters = [64, 64, 128], s=2,stage = 4, block='a')

    X=conv_inception_naive_resblock(X, kernel_lr_4, filters = [128, 128, 256], s=1, stage=5, block='A')
    X = convolutional_block(X, f = kernel_lr_4, filters = [128, 128, 256], s=2,stage = 5, block='a')

    X=conv_inception_naive_resblock(X, kernel_lr_5, filters = [256, 256, 512], s=1, stage=6, block='A')
    X = convolutional_block(X, f = kernel_lr_5, filters = [256, 256, 512], s=2,stage = 6, block='a')

   # X=conv_inception_naive_resblock(X, kernel_lr_4, filters = [512, 512, 1024], s=1, stage=7, block='A')
    #X = convolutional_block(X, f = kernel_lr_4, filters = [512, 512, 1024], s=2,stage = 7, block='a')


    X = keras.layers.LSTM(units=256, return_sequences=False)(X)
    X = keras.layers.BatchNormalization()(X)
    X = keras.layers.Activation('relu')(X)

    X = keras.layers.Dense(classes, activation='sigmoid', name='fc' + str(classes))(X)
    
    
    # Create model
    model = keras.models.Model(inputs = X_input, outputs = X, name='ResNet_Inc_LSTM')
    
    # try:
    #     model = multi_gpu_model(model, gpus=4, cpu_relocation=True)
    #     print("Training on 4 GPUs")
    # except:
    #     print("Training on 1 GPU/CPU")

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['categorical_accuracy'])

    return model