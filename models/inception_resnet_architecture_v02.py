

"""ResNet_module.ipynb

Created on Wed Nov 13 16:11:12 2019

@author: Mojtaba Jafaritadi, Ph.D.
"""

import numpy as np
from tensorflow.keras import backend
from keras import layers
from keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalAveragePooling1D,LeakyReLU
from keras.models import Model, load_model
from keras import regularizers 
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
from keras.initializers import glorot_uniform,he_normal
import scipy.misc
from matplotlib.pyplot import imshow
from keras.callbacks import  Callback, ModelCheckpoint, ReduceLROnPlateau, LearningRateScheduler
from keras.layers import Input, ZeroPadding1D, Dropout, LSTM, CuDNNLSTM, GRU, concatenate,Concatenate, Bidirectional,RepeatVector, Reshape,Lambda
from keras.layers.convolutional import Conv1D, AveragePooling1D, MaxPooling1D
from keras.models import Sequential
from keras.regularizers import l1,l2
from keras import optimizers
from keras.layers import TimeDistributed
import keras.backend as K
from keras.utils import multi_gpu_model



def conv1d(x,numfilt,filtsz,strides=1,pad='same',act=True,name=None):
  x = Conv1D(filters=numfilt,kernel_size=filtsz,strides=strides,padding=pad,name=name+'conv1d',kernel_initializer = he_normal(seed=0))(x)
  x = BatchNormalization(name=name+'conv1d'+'bn')(x)
  if act:
    x = Activation('relu',name=name+'conv1d'+'act')(x)
  return x


def incresA(x,scale,name=None):
    pad = 'same'
    branch0 = conv1d(x,32,1,1,pad,True,name=name+'b0')
    branch1 = conv1d(x,32,1,1,pad,True,name=name+'b1_1')
    branch1 = conv1d(branch1,32,3,1,pad,True,name=name+'b1_2')
    branch2 = conv1d(x,32,1,1,pad,True,name=name+'b2_1')
    branch2 = conv1d(branch2,48,3,1,pad,True,name=name+'b2_2')
    branch2 = conv1d(branch2,64,3,1,pad,True,name=name+'b2_3')
    branches = [branch0,branch1,branch2]
    mixed = Concatenate(name=name + '_concat')(branches)
    filt_exp_1x1 = conv1d(mixed,384,1,1,pad,False,name=name+'filt_exp_1x1')
    final_lay = Lambda(lambda inputs, scale: inputs[0] + inputs[1] * scale,
                      output_shape=backend.int_shape(x)[1:],
                      arguments={'scale': scale},
                      name=name+'act_scaling')([x, filt_exp_1x1])
    return final_lay

def incresB(x,scale,name=None):
    pad = 'same'
    branch0 = conv1d(x,192,1,1,pad,True,name=name+'b0')
    branch1 = conv1d(x,128,1,1,pad,True,name=name+'b1_1')
    branch1 = conv1d(branch1,160,7,1,pad,True,name=name+'b1_2')
    branch1 = conv1d(branch1,192,7,1,pad,True,name=name+'b1_3')
    branches = [branch0,branch1]
    mixed = Concatenate(name=name + '_mixed')(branches)
    filt_exp_1x1 = conv1d(mixed,1152,1,1,pad,False,name=name+'filt_exp_1x1')
    final_lay = Lambda(lambda inputs, scale: inputs[0] + inputs[1] * scale,
                      output_shape=backend.int_shape(x)[1:],
                      arguments={'scale': scale},
                      name=name+'act_scaling')([x, filt_exp_1x1])
    return final_lay


def incresC(x,scale,name=None):
    pad = 'same'
    branch0 = conv1d(x,192,1,1,pad,True,name=name+'b0')
    branch1 = conv1d(x,192,1,1,pad,True,name=name+'b1_1')
    branch1 = conv1d(branch1,224,3,1,pad,True,name=name+'b1_2')
    branch1 = conv1d(branch1,256,3,1,pad,True,name=name+'b1_3')
    branches = [branch0,branch1]
    mixed = Concatenate(name=name + '_mixed')(branches)
    filt_exp_1x1 = conv1d(mixed,2048,1,1,pad,False,name=name+'fin1x1')
    final_lay = Lambda(lambda inputs, scale: inputs[0] + inputs[1] * scale,
                      output_shape=backend.int_shape(x)[1:],
                      arguments={'scale': scale},
                      name=name+'act_saling')([x, filt_exp_1x1])
    return final_lay


def InceptionResNet1D(n_timesteps, n_features, classes = 9):

    input_x = Input(shape=(n_timesteps, n_features), dtype='float32')

    x = conv1d(input_x,32,3,2,'same',True,name='conv1')
    x = conv1d(x,32,3,1,'same',True,name='conv2')
    x = conv1d(x,64,3,1,'same',True,name='conv3')

    x_11 = MaxPooling1D(pool_size=3,strides=1,padding='same',name='stem_br_11'+'_maxpool_1')(x)
    print(x_11.shape)
    x_12 = conv1d(x,64,3,1,'same',True,name='stem_br_12')
    print(x_12.shape)
    x = Concatenate(name = 'stem_concat_1')([x_11,x_12])

    x_21 = conv1d(x,64,1,1,'same',True,name='stem_br_211')
    x_21 = conv1d(x_21,64,7,1,'same',True,name='stem_br_212')
    x_21 = conv1d(x_21,64,7,1,'same',True,name='stem_br_213')
    x_21 = conv1d(x_21,96,3,1,'valid',True,name='stem_br_214')

    x_22 = conv1d(x,64,1,1,'same',True,name='stem_br_221')
    x_22 = conv1d(x_22,96,3,1,'valid',True,name='stem_br_222')

    x = Concatenate(axis=2, name = 'stem_concat_2')([x_21,x_22])

    x_31 = conv1d(x,192,3,1,'valid',True,name='stem_br_31')
    x_32 = MaxPooling1D(3,strides=1,padding='valid',name='stem_br_32'+'_maxpool_2')(x)
    x = Concatenate(name = 'stem_concat_3')([x_31,x_32])


    x = incresA(x,0.15,name='incresA_1')
    x = incresA(x,0.15,name='incresA_2')
    x = incresA(x,0.15,name='incresA_3')
    x = incresA(x,0.15,name='incresA_4')

    #35 × 35 to 17 × 17 reduction module.
    x_red_11 = MaxPooling1D(3,strides=2,padding='valid',name='red_maxpool_1')(x)

    x_red_12 = conv1d(x,384,3,2,'valid',True,name='x_red1_c1')

    x_red_13 = conv1d(x,256,1,1,'same',True,name='x_red1_c2_1')
    x_red_13 = conv1d(x_red_13,256,3,1,'same',True,name='x_red1_c2_2')
    x_red_13 = conv1d(x_red_13,384,3,2,'valid',True,name='x_red1_c2_3')

    x = Concatenate(name = 'stem_concat_4')([x_red_11,x_red_12,x_red_13])

    #Inception-ResNet-B modules
    x = incresB(x,0.1,name='incresB_1')
    x = incresB(x,0.1,name='incresB_2')
    x = incresB(x,0.1,name='incresB_3')
    x = incresB(x,0.1,name='incresB_4')
    x = incresB(x,0.1,name='incresB_5')
    x = incresB(x,0.1,name='incresB_6')
    x = incresB(x,0.1,name='incresB_7')

    #17 × 17 to 8 × 8 reduction module.
    x_red_21 = MaxPooling1D(3,strides=2,padding='valid',name='red_maxpool_2')(x)

    x_red_22 = conv1d(x,256,1,1,'same',True,name='x_red2_c11')
    x_red_22 = conv1d(x_red_22,384,3,2,'valid',True,name='x_red2_c12')

    x_red_23 = conv1d(x,256,1,1,'same',True,name='x_red2_c21')
    x_red_23 = conv1d(x_red_23,256,3,2,'valid',True,name='x_red2_c22')

    x_red_24 = conv1d(x,256,1,1,'same',True,name='x_red2_c31')
    x_red_24 = conv1d(x_red_24,256,3,1,'same',True,name='x_red2_c32')
    x_red_24 = conv1d(x_red_24,256,3,2,'valid',True,name='x_red2_c33')

    x = Concatenate(name = 'stem_concat_5')([x_red_21,x_red_22,x_red_23,x_red_24])

    #Inception-ResNet-C modules
    x = incresC(x,0.2,name='incresC_1')
    x = incresC(x,0.2,name='incresC_2')
    x = incresC(x,0.2,name='incresC_3')

    #TOP
    x = GlobalAveragePooling1D(data_format='channels_last')(x)
    x = Dropout(0.5)(x)

    x = Dense(classes, activation='sigmoid', name='fc' + str(classes))(x)
    
    
    # Create model
    model = Model(inputs = input_x, outputs = x, name='InceptionResNet1D')
    #model = Model(inputs=[input1, input2], outputs=X,name='DenseResNet1D')
    
    try:
        model = multi_gpu_model(model, gpus=4, cpu_relocation=True)
        print("Training on 4 GPUs")
    except:
        print("Training on 1 GPU/CPU")

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['categorical_accuracy'])

    return model