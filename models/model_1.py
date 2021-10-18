"""
Residual network model (stage_1_2)

@authors: Joaquin Rives, Mojtaba Jafaritadi, Ph.D.
@email: joaquin.rives01@gmail.com
"""

from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras import Input, Model
from sklearn.model_selection import StratifiedKFold
# from logger import Logger
from preprocess_and_segmentation import load_data, segment_all_dict_data, reshape_segmented_arrays
from preprocessor import preprocess_input_data
import numpy as np
import pandas as pd
import os
from tensorflow.keras.layers import Dense, LSTM, Dropout, Convolution1D, LeakyReLU, BatchNormalization
from tensorflow.keras.utils import multi_gpu_model
from utils import encode_labels, custom_multiclass_f1, multiclass_f1
import shutil
import ntpath
from collections import Counter
from tensorflow.keras.models import load_model


def stage_1_1(n_timesteps, n_features, n_outputs):
    """ CNN-LSTM model (stage_1_1) """

    input = Input(shape=(n_timesteps, n_features), dtype='float32')

    x = Convolution1D(12, 3, padding='same')(input)
    x = LeakyReLU(alpha=0.3)(x)
    x = Convolution1D(12, 3, padding='same')(x)
    x = LeakyReLU(alpha=0.3)(x)
    x = Convolution1D(12, 24, strides=2, padding='same')(x)
    x = LeakyReLU(alpha=0.3)(x)
    x = Dropout(0.2)(x)
    x = Convolution1D(12, 3, padding='same')(x)
    x = LeakyReLU(alpha=0.3)(x)
    x = Convolution1D(12, 3, padding='same')(x)
    x = LeakyReLU(alpha=0.3)(x)
    x = Convolution1D(12, 24, strides=2, padding='same')(x)
    x = LeakyReLU(alpha=0.3)(x)
    x = Dropout(0.2)(x)
    x = Convolution1D(12, 3, padding='same')(x)
    x = LeakyReLU(alpha=0.3)(x)
    x = Convolution1D(12, 3, padding='same')(x)
    x = LeakyReLU(alpha=0.3)(x)
    x = Convolution1D(12, 24, strides=2, padding='same')(x)
    x = LeakyReLU(alpha=0.3)(x)
    x = Dropout(0.2)(x)
    x = Convolution1D(12, 3, padding='same')(x)
    x = LeakyReLU(alpha=0.3)(x)
    x = Convolution1D(12, 3, padding='same')(x)
    x = LeakyReLU(alpha=0.3)(x)
    x = Convolution1D(12, 24, strides=2, padding='same')(x)
    x = LeakyReLU(alpha=0.3)(x)
    cnnout = Dropout(0.2)(x)

    x = LSTM(units=200, return_sequences=True)(cnnout)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.3)(x)
    x = LSTM(units=200)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.3)(x)
    output = Dense(n_outputs, activation='sigmoid')(x)

    model = Model(inputs=input, outputs=output)

    return model

