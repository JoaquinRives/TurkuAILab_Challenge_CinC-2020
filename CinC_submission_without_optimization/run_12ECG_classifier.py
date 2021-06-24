#!/usr/bin/env python

# command:  python evaluate_12ECG_score.py labels output scores.csv
# labels: D:\OneDrive\Desktop\CinC_project\data\test_balanced

import joblib
from keras.engine.saving import load_model, model_from_json
from preprocessor import preprocess_input
from main_model import MainModel
import tensorflow.keras as keras


def run_12ECG_classifier(data, header_data, classes, model):
    # pre-process input signals
    input_data = preprocess_input(data)

    # predict
    current_label, current_score = model.predict(input_data)

    return current_label[0], current_score[0]


def load_12ECG_model():
    # load the model from disk

    # # Models stage 1
    # stage_1_1 = load_model('models/stage_1_1.h5')  # CNN-LSTM
    # stage_1_2 = keras.models.load_model('models/stage_1_2.h5')  # residual network

    #####################################################################################
    with open('models/stage_1_1.json', 'r') as json_file:
        loaded_model_json = json_file.read()
        stage_1_1 = model_from_json(loaded_model_json)
        stage_1_1.load_weights("models/stage_1_1_w.h5")

    with open('models/stage_1_2.json', 'r') as json_file:
        loaded_model_json = json_file.read()
        stage_1_2 = keras.models.model_from_json(loaded_model_json)
        stage_1_2.load_weights("models/stage_1_2_w.h5")

    with open('models/stage_1_3.json', 'r') as json_file:
        loaded_model_json = json_file.read()
        stage_1_3 = keras.models.model_from_json(loaded_model_json)
        stage_1_3.load_weights("models/stage_1_3_w.h5")

    with open('models/stage_1_4.json', 'r') as json_file:
        loaded_model_json = json_file.read()
        stage_1_4 = keras.models.model_from_json(loaded_model_json)
        stage_1_4.load_weights("models/stage_1_4_w.h5")

    ####################################################################################

    # Model stage 2
    stage_2 = load_model('models/stage_2.h5')  # LSTM

    # Main model
    loaded_model = MainModel(stage_1_1, stage_1_2, stage_1_3, stage_1_4, stage_2)

    return loaded_model
