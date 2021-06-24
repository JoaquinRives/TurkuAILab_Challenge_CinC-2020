import numpy as np
from tensorflow.keras.callbacks import Callback
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from utils import custom_multiclass_f1


class MetricsCallback(Callback):

    def __init__(self, validation_data):
        self.validation_data = validation_data

    def on_train_begin(self, logs={}):
        self.val_f1s = []
        self.val_fbetas = []

    def on_epoch_end(self, epoch, logs={}):
        val_predict = (np.asarray(self.model.predict(self.validation_data[0]))).round()
        val_targ = self.validation_data[1]
        _val_f1 = f1_score(val_targ, val_predict, average='macro')
        _val_fbeta = custom_multiclass_f1(val_targ, val_predict)
        self.val_f1s.append(_val_f1)
        self.val_fbetas.append(_val_fbeta)
        print()
        print("             — val_f1: % f — _val_fbeta: % f " % (_val_f1, _val_fbeta))
        print()
        return