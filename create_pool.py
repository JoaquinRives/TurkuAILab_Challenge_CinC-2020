"""
Training of any of the models of the stage 1

@author: Joaquin Rives
"""
import re

import h5py
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

from sklearn.model_selection import StratifiedKFold
import os

from tensorflow.python.keras.models import load_model

from utils import custom_multiclass_f1, multiclass_f1, onehot_labels
import shutil
import ntpath
import h5py
import numpy as np
import numpy as np
from preprocess_and_segmentation import load_data, segment_all_dict_data, reshape_segmented_arrays
from preprocessor import preprocess_input_data
from utils import encode_labels
import tensorflow.keras as keras
import tensorflow.keras.backend as K
import pandas as pd
import sys
import matplotlib.pyplot as plt
# from tensorflow_addons.metrics import FBetaScore
from metrics import MetricsCallback
from data_generator import DataGenerator, DataGeneratorFromPool

from models.model_1 import stage_1_1
from models.model_2 import ResNet1D
from models.model_architecture_V01 import Inc_ResNet_LSTM_v01, Inc_ResNet_LSTM_v02

import tensorflow as tf
from tensorflow.keras import backend as K


def mcc_loss(y_true, y_pred):
    tp = K.sum(K.cast(y_true * y_pred, 'float'), axis=0)
    tn = K.sum(K.cast((1 - y_true) * (1 - y_pred), 'float'), axis=0)
    fp = K.sum(K.cast((1 - y_true) * y_pred, 'float'), axis=0) * 1e2
    fn = K.sum(K.cast(y_true * (1 - y_pred), 'float'), axis=0) / 1e2

    up = tp * tn - fp * fn
    down = K.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))

    mcc = up / (down + K.epsilon())
    mcc = tf.where(tf.math.is_nan(mcc), tf.zeros_like(mcc), mcc)

    return 1 - K.mean(mcc)


class SWA(keras.callbacks.Callback):
    """ Stochastic Weight Averaging """

    def __init__(self, filepath, swa_epoch):
        super(SWA, self).__init__()
        self.filepath = filepath
        self.swa_epoch = swa_epoch

    def on_train_begin(self, logs=None):
        self.nb_epoch = self.params['epochs']
        print('Stochastic weight averaging selected for last {} epochs.'
              .format(self.nb_epoch - self.swa_epoch))

    def on_epoch_end(self, epoch, logs=None):

        if epoch == self.swa_epoch:
            self.swa_weights = self.model.get_weights()

        elif epoch > self.swa_epoch:
            for i in range(len(self.swa_weights)):
                self.swa_weights[i] = (self.swa_weights[i] *
                                       (epoch - self.swa_epoch) + self.model.get_weights()[i]) / (
                                              (epoch - self.swa_epoch) + 1)
        else:
            pass

    def on_train_end(self, logs=None):
        self.model.set_weights(self.swa_weights)
        print('Final model parameters set to stochastic weight average.')
        self.model.save_weights(self.filepath)
        print('Final stochastic averaged weights saved to file.')


def cross_validation():
    """ Subject cross-validation """

    global epochs, batch_size, n_folds, exp_path, aug_prob, swa_epochs, data_dir

    # split the subjects on n folds keeping balance
    with h5py.File(data_dir, 'r') as f:
        labels = [l for l in f['labels']]
        ids_pool = [str(i).split("'")[1] for i in f['ids']]

    ids_labels = {k: v for k, v in zip(ids_pool, labels)}
    ids_labels = pd.DataFrame([{'subject': k, 'label': v} for k, v in ids_labels.items()])
    ids_labels['label_str'] = [str(l) for l in ids_labels['label']]

    # split the subjects on n folds keeping balance
    ids = ids_labels['subject']
    skf = StratifiedKFold(n_splits=n_folds, random_state=None, shuffle=True)

    subj_folds = [(ids[test_index]) for train_index, test_index in skf.split(ids_labels['subject'],
                                                                             ids_labels['label_str']
                                                                             )]

    # true labels of each subject
    subject_labels = {ID: None for ID in list(ids)}
    for ID, label in zip(ids_labels['subject'], ids_labels['label']):
        subject_labels[ID] = label

    # to save the predictions of each subject
    subject_predictions = {ID: [] for ID in list(ids)}

    # to save the f1-score of each fold
    scores = {}
    scores_custom = {}

    for i, validation_fold in enumerate(subj_folds):
        print(f"\n\nFold {i} ------------------------------------------------- \n")

        # selector
        mask = np.isin(ids_pool, validation_fold)

        # validation
        idx_val = np.array(range(len(mask)))[mask]
        idx_train = np.array(range(len(mask)))[np.invert(mask)]

        with h5py.File(data_dir, 'r') as f:
            labels_train = [f['labels'][j] for j in idx_train]
            labels_val = [f['labels'][j] for j in idx_val]
            ids_train = [str(f['ids'][j]).split("'")[1] for j in idx_train]
            ids_val = [str(f['ids'][j]).split("'")[1] for j in idx_val]

        dg_train = DataGeneratorFromPool(data_dir, idx_train, batch_size=batch_size, augmentation=use_aug, aug_prob=aug_prob)
        dg_val = DataGeneratorFromPool(data_dir, idx_val, batch_size=batch_size, augmentation=False, shuffle=False)


        # callbacks

        # to show the f1 and fbeta during training
         # metrics_callback = MetricsCallback(validation_data=(arr_seg_validation, arr_labels_validation))

        #  NOT use early stopping if SWA
        # earlyStopping = EarlyStopping(monitor='val_categorical_accuracy', patience=10, verbose=0, mode='max')
        mcp_save = ModelCheckpoint(os.path.join(exp_path, f"model_fold_{i}.h5"),
                                   save_best_only=True, monitor='val_categorical_accuracy', mode='max')
        reduce_lr = ReduceLROnPlateau(monitor='val_categorical_accuracy', factor=0.1, patience=7, verbose=1,
                                           epsilon=1e-4,
                                           mode='max')
        ################################################################################################################
        # Model

        if model_to_train == 'model_1_1':
            # model 1
            model = stage_1_1(segment_size, n_channels, n_classes)
            # TODO settings = # load lr, momentum...
        elif model_to_train == 'model_1_2':
            # model 2
            model = ResNet1D(input_shape=(segment_size, n_channels), classes=n_classes)
        elif model_to_train == 'model_1_3':
            # model 3
            model = Inc_ResNet_LSTM_v01(segment_size, n_channels, classes=n_classes)
        elif model_to_train == 'model_1_4':
            # model 4
            model = Inc_ResNet_LSTM_v02(segment_size, n_channels, classes=n_classes)
        else:
            raise ValueError(f"Model {model_to_train} not known.")

        opt = keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)

        if use_mcc_loss:
            loss = mcc_loss
        else:
            loss = 'binary_crossentropy'

        model.compile(optimizer='adam', loss=loss, metrics=['categorical_accuracy']) # , fbeta_score])

        ################################################################################################################
        # stocastic weight averaging of the last n epochs
        swa = SWA(os.path.join(exp_path, f"model_fold_{i}_weights_SWA.h5"), epochs - swa_epochs)

        # history = model.fit(arr_seg_train_balanced, arr_labels_train_balanced, epochs=epochs, batch_size=batch_size,
        #                     verbose=2, validation_data=(arr_seg_validation, arr_labels_validation), shuffle=True,
        #                     callbacks=[mcp_save, reduce_lr, swa, metrics_callback])

        history = model.fit(dg_train, epochs=epochs, verbose=2,
                            validation_data=dg_val, shuffle=True,
                            callbacks=[mcp_save, reduce_lr])  # swa, metrics_callback])

        del model
        model = load_model(os.path.join(exp_path, f"model_fold_{i}.h5"))
    #
    #     # re-load best model weights
    #     model.load_weights(os.path.join(exp_path, f"model_fold_{i}_weights_SWA.h5"))
    #
        results = model.evaluate(dg_val, verbose=1)
        predictions = model.predict(dg_val, verbose=2)


        labels_val = []
        for b in [dg_val[i] for i in range(len(dg_val))]:
            for l in b[1]:
                labels_val.append(l)

        # print fold results
        print("Validation (loss | acc):", results)
    #
    #     f1_score, f1_score_list = multiclass_f1(labels_val, predictions, return_list=True)
    #     print("\nf1 score:", f1_score)
    #     print(f1_score_list)
    #
    #     f1_score_custom, f1_score_custom_list = custom_multiclass_f1(labels_val, predictions,
    #                                                                  return_list=True)
    #     print("\nf1 score (custom):", f1_score_custom)
    #     print(f1_score_custom_list)
    #
        # save predictions
        for ID, pred in zip(ids_val, predictions):
            subject_predictions[ID].append(pred)
    #
    #     # save f1-score
    #     scores[f"fold_{i}"] = f1_score
    #     scores_custom[f"fold_{i}"] = f1_score_custom
    #
    #     # save f1-score list (text file):
    #     with open(os.path.join(exp_path, "scores.txt"), 'a') as f:
    #         f.write(f"Fold {str(i)}:\n"
    #                 f"{str(f1_score_list)} (f1-score by class) \n"
    #                 f"{str(f1_score_custom_list)} (f1 score (custom) by class) \n")
    #
    # # Average f-1 score
    # m, s = np.mean([v for v in scores.values()]), np.std([v for v in scores.values()])
    # m_c, s_c = np.mean([v for v in scores_custom.values()]), np.std([v for v in scores_custom.values()])
    #
    # save labels (to disk)
    np.save(os.path.join(exp_path, "subject_labels.npy"), subject_labels)

    # save predictions (to disk)
    np.save(os.path.join(exp_path, "subject_predictions.npy"), subject_predictions)
    #
    # # save f1-scores (to disk)
    # np.save(os.path.join(exp_path, "scores.npy"), scores)
    # np.save(os.path.join(exp_path, "scores_custom.npy"), scores_custom)
    #
    # print("\n==========================================================\n")
    # print(f"CV f1-score: {str(m)} (+/- {str(s)}) \nCV f1-score (custom): {str(m_c)} (+/- {str(s_c)})")
    #
    # # save f1-scores (text file)
    # with open(os.path.join(exp_path, "scores.txt"), 'a') as f:
    #     f.write("\n\n ==> Score by CV:")
    #     f.write(f"\n{str(scores)} (f1-score) \n{str(scores_custom)} (f1-score (custom))")
    #     f.write("\n\n ==> Average score CV:")
    #     f.write(f"\nCV f1-score: {str(m)} (+/- {str(s)}) \nCV f1-score (custom): {str(m_c)} (+/- {str(s_c)})\n\n")

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig(os.path.join(exp_path, f"{model_to_train}_loss_plot.png"))

if __name__ == '__main__':

    # Config
    use_mcc_loss = False

    #model_to_train = sys.argv[1]
    model_to_train = 'model_1_1'  # model_1_2, model_1_3, model_1_4
    experiment_name = "v3"
    experiments_dir = "experiments/stage_1"
    epochs = 8
    swa_epochs = 3
    n_folds = 4
    segment_size = 3000
    overlap = 0.33
    batch_size = 50
    freq = 500  # Hz
    ch_spectrum = []
    max_freq_spectrum = 175
    n_channels = 12 + len(ch_spectrum)
    use_aug = False  # TODO
    aug_prob = 0.2
    exp_path = os.path.join(experiments_dir, f"{model_to_train}_{experiment_name}")

    dx_map = r'datasets/Dx_map.csv'
    df_map = pd.read_csv(dx_map)
    print(f"dx_map shape: {df_map.shape}")
    n_classes = df_map.shape[0]
    data_dir = r'datasets/dp_train_3000_0.25_001.hdf5'

    # create directory for the experiment
    if not os.path.exists(exp_path):
        os.makedirs(exp_path)
    # else:
    #     raise NameError(f"Already exist an experiment with the name '{experiment_name}'"
    #                     f" in the '{experiments_dir}' directory.")

    # save a copy of the script
    # shutil.copy(__file__, os.path.join(exp_path, ntpath.basename(__file__)))


    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # To create the data pool
    datasets = [i for i in os.listdir('datasets') if re.match(r"\d\-", i)]

    data = {}
    for subset in datasets:
        subdata = load_data(os.path.join('datasets', subset))
        subdata = preprocess_input_data(subdata, resample=freq)
        data.update(subdata)

    data = segment_all_dict_data(data, segment_size, overlap)

    arr_of_segments, arr_of_labels, arr_of_IDs = reshape_segmented_arrays(data, shuffle_IDs=True,
                                                                          # Do not shuffle the segments to keep the
                                                                          # order in time of the predictions
                                                                          shuffle_segments=False,
                                                                          segment_standardization_flag=True)
    # Encode IDs
    arr_of_IDs = [np.string_(i[0]) for i in arr_of_IDs]
    arr_of_IDs = np.array(arr_of_IDs)
    # Encode labels
    arr_of_labels = np.array([i[0]['Dx'] for i in arr_of_labels])

    arr_of_labels_all = onehot_labels(arr_of_labels, mode='all')
    arr_of_labels_scored = onehot_labels(arr_of_labels, mode='scored')
    arr_of_labels_unscored = onehot_labels(arr_of_labels, mode='unscored')

    aa = np.sum(arr_of_labels_scored, axis=1)

    bb = aa != 0

    arr_of_segments = arr_of_segments[bb]
    arr_of_labels = arr_of_labels[bb]
    arr_of_IDs = arr_of_IDs[bb]

    arr_of_labels_all= arr_of_labels_all[bb]
    arr_of_labels_scored = arr_of_labels_scored[bb]
    arr_of_labels_unscored = arr_of_labels_unscored[bb]

    # split a test set
    ids_unique = np.unique(arr_of_IDs)
    test_subjects = np.random.choice(ids_unique, 1500)

    mask = np.isin(arr_of_IDs, test_subjects).squeeze()
    idx_test = np.array(range(len(mask)))[mask]
    idx_train = np.array(range(len(mask)))[np.invert(mask)]


    # data data pool
    with h5py.File('datasets/dp_train_3000_0.33_scored.hdf5', 'w') as f:
        signals = f.create_dataset("signals", data=arr_of_segments[idx_train])
        labels_all = f.create_dataset("labels_all", data=arr_of_labels_all[idx_train])
        labels_scored = f.create_dataset("labels_scored", data=arr_of_labels_scored[idx_train])
        labels_unscored = f.create_dataset("labels_unscored", data=arr_of_labels_unscored[idx_train])
        ids = f.create_dataset("ids", data=arr_of_IDs[idx_train])

    # test data pool
    with h5py.File('datasets/dp_test_3000_0.33_scored.hdf5', 'w') as f:
        signals = f.create_dataset("signals", data=arr_of_segments[idx_test])
        labels_all = f.create_dataset("labels_all", data=arr_of_labels_all[idx_test])
        labels_scored = f.create_dataset("labels_scored", data=arr_of_labels_scored[idx_test])
        labels_unscored = f.create_dataset("labels_unscored", data=arr_of_labels_unscored[idx_test])
        ids = f.create_dataset("ids", data=arr_of_IDs[idx_test])
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

