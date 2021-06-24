import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.metrics import multilabel_confusion_matrix
from shutil import copy2
import os
import re
import tensorflow.keras as keras
from keras.engine.saving import load_model

competition_labels = ["AF", "I-AVB", "LBBB", "Normal", "PAC", "PVC", "RBBB", "STD", "STE"]


# TODO: use this to encode the labels??
def one_hot(labels, n_classes=9):
    """ One_Hot """
    expansion = np.eye(n_classes)
    y = expansion[:, labels].T

    return y


def encode_labels(labels):
    """ One-hot label encoding """
    competition_labels = ('AF', 'I-AVB', 'LBBB', 'Normal', 'PAC', 'PVC', 'RBBB', 'STD', 'STE')
    labels_encoded = np.zeros(shape=(labels.shape[0], 9), dtype=int)
    for i in range(labels.shape[0]):
        for j in range(9):
            if competition_labels[j] in labels[i]:
                labels_encoded[i][j] = 1
            else:
                labels_encoded[i][j] = 0
    return labels_encoded


dx_mapping_unscored_path = 'datasets/dx_mapping_unscored.csv'
dx_mapping_scored_path = 'datasets/dx_mapping_scored.csv'


def onehot_labels(data_labels, mode='all'):
    """ One-hot label encoding """

    # load Dx map
    dx_mapping_unscored = pd.read_csv(dx_mapping_unscored_path)
    dx_mapping_scored = pd.read_csv(dx_mapping_scored_path)

    if mode == 'all':
        dx_map = pd.concat([dx_mapping_scored, dx_mapping_unscored])
    elif mode == 'scored':
        dx_map = dx_mapping_scored
    elif mode == 'unscored':
        dx_map = dx_mapping_unscored

    # dict_map = dict(zip(df_map['SNOMED code'], df_map['Abbreviation']))

    labels = np.array([v for v in dx_map['SNOMED CT Code'].values])

    labels_encoded = np.zeros(shape=(len(data_labels), len(labels)), dtype=int)
    for i in range(len(data_labels)):
        for j in range(len(labels)):
            # print(labels[j])
            # print(data_labels[i])
            if str(labels[j]) in str(data_labels[i]):
                labels_encoded[i][j] = 1
            else:
                labels_encoded[i][j] = 0
    return labels_encoded


def multiclass_f1(true, pred, return_list=False):
    """ Calculates the f1-score for each class and returns the average f1-score """

    y_pred = pred.copy()
    y_true = true.copy()

    y_pred = np.where(y_pred > 0.5, 1, 0)

    conf_matrix = multilabel_confusion_matrix(y_true, y_pred)
    f1_list = []
    for m in conf_matrix:
        f1_list.append((2 * m[1, 1]) / ((2 * m[1, 1]) + m[0, 1] + m[1, 0]))

    if return_list:
        return np.mean(f1_list), f1_list

    return np.mean(f1_list)


def custom_multiclass_f1(true, pred, return_list=False):
    """ Calculates the global f1-score following the PhysioNet/CinC 2020 challenge criteria """

    # accuracies of each individual prediction
    y_pred = pred.copy()
    y_true = true.copy()

    y_pred = np.where(y_pred > 0.5, 1, 0)

    # explicitly
    from sklearn.metrics import multilabel_confusion_matrix
    conf_matrix = multilabel_confusion_matrix(y_true, y_pred)
    f1_list = []
    for m in conf_matrix:
        f1_list.append((5 * m[1, 1]) / ((5 * m[1, 1]) + m[0, 1] + (4*m[1, 0])))

    if return_list:
        return np.mean(f1_list), f1_list

    return np.mean(f1_list)


# draft
def split_data_train_test():
    """ Split the data files on train/test sets """

    data_dir = 'data/Training_WFDB'
    train_dir = 'data/train'
    test_dir = 'data/test'
    split = 0.2

    if not os.path.exists(train_dir):
        os.mkdir(train_dir)
    if not os.path.exists(test_dir):
        os.mkdir(test_dir)

    subjects = []
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith(".mat"):

                # subject id
                subj_id = re.match(r"^(\w+).mat", file).group(1)
                subjects.append(subj_id)

    np.random.shuffle(subjects)

    test_size = int(len(subjects) * split)

    test_subjects = [subjects[i] for i in np.random.choice(len(subjects), test_size)]
    train_subjects = [subj for subj in subjects if subj not in test_subjects]

    for subj in test_subjects:
        copy2(os.path.join(data_dir, f"{subj}.mat"), test_dir)
        copy2(os.path.join(data_dir, f"{subj}.hea"), test_dir)

    for subj in train_subjects:
        copy2(os.path.join(data_dir, f"{subj}.mat"), train_dir)
        copy2(os.path.join(data_dir, f"{subj}.hea"), train_dir)


# draft
def split_train_validation_part_2(subject_predictions, subject_labels,n_outputs=111, n_variables=18, split=0.33):
    """ Splits train/validation sets for the model_1_part_2"""

    n_timesteps = [len(v) for v in subject_predictions.values()][0]
    # n_variables = 18
    # n_outputs = n_outputs

    assert len(subject_labels) == len(subject_predictions), "Labels and predictions have different shapes"

    subjects = [k for k in subject_labels.keys()]
    np.random.shuffle(subjects)
    index_split = int(len(subjects) * split)
    subjects_train = subjects[index_split:]
    subjects_validation = subjects[:index_split]

    # keep the pairwise order of subjects between labels-predictions

    # training
    X_train = np.zeros((len(subjects_train), n_timesteps, n_variables))
    y_train = np.zeros((len(subjects_train), n_outputs))
    for i, subj in enumerate(subjects_train):
        X_train[i, :, :] = subject_predictions[subj]
        y_train[i, :] = subject_labels[subj]

    # validation
    X_validation = np.zeros((len(subjects_validation), n_timesteps, n_variables))
    y_validation = np.zeros((len(subjects_validation), n_outputs))
    for i, subj in enumerate(subjects_validation):
        X_validation[i, :, :] = subject_predictions[subj]
        y_validation[i, :] = subject_labels[subj]

    return X_train, y_train, X_validation, y_validation


def split_train_test(arr_of_segments, arr_of_labels, arr_of_IDs, split=0.33):
    """ Splits the subjects on train and test sets """

    ids_unique = np.unique(arr_of_IDs)
    test_size = int(ids_unique.shape[0] * split)
    test_ids = ids_unique[np.random.choice(len(ids_unique), test_size)]

    # selector
    selector = np.isin(arr_of_IDs.squeeze(), test_ids)

    # test
    arr_seg_test = arr_of_segments[selector]
    arr_labels_test = arr_of_labels[selector]
    arr_IDs_test = arr_of_IDs[selector]

    # train
    arr_seg_train = arr_of_segments[np.invert(selector)]
    arr_labels_train = arr_of_labels[np.invert(selector)]
    arr_IDs_train = arr_of_IDs[np.invert(selector)]

    return arr_seg_train, arr_labels_train, arr_IDs_train, arr_seg_test, arr_labels_test, arr_IDs_test


def load_and_assemble():
    pass


def convert_model_to_json(model, use_tensorflow=False):
    """
    Converts a model that was saved on a single HDF5 file to JSON
    saving the weights in a independent HDF5 file.

    model.h5  -->  model.json + model_w.h5
    """

    # get the complete filename without the extension
    filename = os.path.splitext(model)[0]

    if use_tensorflow:
        model = keras.models.load_model(model)

        # serialize model to JSON
        model_json = model.to_json()
        with open(f"{filename}.json", "w") as json_file:
            json_file.write(model_json)

        # serialize weights to HDF5
        model.save_weights(f"{filename}_w.h5")

        print(f"Created {filename}_w.h5 & {filename}.json files.")

    else:
        model = load_model(model)

        # serialize model to JSON
        model_json = model.to_json()
        with open(f"{filename}.json", "w") as json_file:
            json_file.write(model_json)

        # serialize weights to HDF5
        model.save_weights(f"{filename}_w.h5")

        print(f"Created {filename}_w.h5 & {filename}.json files.")
        



# convert_model_to_json('submission_5/models/stage_1_1.h5', use_tensorflow=False)
# convert_model_to_json('submission_5/models/stage_1_2.h5', use_tensorflow=True)
# convert_model_to_json('submission_5/models/stage_1_3.h5', use_tensorflow=True)
# convert_model_to_json('submission_5/models/stage_1_4.h5', use_tensorflow=True)






















