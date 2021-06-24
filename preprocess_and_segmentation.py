from multiprocessing import cpu_count, Pool
from multiprocessing.pool import ThreadPool
from timeit import default_timer as timer
import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.stats import zscore
import os
import re
import matplotlib.pyplot as plt
from tqdm import tqdm

from preprocessor import preprocess_input_data
from random import shuffle
from utils import encode_labels

fs = 500  # Hz
lead_labels = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']


def load_with_codes(path_data, path_map):
    competition_labels = ["AF", "I-AVB", "LBBB", "Normal", "PAC", "PVC", "RBBB", "STD", "STE"]

    # load Dx map
    df_map = pd.read_csv(path_map)
    dict_map = dict(zip(df_map['SNOMED code'], df_map['Abbreviation']))

    # load data
    data = load_data(path_data)

    all_codes = []

    subject_codes = {}
    for k, v in data.items():
        subject_codes[k] = data[k]['info']['Dx']

    # translate the codes to labels
    subject_labels = {}
    for subj, codes in subject_codes.items():

        subject_labels[subj] = []
        for c in codes.split(","):
            all_codes.append(c)
            try:
                c = int(c)
            except:
                print('exception with code:', c)
                c = None

            if c in dict_map.keys():
                subject_labels[subj].append(dict_map[c])

    for subj, labels in subject_labels.items():
        known_labels = []
        for label in labels:
            known_labels.append(label)

        data[subj]['info']['Dx'] = ",".join(known_labels) if len(known_labels) > 1 else known_labels[0]

    return data


def onehot_labels(data_labels, path_map):
    """ One-hot label encoding """

    # load Dx map
    df_map = pd.read_csv(path_map)
    dict_map = dict(zip(df_map['SNOMED code'], df_map['Abbreviation']))
    labels = np.array([v for v in dict_map.values()])

    labels_encoded = np.zeros(shape=(len(data_labels), len(labels)), dtype=int)
    for i in range(len(data_labels)):
        for j in range(len(labels)):
            if labels[j] in data_labels[i]:
                labels_encoded[i][j] = 1
            else:
                labels_encoded[i][j] = 0
    return labels_encoded


def load_data_v2(files):
    """ Load the subject signals (".mat" files) and info (".hea" files) into a dictionary """
    if not isinstance(files, (list, np.ndarray)):
        files = list(files)

    print(f"loading data...")
    # signals (".mat" files)
    data = {}
    for file in files:
        if file.endswith(".mat"):

            # subject id
            subj_id = re.search(r"(\w+).mat$", file).group(1)
            # create dictionary for the subject
            data[subj_id] = {}

            # load signals from the mat file
            signals = loadmat(file)['val']

            # add the signals to the data dictionary
            for i in range(signals.shape[0]):
                data[subj_id][lead_labels[i]] = signals[i, :]

            # labels and info
            labels_file = re.sub("mat$", "hea", file)

            with open(labels_file, 'r') as f:
                text = f.read()

            # dictionary to save all the info about the subject
            info = {}

            # Frequency
            freq = text.split("\n")[0].split(" ")[2]
            info['freq'] = int(freq)
            # Age
            age = re.search(r"#Age: (.+?)\n", text).group(1)
            info['Age'] = age
            # Sex
            sex = re.search(r"#Sex: (.+?)\n", text).group(1)
            info['Sex'] = sex
            # Dx
            dx = re.search(r"#Dx: (.+?)\n", text).group(1)
            info['Dx'] = dx
            # Rx
            rx = re.search(r"#Rx: (.+?)\n", text).group(1)
            info['Rx'] = rx
            # Hx
            hx = re.search(r"#Hx: (.+?)\n", text).group(1)
            info['Hx'] = hx
            # Sx
            sx = re.search(r"#Sx: (.+?)\n", text).group(1)
            info['Sx'] = sx

            # add the info to the data dictionary
            data[subj_id]['info'] = info

    return data



def load_data(data_dir):
    """ Load the subject signals (".mat" files) and info (".hea" files)
        into a dictionary """
    print(f"loading {data_dir.split('/')[-1]} ...")
    # signals (".mat" files)
    data = {}
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith(".mat"):

                # subject id
                subj_id = re.match(r"^(\w+).mat", file).group(1)
                # create dictionary for the subject
                data[subj_id] = {}

                # load signals from the mat file
                signals = loadmat(os.path.join(root, file))['val']

                # add the signals to the data dictionary
                for i in range(signals.shape[0]):
                    data[subj_id][lead_labels[i]] = signals[i, :]

    # labels and extra info (".hea" files)
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith(".hea"):

                # subject id
                subj_id = re.match(r"^(\w+).hea", file).group(1)

                with open(os.path.join(data_dir, file), 'r') as f:
                    text = f.read()

                # dictionary to save all the info about the subject
                info = {}

                # Frequency
                freq = text.split("\n")[0].split(" ")[2]
                info['freq'] = int(freq)
                # Age
                age = re.search(r"#Age: (.+?)\n", text).group(1)
                info['Age'] = age
                # Sex
                sex = re.search(r"#Sex: (.+?)\n", text).group(1)
                info['Sex'] = sex
                # Dx
                dx = re.search(r"#Dx: (.+?)\n", text).group(1)
                info['Dx'] = dx
                # Rx
                rx = re.search(r"#Rx: (.+?)\n", text).group(1)
                info['Rx'] = rx
                # Hx
                hx = re.search(r"#Hx: (.+?)\n", text).group(1)
                info['Hx'] = hx
                # Sx
                sx = re.search(r"#Sx: (.+?)\n", text).group(1)
                info['Sx'] = sx

                # add the info to the data dictionary
                data[subj_id]['info'] = info

    return data


def load_and_translate(path_data, path_map, return_codes=False):
    competition_labels = ["AF", "I-AVB", "LBBB", "Normal", "PAC", "PVC", "RBBB", "STD", "STE"]

    # load Dx map
    df_map = pd.read_csv(path_map)
    dict_map = dict(zip(df_map['SNOMED code'], df_map['Abbreviation']))

    # load data
    data = load_data(path_data)

    all_codes = []

    subject_codes = {}
    for k, v in data.items():
        subject_codes[k] = data[k]['info']['Dx']

    # translate the codes to labels
    subject_labels = {}
    for subj, codes in subject_codes.items():

        subject_labels[subj] = []
        for c in codes.split(","):
            all_codes.append(c)
            try:
                c = int(c)
            except:
                c = 0

            if c in dict_map.keys():
                subject_labels[subj].append(dict_map[c])

    for subj, labels in subject_labels.items():
        known_labels = []
        for label in labels:
            if label in competition_labels:
                known_labels.append(label)

        if len(known_labels) == 0:
            known_labels.append('other')

        data[subj]['info']['Dx'] = ",".join(known_labels) if len(known_labels) > 1 else known_labels[0]

    if return_codes:
        return data, all_codes

    return data


def window_stack(a, win_width, overlap):
    if overlap == 0:
        return np.vstack([a[i:(i + win_width)] for i in np.arange(0, (len(a) - win_width + 1), win_width).astype(int)])
    else:
        stride = 1 - overlap
        return np.vstack(
            [a[i:(i + win_width)] for i in np.arange(0, (len(a) - win_width + 1), win_width * stride).astype(int)])


def segmenting_data(dict_of_data, seg_width=2500, overlap_perc=0.5):

    segmented_signals = {}

    info = dict_of_data['info']
    del dict_of_data['info']

    for key in dict_of_data.keys():
        segmented_signals[key] = window_stack(dict_of_data[key], seg_width, overlap=overlap_perc)

    # add the info/label back to the dict
    segmented_signals['info'] = np.repeat(info, len(segmented_signals[lead_labels[1]]))

    return segmented_signals


def segment_all_dict_data(data_dict, seg_width, overlap_perc, parallelize=False, parallel_mode='multiprocessing'):
    """
    This function abstracts all the segmenting and reshaping tasks that are performed
    by the segmenting_data and reshape_segmented_arrays functions
    """
    segmented_dict_of_data = {}
    # segment the signals into smaller pieces and put all reshaped data into a mother dictionary
    for key in data_dict.keys():
        if data_dict[key][lead_labels[0]].shape[0] > seg_width:
            segmented_dict_of_data[key] = segmenting_data(data_dict[key], seg_width, overlap_perc)

    return segmented_dict_of_data


def reshape_segmented_arrays(input_dict, shuffle_IDs=True, shuffle_segments=False,
                             segment_standardization_flag=True):
    list_of_swapped_stack = []
    list_of_ID_arrays = []
    list_of_label_arrays = []

    print("reshaping segmented arrays...")
    for key in input_dict.keys():

        # list of the matrices of segmented data in 12 channel
        dict_data = input_dict[key]
        ID = key

        data_list = [v for k, v in dict_data.items() if k != 'info']

        # stacking all the data into one array
        data_stacked_array = np.stack(data_list, axis=0)

        if shuffle_segments:
            random_indices = np.random.randint(0, data_stacked_array.shape[1], data_stacked_array.shape[1])

            data_stacked_array = data_stacked_array[:, random_indices, :]

        # swap the axes
        swaped_stack = np.swapaxes(np.swapaxes(data_stacked_array, 0, 2), 0, 1).astype('float32')
        ID_for_segments = np.repeat(ID, swaped_stack.shape[0])
        label_for_segments = dict_data['info']

        # append to their corresponding lists
        list_of_swapped_stack.append(swaped_stack)
        list_of_ID_arrays.append(ID_for_segments)
        list_of_label_arrays.append(label_for_segments)

    # shuffle the order of subjects in every list
    if shuffle_IDs:
        # generate random indices
        perm = list(range(len(list_of_ID_arrays)))
        shuffle(perm)

        # rearrange the lists
        list_of_swapped_stack = [list_of_swapped_stack[index] for index in perm]
        list_of_ID_arrays = [list_of_ID_arrays[index] for index in perm]
        list_of_label_arrays = [list_of_label_arrays[index] for index in perm]

    # transform the lists into numpy arrays by stacking along first axis
    array_of_segments = np.concatenate(list_of_swapped_stack, axis=0)
    array_of_IDs = np.concatenate(list_of_ID_arrays, axis=0)[:, np.newaxis]
    array_of_labels = np.concatenate(list_of_label_arrays, axis=0)[:, np.newaxis]

    # normalize every segemnt
    if segment_standardization_flag:
        def segment_standardization(my_segment):
            from sklearn.preprocessing import StandardScaler

            s = StandardScaler()

            # fit on training data
            normalized_segment = s.fit_transform(my_segment[:, np.newaxis])

            return normalized_segment.ravel()

        print("Standardizing...")
        start = timer()
        array_of_segments = np.apply_along_axis(segment_standardization, 1, array_of_segments)
        end = timer()
        print(end-start)

    # print the shapes
    print('shape of the array of segments is :', array_of_segments.shape)
    print('shape of the array of IDs is :', array_of_IDs.shape)
    print('shape of the array of labels is :', array_of_labels.shape)

    # convert to float32
    array_of_segments = array_of_segments.astype('float32')

    return array_of_segments, array_of_labels, array_of_IDs


def plot_segment(inputarray, seg_indx, axis1, axis2):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(inputarray[seg_indx, :, axis1:axis2])
    plt.show()
    return fig


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


def convert_to_arrays(input_dict, shuffle_IDs=True):
    """ Transform the dictionary of data to arrays """
    list_of_swapped_stack = []
    list_of_IDs = []
    list_of_labels = []

    for key in input_dict.keys():
        dict_data = input_dict[key]
        data_list = [v for k, v in dict_data.items() if k != 'info']

        # stacking all the data into one array
        data_stacked_array = np.stack(data_list, axis=0)

        # swap the axes
        swaped_stack = np.swapaxes(np.swapaxes(data_stacked_array, 0, 2), 0, 1)

        label_for_segments = dict_data['info']

        # append to their corresponding lists
        list_of_swapped_stack.append(swaped_stack)
        list_of_IDs.append(key)  # ID of the subject
        list_of_labels.append(label_for_segments)

    # shuffle the order of subjects in every list
    if shuffle_IDs:
        # generate random indices
        perm = list(range(len(list_of_IDs)))
        shuffle(perm)

        # rearrange the lists
        list_of_swapped_stack = [list_of_swapped_stack[index] for index in perm]
        list_of_IDs = [list_of_IDs[index] for index in perm]
        list_of_labels = [list_of_labels[index] for index in perm]

    # transform the lists into numpy arrays
    array_of_signals = np.array(list_of_swapped_stack)
    array_of_IDs = np.array(list_of_IDs)
    array_of_labels = np.array(list_of_labels)

    # convert to float32
    array_of_signals = array_of_signals.astype('float32')

    return array_of_signals, array_of_labels, array_of_IDs


def create_partition(input_dict, shuffle_IDs=True, segment_standardization=True):
    """ Transform the dictionary of data to arrays """
    list_of_swapped_stack = []
    list_of_IDs = []
    list_of_labels = []

    for key in input_dict.keys():

        dict_data = input_dict[key]
        data_list = [v for k, v in dict_data.items() if k != 'info']

        # stacking all the data into one array
        data_stacked_array = np.stack(data_list, axis=0)

        # swap the axes
        swaped_stack = np.swapaxes(np.swapaxes(data_stacked_array, 0, 2), 0, 1)

        if segment_standardization:
            for i in range(swaped_stack.shape[0]):
                for j in range(swaped_stack.shape[2]):
                    swaped_stack[i, :, j] = zscore(swaped_stack[i, :, j])

        label_for_segments = dict_data['info']

        # append to their corresponding lists
        list_of_swapped_stack.append(swaped_stack)
        list_of_IDs.append(key)  # ID of the subject
        list_of_labels.append(label_for_segments)

    # shuffle the order of subjects in every list
    if shuffle_IDs:
        # generate random indices
        perm = list(range(len(list_of_IDs)))
        shuffle(perm)

        # rearrange the lists
        list_of_swapped_stack = [list_of_swapped_stack[index] for index in perm]
        list_of_IDs = [list_of_IDs[index] for index in perm]
        list_of_labels = [list_of_labels[index] for index in perm]

    # Encode labels
    list_of_labels_encoded = np.array([i[0]['Dx'] for i in list_of_labels])
    list_of_labels_encoded = encode_labels(list_of_labels_encoded)

    # Dictionary IDs:labels
    dict_labels = {k: v for k, v in zip(list_of_IDs, list_of_labels_encoded)}

    # split_train_test (the subjects are already shuffled)
    split = 0.25
    index = int(len(list_of_swapped_stack) * split)
    list_signals_test, list_IDs_test = list_of_swapped_stack[:index], list_of_IDs[:index]
    list_signals_train, list_IDs_train = list_of_swapped_stack[index:], list_of_IDs[index:]

    partition = {
        'train': {k: v for k, v in zip(list_IDs_train, list_signals_train)},
        'validation': {k: v for k, v in zip(list_IDs_test, list_signals_test)}
    }

    return partition, dict_labels


def dict_to_array(input_dict, shuffle_IDs=True):
    list_of_swapped_stack = []
    list_of_IDs = []
    list_of_labels = []

    for key in input_dict.keys():

        # list of the matrices of segmented data in 12 channel
        dict_data = input_dict[key]
        ID = key

        data_list = [v for k, v in dict_data.items() if k != 'info']

        # stacking all the data into one array
        data_stacked_array = np.stack(data_list, axis=0)

        # swap the axes# TODO I have changed this
        swaped_stack = np.swapaxes(np.swapaxes(data_stacked_array, 0, 2), 0, 1)
        ID_for_segments = ID
        label_for_segments = dict_data['info']

        # append to their corresponding lists
        list_of_swapped_stack.append(swaped_stack)
        list_of_IDs.append(ID_for_segments)
        list_of_labels.append(label_for_segments)

    # shuffle the order of subjects in every list
    if shuffle_IDs:
        # generate random indices
        perm = list(range(len(list_of_IDs)))
        shuffle(perm)

        # rearrange the lists
        list_of_swapped_stack = [list_of_swapped_stack[index] for index in perm]
        list_of_ID_arrays = [list_of_IDs[index] for index in perm]
        list_of_label_arrays = [list_of_labels[index] for index in perm]

    # transform the lists into numpy arrays by stacking along first axis
    array_of_signals = np.concatenate(np.array(list_of_swapped_stack)[:, np.newaxis], axis=0)
    array_of_IDs = np.array(list_of_IDs)[:, np.newaxis]
    array_of_labels = np.array(list_of_labels)[:, np.newaxis]

    # print the shapes
    print('shape of the array of IDs is :', array_of_IDs.shape)
    print('shape of the array of labels is :', array_of_labels.shape)

    # convert to float32
    array_of_segments = array_of_signals.astype('float32')

    return array_of_segments, array_of_labels, array_of_IDs


if __name__ == '__main__':
    data_dir = 'sample_of_data/Training_WFDB'

    # load data
    data = load_data(data_dir)

    # pre-process signals
    data = preprocess_input_data(data)

    # segment signal
    data = segment_all_dict_data(data, 500, 0.5)

    # reshape to array
    arr_of_segments, arr_of_labels, arr_of_IDs = reshape_segmented_arrays(data,
                                                                          shuffle_IDs=True,
                                                                          shuffle_segments=True,
                                                                          segment_standardization_flag=True
                                                                          )

    # Check labels
    from collections import Counter

    labels = [i[0]['Dx'] for i in arr_of_labels]
    label_count = Counter(labels)
    print(label_count)

    # plt.bar(label_count.keys(), label_count.values())
    # plt.show()

    arr_seg_train, arr_labels_train, arr_IDs_train, \
    arr_seg_test, arr_labels_test, arr_IDs_test = split_train_test(arr_of_segments, arr_of_labels, arr_of_IDs,
                                                                   split=0.33)

