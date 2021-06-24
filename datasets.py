"""
1- CPSC2018 training set, 6,877 recordings

2 - China 12-Lead ECG Challenge Database (unused CPSC2018 data), 3,453 recordings

5 - PTB-XL electrocardiography Database, 21,837 recordings:
 length = 5000
 freq = 500hz

 6 - Georgia 12-Lead ECG Challenge Database, 10,344 recordings:
    length = 5000 (and a very few 2500)
    freq = 500 hz

3- St Petersburg INCART 12-lead Arrhythmia Database, 74 recordings
    length = 2x(462600)
    freq = 257 hz

4 - PTB Diagnostic ECG Database, 516 recordings:
    length = 16000-60000 (32000-12000)
    freq = 1000 hz
"""
import copy
import os
import re
from timeit import default_timer as timer
import random
from multiprocessing import cpu_count, Pool
from multiprocessing.pool import ThreadPool

import numpy as np
from preprocess_and_segmentation import load_data, segment_all_dict_data, reshape_segmented_arrays, lead_labels
from preprocessor import preprocess_input_data
import pandas as pd

data_dir = r'datasets/'
map_file = r'datasets/Dx_map.csv'
segment_size = 2500
overlap = 0.5
freq = 500  # Hz

# test_subjects = np.load('dataset/test_keys.npy', allow_pickle=True)
datasets = [i for i in os.listdir(data_dir) if re.match(r"\d\-", i)]


data = {}
for subset in datasets:
    subdata = load_data(os.path.join(data_dir, subset))
    subdata = preprocess_input_data(subdata, resample=freq)
    data.update(subdata)
#
# keys = list(data.keys())
#
# sample = {}
# for k in keys[:5000]:
#     sample.update({k: copy.deepcopy(data[k])})
#
# # np.save('datasets/sample.npy', sample)
# np.save('datasets/sample_5000.npy', sample)

# sample = np.load('datasets/sample.npy', allow_pickle=True).item()
# sample = np.load('datasets/sample_5000.npy', allow_pickle=True).item()

data = segment_all_dict_data(data, segment_size, 0.5, parallelize=False, parallel_mode='threading')

arr_of_segments, arr_of_labels, arr_of_IDs = reshape_segmented_arrays(data, shuffle_IDs=True,
                                                                          # Do not shuffle the segments to keep the
                                                                          # order in time of the predictions
                                                                          shuffle_segments=False,
                                                                          segment_standardization_flag=True)

