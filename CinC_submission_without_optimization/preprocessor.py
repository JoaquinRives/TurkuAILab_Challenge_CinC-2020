from scipy.io import loadmat
import numpy as np
import os
from keras.models import load_model
import numpy as np
import scipy.signal as signal
from scipy.stats import zscore

# TODO
FS = 500
lf_filter = 0.5  # Hz
hf_filter = 20  # Hz  # TODO
order_filter = 4
segment_size = 2500  # TODO
overlap = 0.5


def butter_filter(data, lf=0.5, hf=20, fs=FS, order=4):
    wbut = [2 * lf / fs, 2 * hf / fs]
    bbut, abut = signal.butter(order, wbut, btype='bandpass')

    if type(data) == dict:
        for key in data:
            data[key] = signal.filtfilt(bbut, abut, data[key])
        return data
    else:
        return signal.filtfilt(bbut, abut, data)


def filter_signals(data):

    for i in range(len(data)):
        # filter (butter bandpass)
        data[i, :] = butter_filter(data[i, :], lf=lf_filter, hf=hf_filter, order=order_filter)

    return data


def window_stack(a, win_width, overlap):
    if overlap == 0:
        return np.vstack([a[i:(i + win_width)] for i in np.arange(0, (len(a) - win_width + 1), win_width).astype(int)])
    else:
        stride = 1 - overlap
        return np.vstack(
            [a[i:(i + win_width)] for i in np.arange(0, (len(a) - win_width + 1), win_width * stride).astype(int)])


def segment_signals(data, standardise=True):

    segmented_signals = []
    for i in range(len(data)):
        segments = window_stack(data[i, :], segment_size, overlap=overlap)

        # Standardization of segments
        if standardise:
            segments = zscore(segments, axis=1)

        segmented_signals.append(np.expand_dims(segments, axis=0))

    stack_arr = np.vstack(segmented_signals)

    return np.swapaxes(np.swapaxes(stack_arr, 0, 2), 0, 1)


def preprocess_input(signals):
    signals = filter_signals(signals)
    signals = segment_signals(signals)

    return signals


