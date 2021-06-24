import numpy as np
import scipy.signal as signal
from scipy.stats import zscore
# from sklearn.preprocessing import MinMaxScaler
import copy

FS = 500
lf_filter = 0.5  # Hz
hf_filter = 30  # Hz  # TODO
order_filter = 4


def butter_filter(data, lf=0.5, hf=30, fs=FS, order=4):
    wbut = [2 * lf / fs, 2 * hf / fs]
    bbut, abut = signal.butter(order, wbut, btype='bandpass')

    if type(data) == dict:
        for key in data:
            data[key] = signal.filtfilt(bbut, abut, data[key])
        return data
    else:
        return signal.filtfilt(bbut, abut, data)


def preprocess_input_data_v2(input_data, standardization_flag=False, padding_flag=False, resample=500):
    """ Pre-process the input data """

    print('processing data...')

    for subj_key in input_data.keys():
        ignore_key = ['info']
        info = input_data[subj_key]['info']
        #################################

        for lead_key in input_data[subj_key].keys() - ignore_key:
            sig = input_data[subj_key][lead_key]

            if resample and info['freq'] != resample:
                num = int((resample / info['freq']) * len(sig))

                sig = signal.resample(sig, num)

            # filter (butter bandpass)
            sig = butter_filter(sig,
                                  lf=lf_filter,
                                  hf=hf_filter,
                                  order=order_filter)
            # Standardization
            if standardization_flag:
                sig = zscore(sig)
                # TODO: try min-max scaler instead
                # sig = min_max.fit_transform(sig.reshape(sig.shape[0], 1))

            # Padding
            if padding_flag:
                # total length of the padded signal
                final_length = 72000
                sig_padded = np.zeros(final_length)
                sig_padded[-len(sig):] = sig
                input_data[subj_key][lead_key] = sig_padded.astype('float32')
            else:
                input_data[subj_key][lead_key] = sig.astype('float32')

    return input_data



def preprocess_input_data(input_data, standardization_flag=False, padding_flag=False, resample=500):
    """ Pre-process the input data """

    print('processing data...')

    for subj_key in input_data.keys():
        print(subj_key)
        ignore_key = ['info']
        info = input_data[subj_key]['info']
        #################################

        for lead_key in input_data[subj_key].keys() - ignore_key:
            sig = input_data[subj_key][lead_key]

            if resample and info['freq'] != resample:
                num = int((resample / info['freq']) * len(sig))

                sig = signal.resample(sig, num)

            # filter (butter bandpass)
            sig = butter_filter(sig,
                                  lf=lf_filter,
                                  hf=hf_filter,
                                  order=order_filter)

            # Standardization
            if standardization_flag:
                sig = zscore(sig)
                # TODO: try min-max scaler instead
                # sig = min_max.fit_transform(sig.reshape(sig.shape[0], 1))

            # Padding
            if padding_flag:
                # total length of the padded signal
                final_length = 72000
                sig_padded = np.zeros(final_length)
                sig_padded[-len(sig):] = sig
                input_data[subj_key][lead_key] = sig_padded.astype('float32')
            else:
                input_data[subj_key][lead_key] = sig.astype('float32')

    return input_data
