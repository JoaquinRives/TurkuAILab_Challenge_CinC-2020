import h5py
import numpy as np
import tensorflow.keras as keras
from old.ekg_data_augmenter import augment_signals
from preprocess_and_segmentation import load_data_v2, segment_all_dict_data, reshape_segmented_arrays
from preprocessor import preprocess_input_data_v2


class DataGeneratorFromPool(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, data_pool, indexes=None, batch_size=32, augmentation=False, aug_prob=0.25,
                 shuffle=True):

        with h5py.File(data_pool, 'r') as f:
            self.dim = f['signals'][0].shape
            self.n_classes = f['labels'][0].shape[-1]
            if not indexes:
                indexes = list(range(len(f['labels'])))
                print(indexes, "<-- indexes length")

        self.data = data_pool
        self.indexes = indexes
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()
        self.augmentation = augmentation
        self.aug_prob = aug_prob

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.indexes) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Generate data
        X, y = self.__data_generation(indexes)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, indexes):
        'Generates data containing batch_size samples'
        # Initialization
        X = np.empty((self.batch_size, *self.dim))
        y = np.empty((self.batch_size, self.n_classes), dtype=int)

        # Generate data
        with h5py.File(self.data, 'r') as f:
            for i, index in enumerate(indexes):
                signals = f['signals'][index]

                # Data augmentations
                if self.augmentation:
                    chance = np.random.rand()
                    if chance < self.aug_prob:
                        signals = augment_signals(signals[None,])

                X[i,] = signals
                y[i,] = f['labels'][index]

        return X, y


class DataGeneratorV2(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, data, length, overlap, hz, batch_size=32, n_classes=111, augmentation=False, aug_prob=0.25,
                 shuffle=True):
        'Initialization'
        self.length = length
        self.overlap = overlap
        self.hz = hz
        self.batch_size = batch_size
        self.data = data
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()
        self.augmentation = augmentation
        self.aug_prob = aug_prob

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.data) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Generate data
        X, y = self.__data_generation(indexes)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.data))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, indexes):
        'Generates data containing batch_size samples'
        # Initialization
        X = np.empty((self.batch_size, *self.dim))
        y = np.empty((self.batch_size, self.n_classes), dtype=int)


        # Generate data
        for i, index in enumerate(indexes):

            ####
            data = load_data_v2(self.data[index])
            data = preprocess_input_data_v2(data, resample=self.hz)
            data = segment_all_dict_data(data, self.length, self.overlap)
            arr_of_segments, arr_of_labels, arr_of_IDs = reshape_segmented_arrays(data,
                                                                                  shuffle_IDs=True,
                                                                                  # Do not shuffle the segments to keep
                                                                                  # the order in time of the predictions
                                                                                  shuffle_segments=False,
                                                                                  segment_standardization_flag=True)
            ####

            signals = arr_of_segments
            labels = arr_of_labels

            # Data augmentations
            if self.augmentation:
                chance = np.random.rand()
                if chance < self.aug_prob:
                    signals = augment_signals(signals[None,])

            X[i,] = signals
            y[i,] = self.labels_array[index]

        return X, y


class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, signal_array, labels_array, batch_size=32, n_classes=9, augmentation=False, aug_prob=0.25,
                 shuffle=True):
        'Initialization'
        self.dim = signal_array.shape[1:]
        self.batch_size = batch_size
        self.labels_array = labels_array
        self.signal_array = signal_array
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()
        self.augmentation = augmentation
        self.aug_prob = aug_prob

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.signal_array) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Generate data
        X, y = self.__data_generation(indexes)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.signal_array))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, indexes):
        'Generates data containing batch_size samples'
        # Initialization
        X = np.empty((self.batch_size, *self.dim))
        y = np.empty((self.batch_size, self.n_classes), dtype=int)

        # Generate data
        for i, index in enumerate(indexes):

            signals = self.signal_array[index]

            # Data augmentations
            if self.augmentation:
                chance = np.random.rand()
                if chance < self.aug_prob:
                    signals = augment_signals(signals[None,])

            X[i,] = signals
            y[i,] = self.labels_array[index]

        return X, y