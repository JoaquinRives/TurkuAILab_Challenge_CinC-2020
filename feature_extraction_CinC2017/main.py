"""
main.py
-------------------
This file calculates features for a test set and makes a prediction.
Implemented code assumes a single-channel lead ECG signal.
:copyright: (c) 2017 by Goodfellow Analytics
-------------------
By: Sebastian D. Goodfellow, Ph.D.
"""

# Compatibility imports
from __future__ import absolute_import, division, print_function

# 3rd party imports
import os
import sys
import pickle
import numpy as np
import pandas as pd
mingw_path = 'C:\\Program Files\\mingw-w64\\x86_64-6.2.0-posix-seh-rt_v5-rev1\\mingw64\\bin'
os.environ['PATH'] = mingw_path + ';' + os.environ['PATH']
import xgboost as xgb

# Local imports
from features.features_submit import *


"""
Get file name from input args
"""
ecg_file = sys.argv[1]


"""
Set Constants
"""
# Sampling frequency
fs = 300  # Hz

# ECG file path for validation/test data
base_path = os.path.dirname(__file__)
validation_path = os.path.abspath(os.path.join(base_path))


"""
Calculate feature vectors
"""
# Instantiate Features object
ecg_features = FeaturesSubmit(
    file_path=os.path.join(validation_path, ecg_file+'.mat'),
    file_name=ecg_file,
    fs=fs,
    feature_groups=[
        'full_waveform_statistics',
        'heart_rate_variability_statistics',
        'template_statistics'
    ]
)

# Calculate ECG features
ecg_features.calculate_features(
    filter_bandwidth=[3, 45], show=False,
    normalize=True, polarity_check=True,
    template_before=0.25, template_after=0.4
)

# Get features DataFrame
features = ecg_features.get_features()


"""
Load Training Model
"""
# Set file path
model_path = os.path.abspath(os.path.join(base_path))
model_file = 'xgb1_model_submission_13.pickle'

# Load Pickled file
with open(os.path.join(model_path, model_file), "rb") as input_file:
    model = pickle.load(input_file)


"""
Make Predictions
"""
# Create prediction DataFrame
predictions = pd.DataFrame(features['file_name'].values, columns=['file_name'])

# Add prediction column
predictions['prediction'] = model.predict(features.drop('file_name', axis=1))


"""
Save Predictions
"""
# File name
save_path = os.path.abspath(os.path.join(base_path))
save_file = 'answers.txt'

# Check for existing file
if not os.path.exists(os.path.join(save_path, save_file)):

    # Create file
    open(os.path.join(save_path, save_file), 'w').close()

# Add ECG prediction
with open(os.path.join(save_path, save_file), 'a+') as output_file:
    output_file.write(str(predictions.ix[0, 'file_name']) + ',' + str(predictions.ix[0, 'prediction']))
