#  _____  ______ _   _ 
# |  __ \|  ____| \ | |
# | |__) | |__  |  \| |
# |  ___/|  __| | . ` |
# | |    | |    | |\  |
# |_|    |_|    |_| \_|
#  ________   __          __  __ _____  _      ______
# |  ____\ \ / /    /\   |  \/  |  __ \| |    |  ____|
# | |__   \ V /    /  \  | \  / | |__) | |    | |__
# |  __|   > <    / /\ \ | |\/| |  ___/| |    |  __|
# | |____ / . \  / ____ \| |  | | |    | |____| |____
# |______/_/ \_\/_/    \_\_|  |_|_|    |______|______|

# EnergyFlow - Python package for high-energy particle physics.
# Copyright (C) 2017-2020 Patrick T. Komiske III and Eric Metodiev
# modified to experiment with smaller dtypes

# standard library imports
from __future__ import absolute_import, division, print_function
#import os
#os.environ['TF_DISABLE_MLIR_GPU_OPS'] = '1'  # Fallback to CPU
# standard numerical library imports

# Data I/O and numerical imports
#import h5py
import numpy as np

# ML imports
import tensorflow as tf

from tensorflow.data import Dataset
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split

# energyflow is not available by default
import energyflow as ef

# to fix keras version problem with energyflow
# via https://stackoverflow.com/questions/76650327/how-to-fix-cannot-import-name-version-from-tensorflow-keras
# from keras import __version__
# tf.keras.__version__ = __version__

from energyflow.archs.efn import PFN
from energyflow.datasets import qg_jets
from energyflow.utils import data_split, remap_pids, to_categorical

# Plotting imports
import matplotlib.pyplot as plt

print("pfn_example.py\tWelcome!")

################################### SETTINGS ###################################
# the commented values correspond to those in 1810.05165
###############################################################################

# data controls, can go up to 2000000 for full dataset
train, val, test = 75000, 10000, 15000
#train, val, test = 1500000, 250000, 250000
use_pids = False
TARGET_DTYPE = np.float32

# network architecture parameters
Phi_sizes, F_sizes = (100, 100, 128), (100, 100, 100)
# Phi_sizes, F_sizes = (100, 100, 256), (100, 100, 100)

# network training parameters
num_epoch = 500
batch_size = 500

from keras.callbacks import EarlyStopping,ModelCheckpoint
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=15)
mc = ModelCheckpoint('best_pfn.keras', monitor='val_acc', mode='max', verbose=1, save_best_only=True)

################################################################################

print('Loading the dataset ...')

# load data
X, y = qg_jets.load(2000000, generator='pythia', pad=True, cache_dir='/Users/nicholasdilullo/Desktop/research/LeBlancLab/pfn_work/efcache')
Y = to_categorical(y, num_classes=2)

X_32 = X.astype(TARGET_DTYPE, copy=True)

Y_32 = Y.astype(TARGET_DTYPE, copy=True)



X_16 = X.astype(np.float16, copy=True)

Y_16 = Y.astype(np.float16, copy=True)

'''
np.savez_compressed(
    "jets_float32_compressed.npz",
    X=X_32,
    Y=Y_32
)
np.savez_compressed(
    "jets_float16.npz_compressed",
    X=X_16,
    Y=Y_16
)
'''

np.savez(
    "jets_float64.npz",
    X=X,
    Y=y
)

np.savez_compressed(
    "jets_float64_compressed.npz",
    X=X,
    Y=y
)