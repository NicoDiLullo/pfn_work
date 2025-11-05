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
X_base, y_base = qg_jets.load(train + val + test, generator='pythia', pad=True, cache_dir='/Users/nicholasdilullo/Desktop/research/LeBlancLab/pfn_work/efcache')

print('Dataset loaded!')

def load_and_run(X, y, dtype):

    X = X.astype(dtype, copy=False)
    print('Datatypes switched!')

    total_elements = X.size
    item_size = X.itemsize
    total_size_bytes = total_elements * item_size

    print("Dataset statistics:")
    print(f'shape: {X.shape}')
    print(f'Total dataset size in bytes with dtype {dtype}: {total_size_bytes}')

    # convert labels to categorical
    Y = to_categorical(y, num_classes=2)

    print('Loaded quark and gluon jets')

    # preprocess by centering jets and normalizing pts
    for x in X:
        mask = x[:, 0] > dtype(0)
        w = x[mask, 0].astype(dtype, copy=False)
        coords = x[mask, 1:3]
        wsum = w.sum(dtype=dtype)
        if wsum > dtype(0):
            num = (coords * w[:, None]).sum(axis=0, dtype=dtype)
            yphi_avg = num / wsum
            x[mask, 1:3] -= yphi_avg.astype(dtype, copy=False)

        # normalize pt in TARGET_DTYPE
        denom = x[:, 0].sum(dtype=dtype)
        if denom > dtype(0):
            x[mask, 0] /= denom

    # handle particle id channel
    if use_pids:
        remap_pids(X, pid_i=3)
    else:
        X = X[:,:,:3]

    print('Finished preprocessing')

    print('Model summary:')

    # build architecture
    pfn = PFN(input_dim=X.shape[-1], Phi_sizes=Phi_sizes, F_sizes=F_sizes)
    return (X, Y)

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC


X_64, Y_64 = load_and_run(X_base.copy(), y_base.copy(), np.float64)
X_32, Y_64 = load_and_run(X_base.copy(), y_base.copy(), np.float32)
X_16, Y_64 = load_and_run(X_base.copy(), y_base.copy(), np.float16)

def run_linear_classifier(X1, X2, label1, label2, seed):
    print(f"\n[Linear Model] Classifying {label1} vs {label2}...")
    n = min(len(X1), len(X2))
    X = np.vstack((X1.reshape(len(X1), -1)[:n],
                   X2.reshape(len(X2), -1)[:n]))
    y = np.array([0]*n + [1]*n)

    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )

    # Logistic Regression
    #lr = LogisticRegression(max_iter=1000, n_jobs=-1)
    #lr.fit(Xtr, ytr)
    #yhat_lr = lr.predict(Xte)
    #print("LogReg acc:", accuracy_score(yte, yhat_lr))

    # Linear SVM
    svm = LinearSVC(dual=False)
    svm.fit(Xtr, ytr)
    yhat_svm = svm.predict(Xte)
    print(accuracy_score(yte, yhat_svm))
#run_linear_classifier(X_64, X_32, 'float64', 'float32', seed=42)
#run_linear_classifier(X_32, X_16, 'float32', 'float16', seed=42)
accs = [run_linear_classifier(X_64, X_16, 'f64', 'f16', s) for s in range(10)]
print(np.mean(accs), np.std(accs))

'''

eq = np.array_equal(X_64.astype(np.float32), X_32)
print("fp64→fp32 equality:", eq)

def run_classifier(X1, X2, label1, label2, seed):
    print(f'Classifying {label1} vs {label2}...')
    features1 = X1.reshape(len(X1), -1)
    features2 = X2.reshape(len(X2), -1)
    features_stacked = np.vstack((features1, features2))
    labels = np.array([0] * len(X1) + [1] * len(X2))

    X_train, X_test, y_train, y_test = train_test_split(features_stacked, labels, test_size=0.2, random_state=seed, stratify=labels)

    classifier = DecisionTreeClassifier(random_state=seed, max_depth=12, min_samples_leaf=50)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy of {label1} vs {label2} classification: {accuracy * 100:.4f}%')
'''
#run_classifier(X_64, X_32, 'float64', 'float32', seed=42)
#run_classifier(X_32, X_16, 'float32', 'float16', seed=42)

'''
accs = [run_classifier(X_64, X_32, 'f64','f32', s) for s in range(10)]
print("f64 vs f32 mean±std:", np.mean(accs), np.std(accs))


#classify
def classifier(seed):
    print('Starting dtype classification...')
    features_64 = X_64.reshape(len(X_64), -1)
    features_32 = X_32.reshape(len(X_32), -1)
    features_16 = X_16.reshape(len(X_16), -1)
    print('Reshaped features for classification.')

    features_stacked = np.vstack(features_64, features_32, features_16)
    labels = np.array([0] * len(X_64) + [1] * len(X_32) + [2] * len(X_16))

    X_train, X_test, y_train, y_test = train_test_split(features_stacked, labels, test_size=0.2, random_state=seed)

    classifier = DecisionTreeClassifier(random_state=seed)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy of dtype classification: {accuracy * 100:.4f}%')

    print('Classifying float32 vs float64...')
    features32vs64 = np.vstack((features_32, features_64))
    labels32vs64 = np.array([0] * len(X_32) + [1] * len(X_64))
    classifier32vs64 = DecisionTreeClassifier(random_state=seed)
    X_train32vs64, X_test32vs64, y_train32vs64, y_test32vs64 = train_test_split(features32vs64, labels32vs64, test_size=0.2, random_state=seed)
    classifier32vs64.fit(X_train32vs64, y_train32vs64)
    y_pred32vs64 = classifier32vs64.predict(X_test32vs64)
    accuracy32vs64 = accuracy_score(y_test32vs64, y_pred32vs64)
    print(f'Accuracy of float32 vs float64 classification: {accuracy32vs64 * 100:.4f}%')

    print('Classifying float16 vs float32...')
    features32v16 = np.vstack((features_32, features_16))
    labels32vs16 = np.array([0] * len(X_32) + [1] * len(X_16))
    classifier32vs16 = DecisionTreeClassifier(random_state=seed)
    X_train32vs16, X_test32vs16, y_train32vs16, y_test32vs16 = train_test_split(features32v16, labels32vs16, test_size=0.2, random_state=seed)
    classifier32vs16.fit(X_train32vs16, y_train32vs16)
    y_pred32vs16 = classifier32vs16.predict(X_test32vs16)
    accuracy32vs16 = accuracy_score(y_test32vs16, y_pred32vs16)
    print(f'Accuracy of float32 vs float16 classification: {accuracy32vs16 * 100:.4f}%')

print("Running classifier w seed 42")
classifier(seed=42)
'''