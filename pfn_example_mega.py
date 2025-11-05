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
#TARGET_DTYPE = np.float32

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
X0, y0 = qg_jets.load(train + val + test, generator='pythia', pad=True, cache_dir='/users/ndilullo/work/pfn_work/efcache')

print('Dataset loaded!')

def load_and_run(dtype, seed):
    tf.keras.utils.set_random_seed(seed)
    if dtype is np.float16:
        tf.keras.mixed_precision.set_global_policy('mixed_float16')

    X = X0.astype(dtype, copy=True)
    print('Datatypes switched!')

    # convert labels to categorical
    Y = to_categorical(y0, num_classes=2)

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

    # do train/val/test split 
    (X_train, X_val, X_test,
    Y_train, Y_val, Y_test) = data_split(X, Y, val=val, test=test)

    print('Done train/val/test split')
    print('Model summary:')

    # build architecture
    pfn = PFN(input_dim=X.shape[-1], Phi_sizes=Phi_sizes, F_sizes=F_sizes)

    # train model
    pfn.fit(X_train, Y_train,
            epochs=num_epoch,
            batch_size=batch_size,
            validation_data=(X_val, Y_val),
            verbose=1,
            callbacks=[es,mc])

    # get predictions on test data
    preds = pfn.predict(X_test, batch_size=1000)

    # get ROC curve
    pfn_fp, pfn_tp, threshs = roc_curve(Y_test[:,1], preds[:,1])

    # get area under the ROC curve
    auc = roc_auc_score(Y_test[:,1], preds[:,1])
    print()
    print('PFN AUC:', auc)
    print()

    # get multiplicity and mass for comparison
    #masses = np.asarray([ef.ms_from_p4s(ef.p4s_from_ptyphims(x).sum(axis=0)) for x in X_test])
    #mults = np.asarray([np.count_nonzero(x[:,0]) for x in X_test])
    #mass_fp, mass_tp, threshs = roc_curve(Y[:,1], -masses)
    #mult_fp, mult_tp, threshs = roc_curve(Y[:,1], -mults)
    #mass_auc = roc_auc_score(Y_test[:,1], masses)
    #mult_auc = roc_auc_score(Y_test[:,1], mults)
    #return auc, {(mass_fp, mass_tp), (mult_fp, mult_tp)}
   # return auc, mass_auc, mult_auc
    return auc

float64_accs = []
float32_accs = []
float16_accs = []
float64_mass_accs = []
float32_mass_accs = []
float16_mass_accs = []
float64_mult_accs = []
float32_mult_accs = []
float16_mult_accs = []

'''float64_oi = []
float32_oi = []
float16_oi = []'''


for i in range(10):
    a  = load_and_run(np.float64, seed=i*42)
    #a, b, c  = load_and_run(np.float64, seed=i*42)
    #x, f64oi = load_and_run(np.float64, seed=i*42)
    float64_accs.append(a)
    #float64_mass_accs.append(b)
    #float64_mult_accs.append(c)
    #float64_oi.append(f64oi)
for i in range(10):
    a = load_and_run(np.float32, seed=i*42)
    # a, b, c = load_and_run(np.float32, seed=i*42)
    #x, f32oi = load_and_run(np.float32, seed=i*42)
    float32_accs.append(a)
    #float32_mass_accs.append(b)
    #float32_mult_accs.append(c)
    #float32_oi.append(f32oi)
for i in range(10):
    a = load_and_run(np.float16, seed=i*42)
    #a, b, c = load_and_run(np.float16, seed=i*42)
    #x, f16oi = load_and_run(np.float16, seed=i*42)
    float16_accs.append(a)
    #float16_mass_accs.append(b)
    #float16_mult_accs.append(c)
    #float16_oi.append(f16oi)

print("float64 accs:", float64_accs)
print("float32 accs:", float32_accs)
print("float16 accs:", float16_accs)
'''
print("float64 otherinfo: " + str(float64_oi))
print("float32 otherinfo: " + str(float32_oi))
print("float16 otherinfo: " + str(float16_oi))
'''
with open('pfn_results.txt', 'w') as f:
    f.write("Float64 AUCs: " + str(float64_accs) + "\n")
    f.write("Float32 AUCs: " + str(float32_accs) + "\n")
    f.write("Float16 AUCs: " + str(float16_accs) + "\n")
    '''
    f.write("Float64 otherinfo: " + str(float64_oi) + "\n")
    f.write("Float32 otherinfo: " + str(float32_oi) + "\n")
    f.write("Float16 otherinfo: " + str(float16_oi) + "\n")
    '''