from __future__ import absolute_import, division, print_function

import numpy as np

print('Loading the dataset ...')

cache_directory = '/Users/nicholasdilullo/Desktop/research/LeBlancLab/pfn_work/efcache'
out_directory = '/Users/nicholasdilullo/Desktop/research/LeBlancLab/pfn_work/efcache/out'
filenames = ["QG_jets_1.npz",
    "QG_jets_2.npz",
    "QG_jets_3.npz",
    "QG_jets_4.npz",
    "QG_jets_5.npz",
    "QG_jets_6.npz",
    "QG_jets_7.npz",
    "QG_jets_8.npz",
    "QG_jets_9.npz",
    "QG_jets_10.npz",
    "QG_jets_11.npz",
    "QG_jets_12.npz",
    "QG_jets_13.npz",
    "QG_jets_14.npz",
    "QG_jets_15.npz",
    "QG_jets_16.npz",
    "QG_jets_17.npz",
    "QG_jets_18.npz",
    "QG_jets_19.npz",
    "QG_jets.npz"]

for filename in filenames:
    print(cache_directory + "/" + filename)

'''
# load data
X, y = qg_jets.load(2000000, generator='pythia', pad=True, cache_dir='/Users/nicholasdilullo/Desktop/research/LeBlancLab/pfn_work/efcache')
Y = to_categorical(y, num_classes=2)

X_32 = X.astype(TARGET_DTYPE, copy=True)

Y_32 = Y.astype(TARGET_DTYPE, copy=True)



X_16 = X.astype(np.float16, copy=True)

Y_16 = Y.astype(np.float16, copy=True)
'''
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
'''