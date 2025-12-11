import numpy as np
import os

print('Loading the dataset ...')

cache_directory = '/Users/nicholasdilullo/Desktop/research/LeBlancLab/pfn_work/efcache'
out_directory = '/Users/nicholasdilullo/Desktop/research/LeBlancLab/pfn_work/efcache/out'
os.makedirs(out_directory, exist_ok=True)

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

def float_32(compressed=False):
    for filename in filenames:
        source = os.path.join(cache_directory, filename)
        with np.load(source) as data:
            X = data["X"].astype(np.float32, copy=True)
            y = data["y"].astype(np.float32, copy=True)
            filename.replace(".npz", "")
            if compressed:
                filename + "_float32compressed.npz"
                write_to = os.path.join(out_directory, filename)
                np.savez_compressed(write_to, X=X, y=y)
            else:
                filename + "_float32.npz"
                write_to = os.path.join(out_directory, filename)
                np.savez_compressed(write_to, X=X, y=y)


def float_16(compressed=False):
    for filename in filenames:
        source = os.path.join(cache_directory, filename)
        with np.load(source) as data:
            X = data["X"].astype(np.float16, copy=True)
            y = data["y"].astype(np.float16, copy=True)
            filename.replace(".npz", "")
            if compressed:
                filename + "_float16compressed.npz"
                write_to = os.path.join(out_directory, filename)
                np.savez_compressed(write_to, X=X, y=y)
            else:
                filename + "_float16.npz"
                write_to = os.path.join(out_directory, filename)
                np.savez_compressed(write_to, X=X, y=y)
def main():
    float_32()
    float_16()

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