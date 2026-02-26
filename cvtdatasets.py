import numpy as np
import os

#print('Loading the dataset ...')

cache_directory = '/users/ndilullo/work/pfn_work/efcache'
out_directory = '/users/ndilullo/work/pfn_work/efcache/out'
os.makedirs(out_directory, exist_ok=True)

filenames = [
    "QG_jets.npz",
    "QG_jets_1.npz",
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
    "QG_jets_19.npz"]

def float_32(compressed=False):
    out_directory_float32 = os.path.join(out_directory, "fp32")
    os.makedirs(out_directory_float32, exist_ok=True)
    for filename in filenames:
        source = os.path.join(cache_directory, filename)
        with np.load(source) as data:
            X = data["X"].astype(np.float32, copy=True)
            y = data["y"].astype(np.float32, copy=True)
            #filename = filename.replace(".npz", "")
            if compressed:
                #filename = filename + "_float32compressed.npz"
                write_to = os.path.join(out_directory_float32, filename)
                np.savez_compressed(write_to, X=X, y=y)
            else:
                #filename = filename + "_float32.npz"
                write_to = os.path.join(out_directory_float32, filename)
                np.savez(write_to, X=X, y=y)


def float_16(compressed=False):
    out_directory_float16 = os.path.join(out_directory, "fp16")
    os.makedirs(out_directory_float16, exist_ok=True)
    for filename in filenames:
        source = os.path.join(cache_directory, filename)
        with np.load(source) as data:
            X = data["X"].astype(np.float16, copy=True)
            y = data["y"].astype(np.float16, copy=True)
            #filename = filename.replace(".npz", "")
            if compressed:
                #filename = filename + "_float16compressed.npz"
                write_to = os.path.join(out_directory_float16, filename)
                np.savez_compressed(write_to, X=X, y=y)
            else:
                #filename = filename + "_float16.npz"
                write_to = os.path.join(out_directory_float16, filename)
                np.savez(write_to, X=X, y=y)

def bfloat_16(compressed=False):
    out_directory_float16 = os.path.join(out_directory, "bf16")
    os.makedirs(out_directory_float16, exist_ok=True)
    for filename in filenames:
        source = os.path.join(cache_directory, filename)
        with np.load(source) as data:
            X = data["X"].astype(np.bfloat16, copy=True)
            y = data["y"].astype(np.bfloat16, copy=True)
            #filename = filename.replace(".npz", "")
            if compressed:
                #filename = filename + "_float16compressed.npz"
                write_to = os.path.join(out_directory_float16, filename)
                np.savez_compressed(write_to, X=X, y=y)
            else:
                #filename = filename + "_float16.npz"
                write_to = os.path.join(out_directory_float16, filename)
                np.savez(write_to, X=X, y=y)
def main():
    float_32(True)
    float_16(True)
    #bfloat_16(True)


if __name__ == "__main__":
    main()