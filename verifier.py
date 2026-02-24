import numpy as np
import os

orig_dir = "/Users/nicholasdilullo/Desktop/research/LeBlancLab/pfn_work/efcache/datasets"
f32_dir  = "/Users/nicholasdilullo/Desktop/research/LeBlancLab/pfn_work/efcache/out/fp32"
f16_dir  = "/Users/nicholasdilullo/Desktop/research/LeBlancLab/pfn_work/efcache/out/fp16"

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

def verifier(fname):
    print(f"{fname}")
    
    original = np.load(os.path.join(orig_dir, fname))
    fp32  = np.load(os.path.join(f32_dir,  fname))
    fp16  = np.load(os.path.join(f16_dir,  fname))

    X, y = original["X"], original["y"]
    X32, y32 = fp32["X"], fp32["y"]
    X16, y16 = fp16["X"], fp16["y"]

    #shapes
    print("Shapes:", X.shape, X32.shape, X16.shape)
    assert X.shape == X32.shape == X16.shape
    assert y.shape == y32.shape == y16.shape
    '''
    always 100000 jets/file and 4 components (structural thing of original dataset)
    so this should always be true
    '''
    assert X.shape[0] == X32.shape[0] == X16.shape[0] == 100000
    assert X.shape[2] == X32.shape[2] == X16.shape[2] == 4

    '''
    labels identical
    0 or 1 should not be different across dtype sizing
    '''
    assert np.array_equal(y, y32)
    assert np.array_equal(y, y16)
    print("Labels match")

    '''
    value differences due to rounding only
    not sure if there is a better way to test this then just doing it again
    '''
    diff32 = np.abs(X.astype(np.float32) - X32)
    diff16 = np.abs(X.astype(np.float16) - X16)

    print("float32 max error:", diff32.max())
    print("float16 max error:", diff16.max())

    assert diff32.max() == 0
    assert diff16.max() == 0

    print("Values match!")

for f in filenames:
    verifier(f)

print("Hooray!")