from energyflow.archs.efn import PFN
from energyflow.datasets import qg_jets
from energyflow.utils import data_split, remap_pids, to_categorical

#42000 is some arbitrary value < train + validate + test
X, y = qg_jets.load(42000, generator='pythia', pad=True, cache_dir='/oscar/data/mleblan6/energyflow')
print("Loaded dtypes -> X:", X.dtype, " y:", y.dtype)
print("Shapes -> X:", X.shape, " y:", y.shape)
