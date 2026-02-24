import numpy as np
import matplotlib.pyplot as plt

# load scores
preds64 = np.load('preds_fp64.npy')
preds32 = np.load('preds_fp32.npy')

# basic sanity check
if preds64.shape != preds32.shape:
    raise ValueError(f"Shape mismatch: {preds64.shape} vs {preds32.shape}")

plt.figure(figsize=(5, 5))

# 2D density plot
plt.hexbin(preds64, preds32, gridsize=100, bins='log')
plt.plot([0, 1], [0, 1], 'k--', linewidth=1)  # y = x line

plt.xlabel('PFN score (float64)')
plt.ylabel('PFN score (float32)')
plt.title('PFN output correlation: float64 vs float32')
cb = plt.colorbar()
cb.set_label('log10(N jets)')

plt.xlim(0, 1)
plt.ylim(0, 1)

plt.tight_layout()
plt.savefig('pfn_corr_fp64_fp32.png')
# plt.show()
plt.close()
