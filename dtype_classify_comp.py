import matplotlib.pyplot as plt
import numpy as np

# Data
labels = [
    "Logistic FP32 vs FP64",
    "LinearSVC FP32 vs FP64",
    "Tree FP32 vs FP64",
    "Logistic FP16 vs FP64",
    "LinearSVC FP16 vs FP64",
    "Tree FP16 vs FP64",
]
means = [0.4758, 0.4755, 0.4695, 0.4759, 0.4755, 0.5114]
stds  = [0.0017, 0.0017, 0.0147, 0.0017, 0.0017, 0.0014]

# Plot
plt.figure(figsize=(7, 4))
plt.axhline(0.5, color='gray', linestyle='--', label='Chance (0.5)')

x = np.arange(len(labels))
plt.errorbar(x, means, yerr=stds, fmt='o', capsize=4, label="Accuracy \pms std")

plt.xticks(x, labels, rotation=30, ha='right')
plt.ylabel("Accuracy")
plt.title("Classifier Accuracy Comparison by Data Type")
plt.legend()
plt.tight_layout()
plt.show()
