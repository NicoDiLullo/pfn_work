import matplotlib.pyplot as plt

sizes = [9472000000, 4736000000, 2368000000]
labels = ['float64', 'float32', 'float16']

plt.bar(labels, sizes, color=['tab:blue'])
plt.title('Dataset (X) Size by Data Type')
plt.xlabel('Data Type')
plt.ylabel('Size (Bytes)')
plt.show()