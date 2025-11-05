Just don't do it.

Some chronicles from trying to get this to run on my mac, and instructions for you to do the same.

First, create your venv as before

python3 -m venv tensorflow.venv
source tensorflow.venv/bin/activate

Then, install the requirements for Mac (the regular requirements - (minus) a bunch of finicky Nvidia packages that do not work on MacOS). 

pip install -r requirements_mac.txt

Install tensorflow metal (needed to leverage the magic of ARM engineering)

pip install tensorflow-metal

Change the cache directory to something on your laptop (line 78)
[TODO better instructions]

You may have some issues with Zenodo and ssl certificates. 
urllib.error.URLError: <urlopen error [SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed: unable to get local issuer certificate (_ssl.c:997)>

and the associated

RuntimeError: Failed to download QG_jets.npz from any source.

To fix:

/Applications/Python\ 3.10/Install\ Certificates.command

^^ run from root

Some issues: 

1) exploding loss:

Epoch 1/500
2025-11-05 00:13:25.562480: I tensorflow/core/grappler/optimizers/custom_graph_optimizer_registry.cc:117] Plugin optimizer for device_type GPU is enabled.
2996/3000 [============================>.] - ETA: 0s - loss: 63.3003 - acc: 0.6890     
Epoch 1: val_acc improved from -inf to 0.65605, saving model to best_pfn.keras
3000/3000 [==============================] - 44s 14ms/step - loss: 63.8205 - acc: 0.6890 - val_loss: 731.6821 - val_acc: 0.6560
Epoch 2/500
2998/3000 [============================>.] - ETA: 0s - loss: 3019.6223 - acc: 0.6922 
Epoch 2: val_acc improved from 0.65605 to 0.70166, saving model to best_pfn.keras
3000/3000 [==============================] - 41s 14ms/step - loss: 3021.6951 - acc: 0.6923 - val_loss: 9110.6514 - val_acc: 0.7017
Epoch 3/500
2997/3000 [============================>.] - ETA: 0s - loss: 20601.8711 - acc: 0.6936 
Epoch 3: val_acc did not improve from 0.70166
3000/3000 [==============================] - 41s 14ms/step - loss: 20607.7539 - acc: 0.6936 - val_loss: 145275.0469 - val_acc: 0.5021
Epoch 4/500
2998/3000 [============================>.] - ETA: 0s - loss: 81993.9375 - acc: 0.6921 
Epoch 4: val_acc improved from 0.70166 to 0.72713, saving model to best_pfn.keras
3000/3000 [==============================] - 41s 14ms/step - loss: 81991.7109 - acc: 0.6921 - val_loss: 97760.1719 - val_acc: 0.7271
Epoch 5/500
2997/3000 [============================>.] - ETA: 0s - loss: 235404.4219 - acc: 0.6920 
Epoch 5: val_acc did not improve from 0.72713
3000/3000 [==============================] - 41s 14ms/step - loss: 235440.1719 - acc: 0.6920 - val_loss: 794288.5625 - val_acc: 0.5062
Epoch 6/500
2997/3000 [============================>.] - ETA: 0s - loss: 537836.6875 - acc: 0.6934 
Epoch 6: val_acc did not improve from 0.72713
3000/3000 [==============================] - 41s 14ms/step - loss: 537951.3125 - acc: 0.6933 - val_loss: 1046555.1250 - val_acc: 0.6650
Epoch 7/500
2999/3000 [============================>.] - ETA: 0s - loss: 1151990.0000 - acc: 0.6925 
Epoch 7: val_acc did not improve from 0.72713
3000/3000 [==============================] - 41s 14ms/step - loss: 1152263.6250 - acc: 0.6925 - val_loss: 3889878.2500 - val_acc: 0.6020
Epoch 8/500
2998/3000 [============================>.] - ETA: 0s - loss: 2080828.1250 - acc: 0.6917 
Epoch 8: val_acc did not improve from 0.72713
3000/3000 [==============================] - 42s 14ms/step - loss: 2083035.7500 - acc: 0.6917 - val_loss: 3533226.7500 - val_acc: 0.6027
Epoch 9/500
2996/3000 [============================>.] - ETA: 0s - loss: 3817889.7500 - acc: 0.6883  
Epoch 9: val_acc did not improve from 0.72713
3000/3000 [==============================] - 42s 14ms/step - loss: 3821664.5000 - acc: 0.6883 - val_loss: 7421041.5000 - val_acc: 0.6768
Epoch 10/500
2998/3000 [============================>.] - ETA: 0s - loss: 6399135.0000 - acc: 0.6880 
Epoch 10: val_acc did not improve from 0.72713
3000/3000 [==============================] - 42s 14ms/step - loss: 6397359.5000 - acc: 0.6881 - val_loss: 4734430.5000 - val_acc: 0.7180
Epoch 11/500
2998/3000 [============================>.] - ETA: 0s - loss: 10812776.0000 - acc: 0.6856 
Epoch 11: val_acc did not improve from 0.72713
3000/3000 [==============================] - 44s 15ms/step - loss: 10814813.0000 - acc: 0.6856 - val_loss: 14921833.0000 - val_acc: 0.6814
Epoch 12/500
3000/3000 [==============================] - ETA: 0s - loss: 16099194.0000 - acc: 0.6883 
Epoch 12: val_acc did not improve from 0.72713
3000/3000 [==============================] - 43s 14ms/step - loss: 16099194.0000 - acc: 0.6883 - val_loss: 25387828.0000 - val_acc: 0.6556

Fix:
1) Add 
import os
os.environ['TF_DISABLE_MLIR_GPU_OPS'] = '1'  # Fallback to CPU
before future import

If don't see the same issues, this was bc of GPU/metal issues

pip uninstall tensorflow keras ml-dtypes tensorflow-io-gcs-filesystem tensorflow-estimator tensorboard
pip install tensorflow==2.18.1 --no-deps
pip install tensorflow-metal


pip install "keras>=3.5.0"
pip install "tensorboard>=2.18,<2.19"
pip install "tensorflow-io-gcs-filesystem>=0.23.1"
pip install "flatbuffers>=24.3.25"
pip install "h5py>=3.11.0"
pip install "ml_dtypes>=0.4.0"