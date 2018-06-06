import h5py
import numpy as np

ds = h5py.File('train_dataset.h5', 'r')
X_train = ds['X_train']
print(X_train.shape)
