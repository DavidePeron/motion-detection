import h5py
import numpy as np




def load_dataset():
	ds = h5py.File('train_dataset.h5', 'r')
	X_train = ds['X_train']
	Y_train = ds['Y_train']

	return X_train, Y_train
