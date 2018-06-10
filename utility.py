import h5py
import numpy as np

def load_dataset():
	ds_train = h5py.File('train_dataset.h5', 'r')
	X_train = ds_train['X_train']
	Y_train = ds_train['Y_train']
	ds_test = h5py.File('test_dataset.h5', 'r')
	X_test = ds_test['X_test']
	Y_test = ds_test['Y_test']
	return X_train, Y_train, X_test, Y_test
