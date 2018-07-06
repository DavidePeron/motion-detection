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

#add num_channels = 1 at the and of each tuple, methods inside ActivityRecognition model require a 4D tensor
def resize_input(X_train, Y_train, X_test, Y_test):
	# X_train = np.reshape(X_train, (np.shape(X_train)[0], np.shape(X_train)[1], np.shape(X_train)[2], 1))
	Y_train = np.reshape(Y_train, (np.shape(Y_train)[0], np.shape(Y_train)[1]))
	# X_test = np.reshape(X_test, (np.shape(X_test)[0], np.shape(X_test)[1], np.shape(X_test)[2], 1))
	Y_test = np.reshape(Y_test, (np.shape(Y_test)[0], np.shape(Y_test)[1]))

	return X_train, Y_train, X_test, Y_test


# Turn data into global frame
def data_to_global_frame(sample, attitude):
	for row in range(0, attitude.shape[0]):
		Cb = np.reshape(attitude[row,:], (3,3))
		for i in range(0, 6, 3):
			sample[row, i:i+3] = np.dot(Cb.T, sample[row, i:i+3].T)
	return sample


