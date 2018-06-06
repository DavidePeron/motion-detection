import scipy.io as sio
import pandas as pd
import numpy as np
import h5py
import matplotlib.pyplot as plt

# Build the structures
activities_dict = {'RUNNING': 0, 'WALKING': 1, 'JUMPING': 2, 'STNDING': 3, 'SITTING': 4, 'XLYINGX': 5, 'FALLING': 6, 'TRANSUP': 7, 'TRANSDW': 8, 'TRNSACC': 9, 'TRNSDCC': 10, 'TRANSIT': 11}


# Read the .mat file
mat = sio.loadmat('ARS_DLR_DataSet_V2.mat')
mat = {k:v for k, v in mat.items() if k[0] != '_'}
data = pd.DataFrame({k: pd.Series(v[0]) for k, v in mat.items()})


array_of_lenghts = np.array([]);
for column in data:
	sample = data[column][0]
	height = sample[:,1].size
	array_of_lenghts = np.append(array_of_lenghts, height) #column vector

#found the input column dimension
min_length = int(np.amin(array_of_lenghts))


#create X_train and Y_train
X_train = []
Y_train = []


# Cycle over all the people in the dataset
for column in data:

	sample = data[column][0]
	attitude = data[column][1]
	sample_activities = data[column][2][0]
	indexes = np.squeeze(data[column][3]) - 1 # Remove 1 since data are saved in matlab and indexes start from 1 D:

	# Change of coordinates to be in global frame
	sample[:,1:] *= attitude[:,1:]

	# Build a matrix for X_train and a vector for Y_train with the same number of elements to be fed by the CNN
	# We use list since the append method is faster
	X_train.append(sample[:min_length,1:])

	Y_train_single = []
	j = 0
	for i in range(sample_activities.size):
		number_of_repetitions = indexes[j+1] - indexes[j] + 1
		activity = activities_dict[sample_activities[i][0]]
		Y_train_single += [activity] * number_of_repetitions
		j = j+2

	Y_train_single = Y_train_single[:min_length]
	Y_train.append(Y_train_single)

# Transform list to numpy array for the sake of the computational flexibility
X_train = np.array(X_train)
Y_train = np.array(Y_train)
Y_train = Y_train.reshape(Y_train.shape[0], Y_train.shape[1], 1)

train_dataset = {'X_train': X_train, 'Y_train': Y_train}

ds = h5py.File('train_dataset.h5', 'w')
ds.create_dataset('X_train', data=X_train)
ds.create_dataset('Y_train', data=Y_train)
ds.close()
