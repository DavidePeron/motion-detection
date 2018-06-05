import scipy.io as sio
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Build the structures
activities_dict = {'RUNNING': 0, 'WALKING': 1, 'JUMPING': 2, 'STNDING': 3, 'SITTING': 4, 'XLYINGX': 5, 'FALLING': 6, 'TRANSUP': 7, 'TRANSDW': 8, 'TRANSACC': 9, 'TRANSDCC': 10, 'TRANSIT': 11}


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
#print(array_of_lenghts)
#print(min_length)



#create X_train and Y_train
X_train = np.empty((min_length, 9, 0))
Y_train = np.empty((min_length, 1, 0))


for column in data:

	sample = data[column][0]
	attitude = data[column][1]
	sample_activities = data[column][2]
	indexes = np.squeeze(data[column][3]) - 1 # Remove 1 since data are saved in matlab and indexes start from 1 D:

	# Change of coordinates to be in global frame
	sample[:,1:] *= attitude[:,1:]
	#print(sample[:min_length,1:].shape)
	X_train = np.column_stack((X_train, sample[:min_length,1:]))
	print(X_train.shape)
	Y_train_single = np.array([])

	j = 0
	for i in range(sample_activities[0].size):
		number_of_repetitions = indexes[j+1] - indexes[j] + 1
		Y_train_single = np.append(Y_train_single, np.repeat(sample_activities[0][i],number_of_repetitions))
		j = j+2
	
	Y_train_single = Y_train_single[:min_length].reshape((min_length,1))

	Y_train = np.append(Y_train, [Y_train_single], axis = 2)

	print(Y_train.shape)
