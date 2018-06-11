import scipy.io as sio
import pandas as pd
import numpy as np
import h5py
import matplotlib.pyplot as plt

def get_min_length(data):
	array_of_lenghts = np.array([]);
	for column in data:
		sample = data[column][0]
		height = sample[:,1].size
		array_of_lenghts = np.append(array_of_lenghts, height) #column vector

	#found the input column dimension
	min_length = int(np.amin(array_of_lenghts))
	return min_length

def preprocessing(sample, indexes):
	# Search for missing data
	for i in range(1, indexes.shape[0] - 1, 2):
		if(indexes[i] != indexes[i+1] - 1):
			shift = indexes[i+1] - indexes[i] - 1
			# Cut sample vector
			sample = np.concatenate((sample[:indexes[i]], sample[indexes[i+1]:]), axis=0)
			# Shift indexes vector
			indexes[i+1:] = [x - shift for x in indexes[i+1:]]
	return sample, indexes

def get_padding(sample, min_length):
	# Calculate in how much inputs can be divided this sample
	n_inputs = int(np.floor(sample.shape[0]/min_length))
	# Compute padding
	d_padding = sample.shape[0] - n_inputs * min_length
	if(d_padding % 2 == 0):
		right_padding = int(d_padding/2)
		left_padding = right_padding
	else:
		right_padding = int(d_padding/2)
		left_padding = right_padding + 1

	return left_padding, right_padding, n_inputs


# Build the structures
activities_dict = {'RUNNING': 0, 'WALKING': 1, 'JUMPING': 2, 'STNDING': 3, 'SITTING': 4, 'XLYINGX': 5, 'FALLING': 6, 'TRANSUP': 7, 'TRANSDW': 8, 'TRNSACC': 9, 'TRNSDCC': 10, 'TRANSIT': 11}

# Read the .mat file
mat = sio.loadmat('ARS_DLR_DataSet_V2.mat')
mat = {k:v for k, v in mat.items() if k[0] != '_'}
data = pd.DataFrame({k: pd.Series(v[0]) for k, v in mat.items()})

# Uncomment this to have maximum window length
# min_length = get_min_length(data)

# Window size in seconds
window_size = 10
# Uncomment this to have fixed window length
min_length = int(window_size/0.01)

X = []
Y = []

# Cycle over all the people in the dataset
for column in data:
	sample = data[column][0]
	attitude = data[column][1]
	sample_activities = data[column][2][0]
	indexes = np.squeeze(data[column][3]) - 1 # Remove 1 since data are saved in matlab and indexes start from 1 D:

	# Change of coordinates to be in global frame
	sample[:,1:] *= attitude[:,1:]

	[sample, indexes] = preprocessing(sample, indexes)
	# Compute number of inputs per sample and paddings
	[left_padding, right_padding, n_inputs] = get_padding(sample, min_length)

	# Build a matrix for X and a vector for Y with the same number of elements to be fed by the CNN
	# We use list since the append method is faster
	# Exapand Y
	Y_single = []
	num_tot = 0
	for i in range(sample_activities.size):
		number_of_repetitions = indexes[2*i+1] - indexes[2*i] + 1
		num_tot += number_of_repetitions
		activity = activities_dict[sample_activities[i][0]]
		Y_single += [activity] * number_of_repetitions

	# Cut paddings out of the sample and Y_single
	sample = sample[left_padding:sample.shape[0] - right_padding, 1:]
	Y_single = Y_single[left_padding:np.shape(Y_single)[0] - right_padding]

	for k in range(0, n_inputs):
		X.append(sample[min_length * k:min_length * (k+1), :])
		Y.append(Y_single[min_length * k: min_length * (k+1)])

# Transform list to numpy array for the sake of the computational flexibility
X = np.array(X)
Y = np.array(Y)
print(np.shape(X))
print(np.shape(Y))
Y = Y.reshape(Y.shape[0], Y.shape[1], 1)

# Take 80% of the dataset as training set
trainingNorm = int(np.ceil(X.shape[0]/100*80))

# Divide dataset in training and test set
X_train = X[:trainingNorm, :, :]
Y_train = Y[:trainingNorm, :, :]
X_test = X[trainingNorm:, :, :]
Y_test = Y[trainingNorm:, :, :]

# print(Y_train.shape)
# print(Y_test.shape)
# Create files dataset
train_dataset = {'X_train': X_train, 'Y_train': Y_train}
train_dataset = {'X_test': X_test, 'Y_test': Y_test}

ds = h5py.File('train_dataset.h5', 'w')
ds.create_dataset('X_train', data=X_train)
ds.create_dataset('Y_train', data=Y_train)
ds.close()

ds = h5py.File('test_dataset.h5', 'w')
ds.create_dataset('X_test', data=X_test)
ds.create_dataset('Y_test', data=Y_test)
ds.close()
