import scipy.io as sio
import pandas as pd
import numpy as np
import h5py
import matplotlib.pyplot as plt
import logging

'''
promemoria per gli indici
a[start:end]		# items start through end-1
a[start:]			# items start through the rest of the array
a[:end]				# items from the beginning through end-1
a[:]				# a copy of the whole array
a[start:end:step]	# start through not past end, by step
a[-1]				# last item in the array
a[-2:]				# last two items in the array
a[:-2]				# everything except the last two items
a[::-1]				# all items in the array, reversed
a[1::-1]			# the first two items, reversed
a[:-3:-1]			# the last two items, reversed
a[-3::-1]			# everything except the last two items, reversed
'''



def get_realization_min_length(data):
	array_of_lenghts = np.array([])
	for column in data:
		sample = data[column][0]
		height = sample[:,1].size
		array_of_lenghts = np.append(array_of_lenghts, height) #column vector

	#found the input column dimension
	min_length = int(np.amin(array_of_lenghts))
	return min_length

def get_pattern_min_length(data):
	array_of_lengths = np.array([])

	for column in data:
		indexes = np.squeeze(data[column][3]) - 1
		for i in range(0, indexes.shape[0], 2):
			length = indexes[i+1] - indexes[i]
			array_of_lengths = np.append(array_of_lengths, length)

	min_length = int(np.amin(array_of_lengths))
	return min_length


#preprocessing auxiliary function
def swap_indexes (a, b):
	tmp = a
	a = b
	b = tmp
	return a,b

def preprocessing(sample, indexes):
	# Search for missing data
	for i in range(1, indexes.shape[0] - 1, 2):
		#the shift between each couple in indexes array is not only 1, 
		#but sometimes it's also bigger than one (in the 14th column there is also an overlap) 
		if(indexes[i] != indexes[i+1] - 1):
			if (indexes[i+1] > indexes[i]):
				gap = indexes[i+1] - indexes[i] - 1
				# Cut sample vector
				sample = np.concatenate((sample[:indexes[i]], sample[indexes[i+1]:]), axis=0)
				# Shift indexes vector
				indexes[i+1:] = [x - gap for x in indexes[i+1:]]
			else:
				gap = indexes[i] - indexes[i+1] - 1
				sample = np.concatenate((sample[:indexes[i+1]], sample[indexes[i]:]), axis = 0)
				indexes[i], indexes[i+1] = swap_indexes(indexes[i], indexes[i+1])
				indexes[i+1:] = [x - gap for x in indexes[i+1:]]
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

# Create an array filled with zeros, exept the label-th one-->useful for create Y array
def convert_to_one_hot(dictionary_length, label):
	one_hot_array = np.zeros(dictionary_length)
	one_hot_array[label] = 1

	return one_hot_array

# In order to add white noise to the previuous dataset
def add_white_noise(pattern):
	noise = np.random.normal(0,1, np.shape(window))
	pattern += noise
	
	return pattern
	# 0 is the mean of the normal distribution you are choosing from
	# 1 is the standard deviation of the normal distribution
	# window_size is the number of elements you get in array noise

# Add noisy pattern to the original list of tuples
def data_augmentation(tuples, label = None):
	original_tuples = tuples
	for i in range(np.shape(original_tuples)[0]):
		noisy_pattern = add_white_noise(original_tuples[i][0])
		tuples.append([noisy_pattern, original_tuples[i][1]])
	return tuples



# Build the structures
activities_dict = {'RUNNING': 0, 'WALKING': 1, 'JUMPING': 2, 'STNDING': 3, 'SITTING': 4, 'XLYINGX': 5, 'FALLING': 6, 'TRANSUP': 7, 'TRANSDW': 8, 'TRNSACC': 9, 'TRNSDCC': 10, 'TRANSIT': 11}

# Read the .mat file
mat = sio.loadmat('ARS_DLR_DataSet_V2.mat')
mat = {k:v for k, v in mat.items() if k[0] != '_'}
data = pd.DataFrame({k: pd.Series(v[0]) for k, v in mat.items()})

# Window size in seconds
window_size = 10
# Uncomment this to have fixed window length
min_pattern_length = get_pattern_min_length(data)
min_length = int(window_size/0.01)

# List of tuples, each tuple will contain X and Y of pattern i
tuples = []



# Cycle over all the people in the dataset
for column in data:
	# Extrapolate data of a single user
	sample = data[column][0]
	attitude = data[column][1]
	sample_activities = data[column][2][0]
	indexes = np.squeeze(data[column][3]) - 1 # Remove 1 since data are saved in matlab and indexes start from 1 D:

	# Remove time from sample matrix
	sample = sample[:,1:]

	# Change of coordinates to be in global frame
	sample *= attitude[:,1:]

	# Remove unlabeled data
	[sample, indexes] = preprocessing(sample, indexes)

	for i in range(0, indexes.shape[0], 2):
		whole_pattern = sample[indexes[i]:indexes[i+1], :]
		label = activities_dict[sample_activities[int(i/2)][0]]
		shift = 10
		i = 0
		while i + min_pattern_length < whole_pattern.shape[0]:
			window = whole_pattern[i:i+min_pattern_length, :]
			one_hot_Y = convert_to_one_hot(len(activities_dict), label)
			tuples.append([window, one_hot_Y])
			i += shift

		# Add the last window
		one_hot_Y = convert_to_one_hot(len(activities_dict), label)
		tuples.append([whole_pattern[-min_pattern_length:, :], one_hot_Y])

# Add noisy patterns to tuples list
tuples = data_augmentation(tuples)

# Turn tuples into a numpy array
tuples = np.array(tuples)

# Shuffle the list of tuples
np.random.shuffle(tuples)


# Separate X and Y into 2 numpy array
X = [i[0] for i in tuples]
Y = [i[1] for i in tuples]


# Transform list to numpy array for the sake of the computational flexibility
X = np.array(X)
Y = np.array(Y).reshape(np.shape(Y)[0], 12)
print('X shape: ' + str(np.shape(X)))
print('Y shape: ' + str(np.shape(Y)))

# Take 80% of the dataset as training set
trainingNorm = int(np.ceil(X.shape[0]/100*80))

# Divide dataset in training and test set
X_train = X[:trainingNorm, :, :]
Y_train = Y[:trainingNorm, :]
X_test = X[trainingNorm:, :, :]
Y_test = Y[trainingNorm:, :]

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
