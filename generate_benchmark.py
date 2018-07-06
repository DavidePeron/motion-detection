import scipy.io as sio
import pandas as pd
import numpy as np
import h5py
import matplotlib.pyplot as plt
import logging

from keras import layers
from keras.layers import Input, ZeroPadding2D, Conv2D, ZeroPadding1D, Conv1D, BatchNormalization, Activation, Flatten, Dense
from keras.layers import AveragePooling2D, MaxPooling1D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D
from keras.models import Model, load_model
from keras import regularizers
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input
from utility import *


import keras.backend as K
K.set_image_data_format('channels_last')


# Build the structures
activities_dict = {'RUNNING': 0, 'WALKING': 1, 'JUMPING': 2, 'STNDING': 3, 'SITTING': 4, 'XLYINGX': 5, 'FALLING': 6, 'TRANSUP': 7, 'TRANSDW': 8, 'TRNSACC': 9, 'TRNSDCC': 10}#, 'TRANSIT': 11}

# Read the .mat file
mat = sio.loadmat('ARS_DLR_Benchmark_Data_Set.mat')
mat = {k:v for k, v in mat.items() if k[0] != '_'}
data = pd.DataFrame({k: pd.Series(v[0]) for k, v in mat.items()})

# How many shifts in the sample
counter = 0


right_predictions = 0
total_predictions = 0
#total_transit = 0
#right_transit = 0
activity_recognizer = load_model('activity_recognizer.h5')

# Cycle over all the people in the dataset
for column in data:
	# Extrapolate data of a single user
	sample = data[column][0]
	attitude = data[column][1]
	sample_activities = data[column][2][0]
	indexes = np.squeeze(data[column][3]) - 1 # Remove 1 since data are saved in matlab and indexes start from 1 D:

	window_size = 27
	left_limit = 0
	shift = 5

	# Remove time from sample and attitude matrix
	sample = sample[:,1:]
	attitude = attitude[:,1:]

	# Turn coordinates into global frame
	sample = data_to_global_frame(sample,attitude)

	# Poiner to the label
	pointer = 0

	# Shift the window over all the sample until the right limit of the window is less than the sample size
	while(left_limit + window_size <= sample.shape[0]):
		counter += 1
		current_window = sample[left_limit:left_limit + window_size, :]
		current_window = current_window.reshape(1, current_window.shape[0], current_window.shape[1])
		prediction = activity_recognizer.predict(current_window)
		predicted_label = np.argmax(prediction)
		# Check whenever the label of true output is changed
		# Takes in account only windows completely contained in a pattern
		if(left_limit + window_size <= indexes[pointer*2+1]):
			total_predictions += 1
			true_label = activities_dict[sample_activities[pointer][0]]
			if(predicted_label == true_label):
				right_predictions += 1

		elif(pointer + 1 < sample_activities.shape[0]):
			if(left_limit >= indexes[(pointer+1) * 2]):
				# If left_limit is greater of the left limit of the pattern in the indexes, then increase pointer
				pointer += 1
				total_predictions += 1
				true_label = activities_dict[sample_activities[pointer][0]]
				print(sample_activities[pointer][0] + " ----------- " + str(true_label))
				print("Predicted label: " + str(predicted_label))
				if(predicted_label == true_label):
					right_predictions += 1
			# else:
			# 	# Transitions, not recognized
			# 	true_label = 11
			# 	total_predictions += 1
			# 	total_transit += 1
			# 	if(predicted_label == true_label):
			# 		right_transit += 1

		# if(predicted_label == true_label):
		# 	right_predictions += 1

		left_limit += shift

print("Number of windows: " + str(counter))
print("Total predictions: " + str(total_predictions))
print("Total transitions: " + str(total_transit))
print("Right transitions: " + str(right_transit))
print("Prediction accuracy: " + str(right_predictions/total_predictions))
