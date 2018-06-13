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
 
 
X = []
Y = []
#create X_train and Y_train, X_test and Y_test
X_train = []
Y_train = []
X_test = []
Y_test = []
 
# Cycle over all the people in the dataset
for column in data:
 
  sample = data[column][0]
  attitude = data[column][1]
  sample_activities = data[column][2][0]
  indexes = np.squeeze(data[column][3]) - 1 # Remove 1 since data are saved in matlab and indexes start from 1 D:
 
  # Change of coordinates to be in global frame
  sample[:,1:] *= attitude[:,1:]
 
  # Build a matrix for X and a vector for Y with the same number of elements to be fed by the CNN
  # We use list since the append method is faster
  X.append(sample[:min_length,1:])
 
  Y_single = []
  j = 0
  for i in range(sample_activities.size):
    number_of_repetitions = indexes[j+1] - indexes[j] + 1
    activity = activities_dict[sample_activities[i][0]]
    Y_single += [activity] * number_of_repetitions
    j = j+2
 
  Y_single = Y_single[:min_length]
  Y.append(Y_single)
 
# Transform list to numpy array for the sake of the computational flexibility
X = np.array(X)
Y = np.array(Y)
Y = Y.reshape(Y.shape[0], Y.shape[1], 1)
 
# Take 80% of the dataset as training set
trainingNorm = int(np.ceil(X.shape[0]/100*80))
 
# Divide dataset in training and test set
X_train = X[:trainingNorm, :, :]
Y_train = Y[:trainingNorm, :, :]
X_test = X[trainingNorm:, :, :]
Y_test = Y[trainingNorm:, :, :]



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
