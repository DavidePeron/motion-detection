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
sample = data['ARS_Susanna_Test_StSit_Sensor_Left'][0]
attitude = data['ARS_Susanna_Test_StSit_Sensor_Left'][1]
sample_activities = data['ARS_Susanna_Test_StSit_Sensor_Left'][2]
indexes = np.squeeze(data['ARS_Susanna_Test_StSit_Sensor_Left'][3]) - 1 # Remove 1 since data are saved in matlab and indexes start from 1 D:

# Change of coordinates to be in global frame
sample[:,1:] *= attitude[:,1:]

#create X_train and Y_train
X_train = np.array(sample[:,1:])
Y_train = np.array([])

j = 0
for i in range(sample_activities[0].size):
	number_of_repetitions = indexes[j+1] - indexes[j] + 1
	Y_train = np.append(Y_train, np.repeat(sample_activities[0][i],number_of_repetitions))
	j = j+2
