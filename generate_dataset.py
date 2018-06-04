import scipy.io as sio
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Build the structures
activities_dict = {'RUNNING': 0, 'WALKING': 1, 'JUMPING': 2, 'STNDING': 3, 'SITTING': 4, 'XLYINGX': 5, 'FALLING': 6, 'TRANSUP': 7, 'TRANSDW': 8, 'TRANSACC': 9, 'TRANSDCC': 10, 'TRANSIT': 11}
X_train = []

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

# # Get patterns
# # for i in range(1,10):
# # X_train_partition = sample[indexes[0]:indexes[1]+1,1:] # +1 because I need to take all the elements between indexes[0] to indexes[1] included
# # X_train.append(X_train_partition.T.reshape((indexes[1]+1-indexes[0])*9,1))
# # print(X_train_partition.T.reshape((indexes[1]+1-indexes[0])*9,1).shape)
# for i in sample[:,0]:
#     print(i)
# # print(sample[:,0])
# # print(sample_activities)
# # print(indexes)
