import scipy.io as sio
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utility import *

# Read the .mat file
mat = sio.loadmat('ARS_DLR_DataSet_V2.mat')
mat = {k:v for k, v in mat.items() if k[0] != '_'}
data = pd.DataFrame({k: pd.Series(v[0]) for k, v in mat.items()})
sample = data['ARS_Susanna_Test_StSit_Sensor_Left'][0]
attitude = data['ARS_Susanna_Test_StSit_Sensor_Left'][1]
activities = data['ARS_Susanna_Test_StSit_Sensor_Left'][2]
indexes = data['ARS_Susanna_Test_StSit_Sensor_Left'][3]


#Save time from sample array
time = sample[:,0]

# Remove time from sample and attitude arrays
sample = sample[:,1:]
attitude = attitude[:,1:]

# Turn data into global frame
sample = data_to_global_frame(sample,attitude)

# To writing in LaTex 
plt.rc('text', usetex = True)



# Transform samples to extract modules 
modules = np.zeros((sample.shape[0], 3)) 
for i in range(0, sample.shape[0]): 
	acc_module = np.sqrt(np.sum(np.square(sample[i,0:3])))
	w_module = np.sqrt(np.sum(np.square(sample[i,3:6])))
	mag_module = np.sqrt(np.sum(np.square(sample[i,6:9]))) 
	modules[i] = np.array([acc_module, w_module, mag_module])


# Accelerotemer
plt.figure()
plt.plot(time, modules[:,0], linewidth = 0.7)
for i in range(0, indexes.shape[1], 2):
	plt.hold()
	plt.axvline(x = indexes[0,i]*0.01 + time[0], color = 'k', linestyle = '-.', linewidth = 0.5)
plt.axvline(x = indexes[0,indexes.shape[1]-1]*0.01 + time[0], color = 'k', linestyle = '-.', linewidth = 0.5)

plt.grid()
plt.xlabel(r"Time [s]")
plt.ylabel(r"Acceleration magnitude [m/s^2]")

plt.tight_layout()
plt.savefig("acceleration_susanna.pdf")

# Gyroscope
plt.figure()
plt.plot(time, modules[:,1], linewidth = 0.7)
for i in range(0, indexes.shape[1], 2):
	plt.hold()
	plt.axvline(x = indexes[0,i]*0.01 + time[0], color = 'k', linestyle = '-.', linewidth = 0.5)
plt.axvline(x = indexes[0,indexes.shape[1]-1]*0.01 + time[0], color = 'k', linestyle = '-.', linewidth = 0.5)

plt.grid()
plt.xlabel(r"Time [s]")
plt.ylabel(r"Angular velocity magnitude")# [rad/s]")

plt.tight_layout()
plt.savefig("angular_velocity_susanna.pdf")


# Magnetic field
plt.figure()
plt.plot(time, modules[:,2], linewidth = 0.7)
for i in range(0, indexes.shape[1], 2):
	plt.hold()
	plt.axvline(x = indexes[0,i]*0.01 + time[0], color = 'k', linestyle = '-.', linewidth = 0.5)
plt.axvline(x = indexes[0,indexes.shape[1]-1]*0.01 + time[0], color = 'k', linestyle = '-.', linewidth = 0.5)

plt.grid()
plt.xlabel(r"Time [s]")
plt.ylabel(r"Magnetic field magnitude")#[Gauss]")

plt.tight_layout()
plt.savefig("magnetic field_susanna.pdf")


