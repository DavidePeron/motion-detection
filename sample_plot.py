import scipy.io as sio
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utility import *

labels_name = ['Running', 'Walking', 'Jumping', 'Standing', 'Sitting', 'Lying', 'Falling', 'Up', 'Down', 'Accelerating', 'Decelerating']
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
plt.xlabel(r"Time [$s$]")
plt.ylabel(r"Acceleration magnitude [$m/s^2$]")

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
plt.ylabel(r"Angular velocity magnitude [$deg/s$]")

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
plt.xlabel(r"Time [$s$]")
plt.ylabel(r"Magnetic field magnitude [$mGauss$]")

plt.tight_layout()
plt.savefig("magnetic field_susanna.pdf")


# Precision and Recall

# Read the .mat file
# mat = sio.loadmat('ARS_DLR_Benchmark_Data_Set.mat')
# mat = {k:v for k, v in mat.items() if k[0] != '_'}
# data = pd.DataFrame({k: pd.Series(v[0]) for k, v in mat.items()})
# for column in data:
# 	activities = data[column][2]
# 	print(activities)

df_labels = pd.read_csv("labels_benchmark.csv", sep="\t", header=None, names=['True', 'Predicted'])
true_labels = df_labels["True"]
predicted_labels = df_labels["Predicted"]

num_labels = 11
precision = np.zeros(num_labels)
recall = np.zeros(num_labels)
true_positives = np.zeros(num_labels)
positives = np.zeros(num_labels)
false_negatives = np.zeros(num_labels)
# Iterate for each label (0 to 10)
for i in range(0, num_labels):
	# Select the labels predicted as i
	predicted_i = np.argwhere(predicted_labels == i)
	positives[i] = len(predicted_i)
	for j in predicted_i:
		if(true_labels[int(j.squeeze())] == i):
			true_positives[i] += 1

	# Search for false negatives i.e. when the true label is i and the CNN predicted a different label
	true_i = np.argwhere(true_labels == i)
	for j in true_i:
		if(predicted_labels[int(j.squeeze())] != i):
			false_negatives[i] += 1
	# Check if total positives for label i is not equal to zero to avoid division by zero errors
	if(positives[i] != 0):
		precision[i] = true_positives[i] / positives[i]
	if(true_positives[i] + false_negatives[i] != 0):
		recall[i] = true_positives[i] / (true_positives[i] + false_negatives[i])

# Precision
fig, ax = plt.subplots(1,1)
x = range(0,num_labels)
ax.plot(x, precision, linewidth = 0.7, marker='o')
plt.grid()
plt.xlabel(r"Activities")
plt.ylabel("Precision")
# Set number of ticks for x-axis
ax.set_xticks(x)
# Set ticks labels for x-axis
ax.set_xticklabels(labels_name, rotation='vertical')
ax.set_yticks(np.linspace(0, 1, 11))
plt.tight_layout()
plt.savefig("precision.pdf")

# Recall
fig, ax = plt.subplots(1,1)
x = range(0,num_labels)
ax.plot(x, recall, linewidth = 0.7, marker='o')
plt.grid()
plt.xlabel(r"Activities")
plt.ylabel("Recall")
# Set number of ticks for x-axis
ax.set_xticks(x)
# Set ticks labels for x-axis
ax.set_xticklabels(labels_name, rotation='vertical')
ax.set_yticks(np.linspace(0, 1, 11))
plt.tight_layout()
plt.savefig("recall.pdf")
