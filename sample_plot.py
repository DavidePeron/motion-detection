import scipy.io as sio
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Read the .mat file
mat = sio.loadmat('ARS_DLR_DataSet_V2.mat')
mat = {k:v for k, v in mat.items() if k[0] != '_'}
data = pd.DataFrame({k: pd.Series(v[0]) for k, v in mat.items()})
sample = data['ARS_Susanna_Test_StSit_Sensor_Left'][0]
attitude= data['ARS_Susanna_Test_StSit_Sensor_Left'][1]
sample = sample*attitude

# Accelerometer
plt.figure()
plt.subplot(221)
plt.plot(sample[:,0], sample[:,1])
plt.grid()
plt.title("X-axis acceleration")
plt.xlabel("Time")
plt.ylabel("Acceleration")

plt.subplot(222)
plt.plot(sample[:,0], sample[:,2])
plt.grid()
plt.title("Y-axis acceleration")
plt.xlabel("Time")
plt.ylabel("Acceleration")

plt.subplot(223)
plt.plot(sample[:,0], sample[:,3])
plt.grid()
plt.title("Z-axis acceleration")
plt.xlabel("Time")
plt.ylabel("Acceleration")

plt.tight_layout()
plt.savefig("acceleration_susanna.pdf")

# Gyroscope
plt.figure()
plt.subplot(221)
plt.plot(sample[:,0], np.abs(sample[:,4]))
plt.grid()
plt.title("X-axis angular velocity")
plt.xlabel("Time")
plt.ylabel("Angular velocity")

plt.subplot(222)
plt.plot(sample[:,0], np.abs(sample[:,5]))
plt.grid()
plt.title("Y-axis angular velocity")
plt.xlabel("Time")
plt.ylabel("Angular velocity")

plt.subplot(223)
plt.plot(sample[:,0], np.abs(sample[:,6]))
plt.grid()
plt.title("Z-axis angular velocity")
plt.xlabel("Time")
plt.ylabel("Angular velocity")

plt.tight_layout()
plt.savefig("angular_velocity_susanna.pdf")

# Magnetic field
plt.figure()
plt.subplot(221)
plt.plot(sample[:,0], np.abs(sample[:,7]))
plt.grid()
plt.title("X-axis magnetic field")
plt.xlabel("Time")
plt.ylabel("Magnetic field")

plt.subplot(222)
plt.plot(sample[:,0], np.abs(sample[:,8]))
plt.grid()
plt.title("Y-axis magnetic field")
plt.xlabel("Time")
plt.ylabel("Magnetic field")

plt.subplot(223)
plt.plot(sample[:,0], np.abs(sample[:,9]))
plt.grid()
plt.title("Z-axis magnetic field")
plt.xlabel("Time")
plt.ylabel("Magnetic field")

plt.tight_layout()
plt.savefig("magnetic_field_susanna.pdf")
