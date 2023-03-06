import numpy as np
import matplotlib.pyplot as plt
import os

os.chdir(r"C:\Users\angus.mcallister\Documents\TFF")

data1 = np.loadtxt(fname="StressStrainData.txt")
data2 = np.zeros(len(data1))

data2 = data1[:,0]*184.73e3

plt.figure(1)
plt.plot(data1[:,0]*100, data1[:,1], label="Experiment")
plt.plot(data1[:,0]*100, data2, label="Elastic")
plt.xlabel("Strain (%)")
plt.ylabel("Stress (MPa)")
plt.legend()
plt.grid()