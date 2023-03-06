import numpy as np
import matplotlib.pyplot as plt
import os

os.chdir(r"R:\TFF\Coherent Welder")

data1 = np.loadtxt(fname="Test_251022.txt", delimiter=",")
data2 = np.loadtxt(fname="Test_171022.txt", delimiter=",")

t1 = np.linspace(0, len(data1)*0.1, len(data1))
t2 = np.linspace(0, len(data2)*0.1, len(data2))

load1 = np.zeros(len(data1))
load2 = np.zeros(len(data2))

for i in range(0, len(data1)):
    load1[i] = data1[i,0] + data1[i,1] + data1[i,2]

for j in range(0, len(data2)):
    load2[j] = data2[j,0] + data2[j,1] + data2[j,2]

# plt.figure(1)
# plt.plot(t, data1[:,0], label="Load Cell A")
# plt.plot(t, data1[:,1], label="Load Cell A")
# plt.plot(t, data1[:,2], label="Load Cell A")
# plt.xlabel("Time (s)")
# plt.ylabel("Load (N)")
# plt.legend()

plt.figure(2)
plt.plot(t1, load1/584)
plt.plot(t2, load2/584)
plt.xlabel("Time(s)")
plt.ylabel("Tension (N/mm)")