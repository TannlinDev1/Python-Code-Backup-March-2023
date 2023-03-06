import numpy as np
import os
import matplotlib.pyplot as plt

os.chdir(r"R:\TFF\Spot Weld\Sheet Weld 300721\Surface Profile")

Z18_V3_t450_P1900 = np.loadtxt(fname="Z18_V3_t450_P1900.txt")
Z18_V3_t650_P1900 = np.loadtxt(fname="Z18_V3_t650_P1900.txt")
Z22_V3_t450_P1900 = np.loadtxt(fname="Z22_V3_t450_P1900.txt")

dZ_t650 = np.max(Z18_V3_t650_P1900[:,1]) - np.min(Z18_V3_t450_P1900[:,1])
plt.figure(1)

plt.plot(Z18_V3_t450_P1900[:,0], Z18_V3_t450_P1900[:,1], label="450 us, 18 mm")
plt.plot(Z18_V3_t650_P1900[:,0], Z18_V3_t650_P1900[:,1], label="650 us, 18 mm ")
plt.plot(Z22_V3_t450_P1900[:,0], Z22_V3_t450_P1900[:,1], label="450 us, 19 mm")
plt.legend()
plt.grid()
plt.xlabel("X Distance (um)")
plt.ylabel("Z Distance (um)")
