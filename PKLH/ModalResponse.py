import numpy as np
import matplotlib.pyplot as plt
import os

os.chdir(r"C:\Users\Tannlin User\Documents\Parallel Kinematic Laser Head Profile\Simulations\Harmonic Response")

MaxMag_8Thou = np.loadtxt(fname="8ThouModalResponse.txt")
MaxMag_6Thou = np.loadtxt(fname="6ThouModalResponse.txt")

# Had to divide displacement at 1628 for 0.15 mm by 100 cos it messed up the graph

plt.figure(1)
plt.plot(MaxMag_6Thou[:,0],MaxMag_6Thou[:,1],label="0.15 mm")
plt.plot(MaxMag_8Thou[:,0],MaxMag_8Thou[:,1],label="0.2 mm")
plt.grid()
plt.legend()
plt.xlabel("Frequency (Hz)")
plt.ylabel("Maximum Magnitude Displacement (mm)")
plt.show()