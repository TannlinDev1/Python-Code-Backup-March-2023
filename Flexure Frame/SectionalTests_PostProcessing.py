import numpy as np
import os
import matplotlib.pyplot as plt

os.chdir(r"R:\TFF\Flexure Actuation\Sectional Tests R2\Results\Creep Testing")

F1 = np.loadtxt(fname="Creep_010722.txt")
F2 = np.loadtxt(fname="Creep_060722.txt")

F1 /= 139.64
F2 /= 139.64

t1 = np.linspace(0, len(F1), len(F1))
t2 = np.linspace(0, len(F2), len(F2))

plt.figure(1)
plt.plot(t1/60, F1, label="Test 1")
plt.plot(t2/60, F2, label="Test 2")
plt.xlabel("Time (hr)")
plt.ylabel("Tension (N/mm)")
plt.legend()