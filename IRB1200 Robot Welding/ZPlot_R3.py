import os
import matplotlib.pyplot as plt
import numpy as np

os.chdir(r"C:\Users\angus.mcallister\Documents\TFF\Robot Profile Following\Calibration Tests\101122")
Z = np.loadtxt(fname="Z.txt")

f = 0.01
# t = np.linspace(0, len(Z)*f, len(Z))

Z1 = np.loadtxt(fname="Z_1.txt")
Z2 = np.loadtxt(fname="Z_2.txt")
Z3 = np.loadtxt(fname="Z_3.txt")

t1 = np.linspace(0, len(Z1)*f, len(Z1))
t2 = np.linspace(0, len(Z2)*f, len(Z2))
t3 = np.linspace(0, len(Z3)*f, len(Z3))

zm1 = np.mean(Z1)
zm2 = np.mean(Z2)
zm3 = np.mean(Z3)

Z1 -= zm1
Z2 -= zm2
Z3 -= zm3

plt.figure(1)
plt.plot(t1, Z1, label="Test A")
plt.xlabel("Time (s)")
plt.ylabel("Normal Distance (mm)")
plt.grid()
plt.title("Segment 1")

plt.figure(2)
plt.plot(t2, Z2, label="Test A")
plt.xlabel("Time (s)")
plt.ylabel("Normal Distance (mm)")
plt.grid()
plt.title("Segment 2")

plt.figure(3)
plt.plot(t3, Z3, label="Test A")
plt.xlabel("Time (s)")
plt.ylabel("Normal Distance (mm)")
plt.grid()
plt.title("Segment 3")
