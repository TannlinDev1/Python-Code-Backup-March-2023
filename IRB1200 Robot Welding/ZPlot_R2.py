import os
import matplotlib.pyplot as plt
import numpy as np

os.chdir(r"C:\Users\angus.mcallister\Documents\TFF\Robot Profile Following\Calibration Tests\260722")

Z1a = np.loadtxt(fname="Z_1.txt")
Z2a = np.loadtxt(fname="Z_2.txt")
Z3a = np.loadtxt(fname="Z_3.txt")
Z4a = np.loadtxt(fname="Z_4.txt")
Z5a = np.loadtxt(fname="Z_5.txt")

os.chdir(r"C:\Users\angus.mcallister\Documents\TFF\Robot Profile Following\Calibration Tests\270722")

Z1b = np.loadtxt(fname="Z_1.txt")
Z2b = np.loadtxt(fname="Z_2.txt")
Z3b = np.loadtxt(fname="Z_3.txt")
Z4b = np.loadtxt(fname="Z_4.txt")
Z5b = np.loadtxt(fname="Z_5.txt")

t1a = np.linspace(0, len(Z1a)*0.02, len(Z1a))
t2a = np.linspace(0, len(Z2a)*0.02, len(Z2a))
t3a = np.linspace(0, len(Z3a)*0.02, len(Z3a))
t4a = np.linspace(0, len(Z4a)*0.02, len(Z4a))
t5a = np.linspace(0, len(Z5a)*0.02, len(Z5a))

t1b = np.linspace(0, len(Z1b)*0.02, len(Z1b))
t2b = np.linspace(0, len(Z2b)*0.02, len(Z2b))
t3b = np.linspace(0, len(Z3b)*0.02, len(Z3b))
t4b = np.linspace(0, len(Z4b)*0.02, len(Z4b))
t5b = np.linspace(0, len(Z5b)*0.02, len(Z5b))

zm1a = np.mean(Z1a)
zm2a = np.mean(Z2a)
zm3a = np.mean(Z3a)
zm4a = np.mean(Z4a)
zm5a = np.mean(Z5a)

Z1a -= zm1a
Z2a -= zm2a
Z3a -= zm3a
Z4a -= zm4a
Z5a -= zm5a

zm1b = np.mean(Z1b)
zm2b = np.mean(Z2b)
zm3b = np.mean(Z3b)
zm4b = np.mean(Z4b)
zm5b = np.mean(Z5b)

Z1b -= zm1b
Z2b -= zm2b
Z3b -= zm3b
Z4b -= zm4b
Z5b -= zm5b

dZ1a = np.max(abs(Z1a))
dZ2a = np.max(abs(Z2a))
dZ3a = np.max(abs(Z3a))
dZ4a = np.max(abs(Z4a))
dZ5a = np.max(abs(Z5a))

dZ1b = np.max(abs(Z1b))
dZ2b = np.max(abs(Z2b))
dZ3b = np.max(abs(Z3b))
dZ4b = np.max(abs(Z4b))
dZ5b = np.max(abs(Z5b))

plt.figure(1)
plt.plot(t1a, Z1a, label="Test A")
plt.plot(t1b, Z1b, label="Test B")
plt.xlabel("Time (s)")
plt.ylabel("Normal Distance (mm)")
plt.grid()
# plt.title("Segment 1, Max Deviation = " +str(np.round(dZ1a, 2))+ " mm")
plt.title("Segment 1")
plt.legend()

plt.figure(2)
plt.plot(t2a, Z2a, label="Test A")
plt.plot(t2b, Z2b, label="Test B")
plt.xlabel("Time (s)")
plt.ylabel("Normal Distance (mm)")
plt.grid()
# plt.title("Segment 2, Max Deviation = " +str(np.round(dZ2a, 2))+ " mm")
plt.title("Segment 2")
plt.legend()

plt.figure(3)
plt.plot(t3a, Z3a, label="Test A")
plt.plot(t3b, Z3b, label="Test B")
plt.xlabel("Time (s)")
plt.ylabel("Normal Distance (mm)")
plt.grid()
# plt.title("Segment 3, Max Deviation = " +str(np.round(dZ3a, 2))+ " mm")
plt.title("Segment 3")
plt.legend()

plt.figure(4)
plt.plot(t4a, Z4a, label="Test A")
plt.plot(t4b, Z4b, label="Test B")
plt.xlabel("Time (s)")
plt.ylabel("Normal Distance (mm)")
plt.grid()
# plt.title("Segment 4, Max Deviation = " +str(np.round(dZ4a, 2))+ " mm")
plt.title("Segment 4")
plt.legend()

plt.figure(5)
plt.plot(t5a, Z5a, label="Test A")
plt.plot(t5b, Z5b, label="Test B")
plt.xlabel("Time (s)")
plt.ylabel("Normal Distance (mm)")
plt.grid()
# plt.title("Segment 5, Max Deviation = " +str(np.round(dZ5a, 2))+ " mm")
plt.title("Segment 5")
plt.legend()

segments = np.linspace(1,5,5)
zm_a = np.array([zm1a, zm2a, zm3a, zm4a, zm5a])
zm_b = np.array([zm1b, zm2b, zm3b, zm4b, zm5b])

plt.figure(6)
plt.plot(segments, zm_a, label="Test A (15 mm/s)")
plt.plot(segments, zm_b, label="Test B (30 mm/s)")
plt.xlabel("Segment No")
plt.ylabel("Mean Normal Distance (mm)")
plt.grid()
plt.legend()