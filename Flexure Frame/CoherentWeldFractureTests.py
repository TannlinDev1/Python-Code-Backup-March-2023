import os
import numpy as np
import matplotlib.pyplot as plt

os.chdir(r"R:\TFF\Welding\Frame Weld Tests\Frame Tension Test Jig\Results\130522")

test = np.loadtxt(fname="200kg_TestA120522.txt")
test2 = np.loadtxt(fname="200kg_TestA120522_2.txt")

dt = 1e-2
N = int(len(test))
N_2 = int(len(test2))

t_f = N*dt
t_f2 = N_2*dt

t = np.linspace(0, t_f, N)
t_2 = np.linspace(0, t_f2, N_2)

plt.plot(t, test/25, label="Sample 1")
plt.plot(t_2, test2/25, label="Sample 2")
plt.xlabel("Time (s)")
plt.ylabel("Tension (N/mm)")
