import numpy as np
import matplotlib.pyplot as plt
import os

os.chdir(r"R:\TFF\Testing\VG\BFG Tests")

F_VG = np.loadtxt("VG_STD.txt")
F_VG_HT = np.loadtxt("VG_HT.txt")

Coefs_VG = np.polyfit(F_VG[:,0], F_VG[:,1], 2)
Coefs_VG_HT = np.polyfit(F_VG_HT[:,0], F_VG_HT[:,1], 2)

d_VG = np.linspace(F_VG[0,0], F_VG[-1,0], 1000)
d_VG_HT = np.linspace(F_VG_HT[0,0], F_VG_HT[-1,0], 1000)

LoBF_VG = np.polyval(Coefs_VG, d_VG)
LoBF_VG_HT = np.polyval(Coefs_VG_HT, d_VG_HT)

plt.scatter(F_VG[:,0], F_VG[:,1], label="Standard")
plt.scatter(F_VG_HT[:,0], F_VG_HT[:,1], label="High Tension")
plt.legend()
plt.plot(d_VG, LoBF_VG, color="tab:blue", linestyle="dashed")
plt.plot(d_VG_HT, LoBF_VG_HT, color="tab:orange", linestyle="dashed")
plt.xlabel("Displacement (mm)")
plt.ylabel("Tension (N/mm)")
plt.grid()
plt.title("Vectorguard Slack Testing")