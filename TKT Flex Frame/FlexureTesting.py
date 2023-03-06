import os
import numpy as np
import matplotlib.pyplot as plt

os.chdir(r"E:\Users\Tannlin User\Documents\TKT Flexure Frame\Sim Results\Double 0.6mm Flexure")

DoubleFlex_06_Sim = np.loadtxt(fname="TXvsF_S.txt")
DoubleFlex_06_E1 = np.loadtxt(fname="TXvsF_E1.txt")
DoubleFlex_06_E2 = np.loadtxt(fname="TXvsF_E2.txt")
DoubleFlex_06_E3 = np.loadtxt(fname="TXvsF_E3.txt")

DoubleFlex_06_Sim[:,0] = DoubleFlex_06_Sim[:,0]*6
DoubleFlex_06_E1[:,0] = DoubleFlex_06_E1[:,0]*np.cos(0.1174607)/25
DoubleFlex_06_E2[:,0] = DoubleFlex_06_E2[:,0]*np.cos(0.1174607)/25
DoubleFlex_06_E3[:,0] = DoubleFlex_06_E3[:,0]*np.cos(0.1174607)/25

deg = 2

Coefs_06_E1 = np.polyfit(DoubleFlex_06_E1[:,0], DoubleFlex_06_E1[:,1], deg = deg)
Coefs_06_E2 = np.polyfit(DoubleFlex_06_E2[:,0], DoubleFlex_06_E2[:,1], deg = deg)
Coefs_06_E3 = np.polyfit(DoubleFlex_06_E3[:,0], DoubleFlex_06_E3[:,1], deg = deg)

F_E1 = np.linspace(0,np.max(DoubleFlex_06_E1[:,0]))
LoBF_06_E1 = np.polyval(Coefs_06_E1, F_E1)

F_E2 = np.linspace(0, np.max(DoubleFlex_06_E2[:,0]))
LoBF_06_E2 = np.polyval(Coefs_06_E2, F_E2)

F_E3 = np.linspace(0, np.max(DoubleFlex_06_E3[:,0]))
LoBF_06_E3 = np.polyval(Coefs_06_E3, F_E3)

plt.figure(1)
plt.plot(DoubleFlex_06_Sim[:,0], DoubleFlex_06_Sim[:,1], color="r",label="Simulated")
plt.plot(F_E1, LoBF_06_E1, color="g", label="Sample 1")
plt.plot(F_E2, LoBF_06_E2, color="b", label="Sample 2")
plt.plot(F_E3, LoBF_06_E3, color="m", label="Sample 3")
plt.legend()
plt.scatter(DoubleFlex_06_E1[:,0], DoubleFlex_06_E1[:,1], color="g")
plt.scatter(DoubleFlex_06_E2[:,0], DoubleFlex_06_E2[:,1], color="b")
plt.scatter(DoubleFlex_06_E3[:,0], DoubleFlex_06_E3[:,1], color="m")
plt.grid()
plt.ylabel("Displacement (mm)")
plt.xlabel("Force (N)")
plt.title("2x 0.6 mm SS304 Flexures")