import os
import numpy as np
import matplotlib.pyplot as plt

os.chdir(r"R:\TFF\Creep Investigation\301\FEA\FEA CAD\Displacement Results")

FvsD_301FH_FEA = np.loadtxt(fname="301FH_F_vs_d.txt", delimiter=",")
FvsD_301FH_A = np.loadtxt(fname="301FH_F_vs_d_A.txt")
FvsD_301FH_FEA_J = np.loadtxt(fname="301FH_F_vs_d_JIG.txt", delimiter=",")
FvsD_301FH_FEA_J2 = np.loadtxt(fname="FvsD_301FH_JIG_2.txt", delimiter=",")

os.chdir(r"R:\TFF\Creep Investigation\420\FEA")

FvsD_420FH_FEA = np.loadtxt(fname="FvsD_420FH_FEA.txt", delimiter=",")
FvsD_420FH_A = np.loadtxt(fname="FvsD_420FH_A.txt")
FvsD_420FH_FEA_J = np.loadtxt(fname="420FH_F_vs_d_JIG.txt", delimiter=",")

FvsD_301FH_FEA[:,1] *= 9.12
FvsD_420FH_FEA[:,1] *= 8.1
FvsD_420FH_FEA_J[:,1] *= 2*202.5/25
FvsD_301FH_FEA_J[:,1] *= 228/25
FvsD_301FH_FEA_J2[:,1] *= 228/25

F_interp_301 = np.array([5.65, 6.07, 6.48, 6.89, 7.26, 7.68, 8.22, 9.12])
F_interp_420 = np.array([5.02, 5.4, 5.76, 6.1, 6.45, 6.83, 7.31, 8.1])

d_interp_301 = np.zeros(len(F_interp_301))
d_interp_420 = np.zeros(len(F_interp_420))

for i in range(len(F_interp_301)):
    d_interp_301[i] = np.interp(F_interp_301[i], FvsD_301FH_FEA_J2[:,1], FvsD_301FH_FEA_J2[:,2])
    d_interp_420[i] = np.interp(F_interp_420[i], FvsD_420FH_FEA_J[:,1], FvsD_420FH_FEA_J[:,2])

plt.figure(1)
plt.plot(FvsD_301FH_FEA[:,2], FvsD_301FH_FEA[:,1], label="FEA",color="r")
plt.plot(FvsD_301FH_A[:,1], FvsD_301FH_A[:,0], label="Analytical",color="g")
plt.plot(FvsD_301FH_FEA_J[:,2], FvsD_301FH_FEA_J[:,1], label="FEA Jigged",color="b")
plt.plot(FvsD_301FH_FEA_J2[:,2], FvsD_301FH_FEA_J2[:,1], label="FEA Jigged 2",color="m")
plt.xlabel("Displacement (mm)")
plt.ylabel("Force per Unit Depth (N)")
plt.legend()
plt.grid()
plt.title("Force Displacement Curve for SS301 FH")

plt.figure(2)
plt.plot(FvsD_420FH_FEA[:,2], FvsD_420FH_FEA[:,1], label="FEA",color="r")
plt.plot(FvsD_420FH_A[:,1], FvsD_420FH_A[:,0], label="Analytical", color="g")
plt.plot(FvsD_420FH_FEA_J[:,2], FvsD_420FH_FEA_J[:,1], label="FEA Jigged", color="b")
plt.scatter(d_interp_420, F_interp_420, color="b")
plt.xlabel("Displacement (mm)")
plt.ylabel("Force per Unit Depth (N)")
plt.legend()
plt.grid()
plt.title("Force Displacement Curve for SS420 FH")