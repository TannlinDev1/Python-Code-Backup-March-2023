import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd

os.chdir(r"R:\TFF\Fully Sized Frame Jig\301FH Setup\Flexure FEA\Response Graphs")

ex_a = pd.read_csv("Response Graph-1.csv")
F_NS_R2 = np.zeros((14,2))
F_NS_R2[:,0] = ex_a.iloc[:,0]
F_NS_R2[:,1] = ex_a.iloc[:,4]
F_NS_R2[:,0] *= 1.58
F_NS_R2[:,1] *= 2

# F_H = np.loadtxt(fname="DvsF_Hole.csv")
# F_N = np.loadtxt(fname="DvsF_Nom.csv")
# F_H2 = np.loadtxt(fname="DvsF_Hole_2.txt")
# F_H3 = np.loadtxt(fname="DvsF_Hole_3.txt")
# F_H4 = np.loadtxt(fname="DvsF_Hole_4.txt")
F_C = np.loadtxt(fname="F_C.txt", delimiter = ",")
F_C[:,1] *= 200
F_F = np.loadtxt(fname="F_F.txt", delimiter = ",")
F_F *= 2
F_R2 = np.loadtxt(fname="DvsF_2R.txt", delimiter=",")
F_R2[:,1] *= 200

# F_H3 *= 2
# F_H4 *= 2

d = np.array([0.01, 0.03, 0.07, 0.15 ,0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95, 1])*1.47

plt.figure(1)
# plt.scatter(d, F_F, label="Free")
# plt.scatter(F_C[:,2], F_C[:,1], label="Single Rad")
plt.plot(F_R2[:,2], F_R2[:,1], label="Double Rad")
plt.plot(F_NS_R2[:,0], F_NS_R2[:,1], label="Double Rad (Nippon)")
plt.xlabel("Displacement (mm)")
plt.ylabel("Force (N)")
plt.grid()
plt.legend()

