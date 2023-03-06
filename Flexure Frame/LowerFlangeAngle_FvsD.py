import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np

os.chdir(r"R:\TFF\Flexure Actuation\4 Bar Linkage\BFG Testing\FEA Results")

URES_60D = pd.read_csv("URES_60D.csv")
URES_65D = pd.read_csv("URES_65D.csv")
URES_70D = pd.read_csv("URES_70D.csv")
URES_75D = pd.read_csv("URES_75D.csv")

URES_60D.iloc[:,0] *= 10
URES_65D.iloc[:,0] *= 10
URES_70D.iloc[:,0] *= 10
URES_75D.iloc[:,0] *= 8

URES_75D_E = np.array([[0.19, 3.14],
                       [0.43, 4.47],
                       [0.67, 6.97]])

err_75D = np.array([[0.34, 0.022, 1.21],
                   [0.34, 0.022, 1.09]])

URES_70D_E = np.array([[0.47, 1.81],
                       [0.7, 3.54],
                       [1.52, 7.7]])

err_70D = np.array([[0.35, 0.45, 0.57],
                   [0.24, 0.75, 0.73]])

URES_65D_E = np.array([[0.91, 4.21],
                       [1.14, 6.45],
                       [2.22, 8.21]])

err_65D = np.array([[0.14, 1.62, 0.455],
                   [0.17, 1.37, 0.455]])

URES_60D_E = np.array([[0.87, 3.59],
                       [1.31, 6.41],
                       [2.91, 8.51]])

err_60D = np.array([[0.24, 0.31, 0.1],
                   [0.44, 0.32, 0.1]])

theta = np.array([60, 65, 70, 75])
F_max_FEA = np.array([11.6, 10.5, 10.46, 10.2])
F_max_E = np.array([8.51, 8.21, 7.3, 6.26])
F_max_err = np.array([[0.1, 0.455, 0.16, 0.59],
                      [0.1, 0.455, 0.24, 0.38]])

Coefs_FEA = np.polyfit(theta, F_max_FEA, 1)
Coefs_E = np.polyfit(theta, F_max_E, 1)

theta_arr = np.linspace(60, 75, 100)

LoBF_FEA = np.polyval(Coefs_FEA, theta_arr)
LoBF_E = np.polyval(Coefs_E, theta_arr)

plt.figure(5)
plt.errorbar(theta, F_max_E, F_max_err, fmt="o", elinewidth=3, capsize=4,label="Experiment")
plt.scatter(theta, F_max_FEA, color="tab:orange",label="FEA")
plt.legend()
plt.plot(theta_arr, LoBF_E, color="tab:blue", linestyle="dashed")
plt.plot(theta_arr, LoBF_FEA, color="tab:orange", linestyle="dashed")
plt.xlabel("Lower Flange Angle (deg)")
plt.ylabel("Maximum Tension (N/mm)")
plt.grid()
plt.title("Flexure Tension Variation")

#
# plt.figure(1)
# plt.plot(URES_75D.iloc[:,1], URES_75D.iloc[:,0],label="FEA")
# plt.errorbar(URES_75D_E[:,0], URES_75D_E[:,1], yerr=err_75D, fmt="o", elinewidth=3, capsize=0,label="Test")
# plt.xlabel("Displacement (mm)")
# plt.ylabel("Reaction Force (N)")
# plt.grid()
# plt.legend()
# plt.title("75 Degree Flexure Test")
#
# plt.figure(2)
# plt.plot(URES_70D.iloc[:,1], URES_70D.iloc[:,0],label="FEA")
# plt.errorbar(URES_70D_E[:,0], URES_70D_E[:,1], yerr=err_70D, fmt="o", elinewidth=3, capsize=0,label="Test")
# plt.xlabel("Displacement (mm)")
# plt.ylabel("Reaction Force (N)")
# plt.grid()
# plt.legend()
# plt.title("70 Degree Flexure Test")
#
# plt.figure(3)
# plt.plot(URES_65D.iloc[:,1], URES_65D.iloc[:,0],label="FEA")
# plt.errorbar(URES_65D_E[:,0], URES_65D_E[:,1], yerr=err_65D, fmt="o", elinewidth=3, capsize=0,label="Test")
# plt.xlabel("Displacement (mm)")
# plt.ylabel("Reaction Force (N)")
# plt.grid()
# plt.legend()
# plt.title("65 Degree Flexure Test")
#
plt.figure(4)
plt.plot(URES_60D.iloc[:,1], URES_60D.iloc[:,0],label="FEA")
plt.errorbar(URES_60D_E[:,0], URES_60D_E[:,1], yerr=err_60D, fmt="o", elinewidth=3, capsize=0,label="Test")
plt.xlabel("Displacement (mm)")
plt.ylabel("Reaction Force (N)")
plt.grid()
plt.legend()
plt.title("60 Degree Flexure Test")
#
# plt.figure(5)