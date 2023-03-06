import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np

# This is a graphing script for visualising how a varied diameter hole affects tension developed
# a flexure

#Import FEA Data

os.chdir(r"R:\TFF\Flexure Tension Relief Testing\Sim Results")

URES_NOM = pd.read_csv("URES_NOM.csv")
URES_5MM = pd.read_csv("URES_5MM.csv")
URES_6MM = pd.read_csv("URES_6MM.csv")
URES_7MM = pd.read_csv("URES_7MM.csv")
URES_8MM = pd.read_csv("URES_8MM.csv")

URES_NOM.iloc[:,0] *= 8
URES_5MM.iloc[:,0] *= 8
URES_6MM.iloc[:,0] *= 8
URES_7MM.iloc[:,0] *= 8
URES_8MM.iloc[:,0] *= 8

# F_5MM = np.interp(URES_NOM.iloc[-1,1], URES_5MM.iloc[:,1], URES_5MM.iloc[:,0])
# F_6MM = np.interp(URES_NOM.iloc[-1,1], URES_6MM.iloc[:,1], URES_6MM.iloc[:,0])
# F_7MM = np.interp(URES_NOM.iloc[-1,1], URES_7MM.iloc[:,1], URES_7MM.iloc[:,0])
# F_8MM = np.interp(URES_NOM.iloc[-1,1], URES_8MM.iloc[:,1], URES_8MM.iloc[:,0])

F_5MM = 7.2
F_6MM = 5.73
F_7MM = 5.35
F_8MM = 4.45

F_NOM = 10.2

dF = np.zeros((4,1))

dF[0,0] = 100*(F_NOM-F_5MM)/F_NOM
dF[1,0] = 100*(F_NOM-F_6MM)/F_NOM
dF[2,0] = 100*(F_NOM-F_7MM)/F_NOM
dF[3,0] = 100*(F_NOM-F_8MM)/F_NOM

D = np.linspace(5,8,4)

URES_NOM_E = np.array([[0.19, 3.14],
                       [0.43, 4.47],
                       [0.67, 6.26]])

URES_5MM_E = np.array([[0.19, 2.15],
                       [0.43, 2.9],
                       [0.67, 4.57]])

err_5MM = np.array([[0.18, 0.224, 0.5],
                   [0.18, 0.206, 0.78]])

URES_6MM_E = np.array([[0.19, 1.74],
                       [0.43, 2.13],
                       [0.67, 3.62]])

err_6MM = np.array([[0.18, 0.292, 0.17],
                   [0.22, 0.22, 0.14]])

URES_7MM_E = np.array([[0.19, 1.44],
                       [0.43, 2.17],
                       [0.67, 2.95]])

err_7MM = np.array([[0.33, 0.249, 0.5],
                   [0.26, 0.267, 0.44]])

URES_8MM_E = np.array([[0.19, 1.16],
                       [0.43, 1.22],
                       [0.67, 2.32]])

err_8MM = np.array([[0.11, 0.129, 0.28],
                   [0.09, 0.241, 0.53]])

dF2 = np.zeros((4,1))
dF2[0,0] = 100*(6.26-URES_5MM_E[2,1])/6.26
dF2[1,0] = 100*(6.26-URES_6MM_E[2,1])/6.26
dF2[2,0] = 100*(6.26-URES_7MM_E[2,1])/6.26
dF2[3,0] = 100*(6.26-URES_8MM_E[2,1])/6.26

Coefs_dF_D = np.polyfit(D, dF, 1)
Coefs_dF2_D = np.polyfit(D, dF2, 1)

D_arr = np.linspace(D[0],D[-1], 100)

LoBF_dF2_D = np.polyval(Coefs_dF2_D, D_arr)
LoBF_dF_D = np.polyval(Coefs_dF_D, D_arr)

plt.figure(1)
plt.plot(URES_5MM.iloc[1:7,1],URES_5MM.iloc[1:7,0],label="5 mm")
# plt.scatter(URES_5MM_E[:,0], URES_5MM_E[:,1])
plt.errorbar(URES_5MM_E[:, 0], URES_5MM_E[:, 1], yerr = err_5MM, fmt="o", elinewidth=3, capsize=10)
plt.xlabel("Displacement (mm)")
plt.ylabel("Tension (N/mm)")
plt.title("Flexure Tension Relief Comparison")
plt.grid()
plt.title("5mm Hole")

plt.figure(2)
plt.plot(URES_6MM.iloc[1:7,1],URES_6MM.iloc[1:7,0],label="6 mm")
plt.errorbar(URES_6MM_E[:, 0], URES_6MM_E[:, 1], yerr = err_6MM, fmt="o", elinewidth=3, capsize=10)
plt.scatter(URES_6MM_E[:,0], URES_6MM_E[:,1])
plt.xlabel("Displacement (mm)")
plt.ylabel("Tension (N/mm)")
plt.grid()
plt.title("6mm Hole")

plt.figure(3)
plt.plot(URES_7MM.iloc[1:6,1],URES_7MM.iloc[1:6,0],label="7 mm")
plt.errorbar(URES_7MM_E[:, 0], URES_7MM_E[:, 1], yerr = err_7MM, fmt="o", elinewidth=3, capsize=10)
plt.scatter(URES_7MM_E[:,0], URES_7MM_E[:,1])
plt.xlabel("Displacement (mm)")
plt.ylabel("Tension (N/mm)")
plt.title("Flexure Tension Relief Comparison")
plt.grid()
plt.title("7mm Hole")

plt.figure(4)
plt.plot(URES_8MM.iloc[1:6,1],URES_8MM.iloc[1:6,0],label="8 mm")
plt.errorbar(URES_8MM_E[:, 0], URES_8MM_E[:, 1], yerr = err_8MM, fmt="o", elinewidth=3, capsize=10)
plt.scatter(URES_8MM_E[:,0], URES_8MM_E[:,1])
plt.xlabel("Displacement (mm)")
plt.ylabel("Tension (N/mm)")
plt.title("Flexure Tension Relief Comparison")
plt.grid()
plt.title("8mm Hole")

plt.figure(5)
plt.scatter(D, dF, color="tab:orange", label="FEA")
plt.scatter(D, dF2, color="tab:blue",label="Experiment")
plt.plot(D_arr, LoBF_dF2_D, color="tab:blue", linestyle="dashed")
plt.plot(D_arr, LoBF_dF_D, color="tab:orange", linestyle="dashed")
plt.xlabel("Hole Diameter (mm)")
plt.ylabel("Loss in Tension (%)")
plt.title("Tension Relief Design - 10 mm Pitch")
plt.legend()
plt.grid()
