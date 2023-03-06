import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

# This script predicts the loss in tension due to the flexure pushers being pitched apart.
# Currently assumes an infinitely stiff frame

os.chdir(r"R:\TFF\Flexure Actuation\Deflection Data")

U_100 = pd.read_csv("URES_100mm.csv")
U_55 = pd.read_csv("URES_55mm.csv")
U_25 = pd.read_csv("URES_25mm.csv")
U_F = pd.read_csv("URES_FLEXURE.csv")

U_100.iloc[:,0] *= 225
U_55.iloc[:,0] *= 225
U_25.iloc[:,0] *= 225
U_F.iloc[:,0] *= 7

U_100_ave = np.mean(U_100.iloc[:,1])
U_55_ave = np.mean(U_55.iloc[:,1])
U_25_ave = np.mean(U_25.iloc[:,1])

F_100 = np.interp(U_100_ave, U_F.iloc[:,1], U_F.iloc[:,0])
F_55 = np.interp(U_55_ave, U_F.iloc[:,1], U_F.iloc[:,0])
F_25 = np.interp(U_25_ave, U_F.iloc[:,1], U_F.iloc[:,0])

dF_100 = 100*(7-F_100)/7
dF_55 = 100*(7-F_55)/7
dF_25 = 100*(7-F_25)/7

plt.figure(1)
plt.plot(U_25.iloc[:,0], U_25.iloc[:,1], label="25 mm")
plt.plot(U_55.iloc[:,0],U_55.iloc[:,1],label="55 mm")
plt.plot(U_100.iloc[:,0], U_100.iloc[:,1], label="100 mm")
plt.legend()
plt.hlines(U_25_ave, U_25.iloc[0,0], U_25.iloc[-1,0], colors="tab:blue", linestyles="dashed")
plt.hlines(U_55_ave, U_55.iloc[0,0], U_55.iloc[-1,0],colors="tab:orange", linestyles="dashed")
plt.hlines(U_100_ave, U_100.iloc[0,0], U_100.iloc[-1,0],colors="tab:green", linestyles="dashed")pitch_arr = np.linspace(25, 100, 100)
Coefs = np.polyfit(pitch, dF, 2)

LoBF = np.polyval(Coefs, pitch_arr)

plt.figure(2)
plt.scatter(pitch, dF)
plt.plot(pitch_arr, LoBF)
plt.xlabel("Pusher Pitch (mm)")
plt.ylabel("Loss in Tension (N)")
plt.grid()
plt.xlabel("Length (mm)")
plt.ylabel("Total Displacement (mm)")
plt.grid()

pitch = np.array([25, 55, 100])
dF = np.array([dF_25, dF_55, dF_100])

