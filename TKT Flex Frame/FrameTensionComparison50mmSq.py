import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd

os.chdir(r"R:\TFF\Assembly Line\118-03-3000\FEA\ResponseGraphs\50 mm Sq")

# dF_3N_csv = pd.read_csv("dF_3N.csv")
# dS_3N_csv = pd.read_csv("dS_3N.csv")
#
# dF_4N_csv = pd.read_csv("dF_4N.csv")
# dS_4N_csv = pd.read_csv("dS_4N.csv")
#
# dF_6N_csv = pd.read_csv("dF_6N.csv")
# dS_6N_csv = pd.read_csv("dS_6N.csv")

dF_12N_csv = pd.read_csv("dF_12N.csv")
dS_12N_csv = pd.read_csv("dS_12N.csv")

os.chdir(r"R:\TFF\Testing\VG\50 mm Sq Testing")

# dF_VG2 = np.loadtxt("VG_STD.txt", delimiter=",")
dF_VG_HT_ASM = np.loadtxt("VG_HT_ASM.txt", delimiter=",")

# dF_VG2[:,0] -= dF_VG2[0,0]
# dF_VG2[:,1] -= dF_VG2[0,1]
# dF_VG2[:,0]  = abs(dF_VG2[:,0])

dF_VG_HT_ASM[:,0] -= dF_VG_HT_ASM[0,0]
dF_VG_HT_ASM[:,1] -= dF_VG_HT_ASM[0,1]
dF_VG_HT_ASM[:,0]  = abs(dF_VG_HT_ASM[:,0])

plt.figure(1)
# plt.scatter(dS_3N_csv.iloc[3:-1,1], dF_3N_csv.iloc[3:-1,4],color="tab:blue",label="3 N/mm")
# plt.scatter(dS_4N_csv.iloc[7:-1,1], dF_4N_csv.iloc[7:-1,4],color="tab:orange",label="4 N/mm Sim")
# plt.scatter(dS_6N_csv.iloc[8:-1,1], dF_6N_csv.iloc[8:-1,4],color="tab:green", label="6 N/mm")
plt.scatter(dS_12N_csv.iloc[11:-1,1], dF_12N_csv.iloc[11:-1,4],color="tab:pink", label="5.5 N/mm")
# plt.plot(dF_VG2[:,0], dF_VG2[:,1]*9.81,color="tab:olive", label="Standard VG")
# # plt.plot(dF_VG2[:,0], dF_VG2[:,1]*9.81,color="tab:red", label="Standard VG")
# plt.plot(dF_VG_HT[:,0], dF_VG_HT[:,1]*9.81,color="tab:cyan", label="High Tension VG")
plt.plot(dF_VG_HT_ASM[:,0], dF_VG_HT_ASM[:,1]*9.81,color="tab:red", label="ASM High Tension VG")
plt.grid()
plt.legend()
plt.xlabel("Displacement (mm)")
plt.ylabel("Force Feedback (N)")
plt.title("Frame Tension, 150mm Sq (t = 100 um)")
