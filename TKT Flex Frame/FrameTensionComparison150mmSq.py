import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd

os.chdir(r"R:\TFF\Assembly Line\118-03-3000\FEA\ResponseGraphs\150 mm Sq")

dF_3N_csv = pd.read_csv("dF_3N.csv")
dS_3N_csv = pd.read_csv("dS_3N.csv")

dF_4N_csv = pd.read_csv("dF_4N.csv")
dS_4N_csv = pd.read_csv("dS_4N.csv")

dF_5N_csv = pd.read_csv("dF_5N.csv")
dS_5N_csv = pd.read_csv("dS_5N.csv")

dF_12N_csv = pd.read_csv("dF_12N.csv") #this says 12N/mm but is actually closer to 5.35 (as indicated by sectional tests on HT VG)
dS_12N_csv = pd.read_csv("dS_12N.csv")

# dF_TFF_csv = np.loadtxt("TFF_LOG.txt", delimiter=",")
# dF_TFF2 = np.loadtxt("TFF_2.txt", delimiter=",")

os.chdir(r"R:\TFF\Testing Results\TFF")

dF_TFF3 = np.loadtxt("TFF_3.txt", delimiter=",")

os.chdir(r"R:\TFF\Testing Results\MPM")

dF_MPML = np.loadtxt("MPM_L.txt", delimiter=",")
dF_MPMS = np.loadtxt("MPM_S.txt", delimiter=",")
dF_MPMS2 = np.loadtxt("MPM_S2.txt", delimiter=",")

os.chdir(r"R:\TFF\Testing Results\VG\150 mm Sq Testing")

dF_VG2 = np.loadtxt("VG_2.txt", delimiter=",")
dF_VG_HT = np.loadtxt("VG_HT.txt", delimiter=",")
dF_VG_HT_ASM = np.loadtxt("VG_HT_ASM.txt", delimiter=",")

dF_TFF3[:,0] -= dF_TFF3[0,0]
dF_TFF3[:,1] -= dF_TFF3[0,1]
dF_TFF3[:,0]  = abs(dF_TFF3[:,0])

dF_MPML[:,0] -= dF_MPML[0,0]
dF_MPML[:,1] -= dF_MPML[0,1]
dF_MPML[:,0]  = abs(dF_MPML[:,0])

dF_MPMS[:,0] -= dF_MPMS[0,0]
dF_MPMS[:,1] -= dF_MPMS[0,1]
dF_MPMS[:,0]  = abs(dF_MPMS[:,0])

dF_MPMS2[:,0] -= dF_MPMS2[0,0]
dF_MPMS2[:,1] -= dF_MPMS2[0,1]
dF_MPMS2[:,0]  = abs(dF_MPMS2[:,0])

# dF_VG[:,0] -= dF_VG[0,0]
# dF_VG[:,1] -= dF_VG[0,1]
# dF_VG[:,0]  = abs(dF_VG[:,0])

dF_VG2[:,0] -= dF_VG2[0,0]
dF_VG2[:,1] -= dF_VG2[0,1]
dF_VG2[:,0]  = abs(dF_VG2[:,0])

dF_VG_HT[:,0] -= dF_VG_HT[0,0]
dF_VG_HT[:,1] -= dF_VG_HT[0,1]
dF_VG_HT[:,0]  = abs(dF_VG_HT[:,0])

dF_VG_HT_ASM[:,0] -= dF_VG_HT_ASM[0,0]
dF_VG_HT_ASM[:,1] -= dF_VG_HT_ASM[0,1]
dF_VG_HT_ASM[:,0]  = abs(dF_VG_HT_ASM[:,0])

plt.figure(1)
# plt.scatter(dS_3N_csv.iloc[3:-1,1], dF_3N_csv.iloc[3:-1,4],color="tab:blue",label="3 N/mm")
plt.scatter(dS_4N_csv.iloc[7:19,1], dF_4N_csv.iloc[7:19,4],color="tab:green",label="4 N/mm")
# plt.scatter(dS_12N_csv.iloc[12:22,1], dF_12N_csv.iloc[12:22,4],color="tab:orange", label="5 N/mm")
# plt.plot(dF_TFF_csv[:,0], dF_TFF_csv[:,1]*9.81,color="tab:red",  label="TFF")
# plt.plot(dF_TFF2[:,0], dF_TFF2[:,1]*9.81,color="tab:blue",  label="TFF 2")
plt.plot(dF_TFF3[:,0], dF_TFF3[:,1]*9.81,color="tab:red",  label="TFF Prototype")
plt.plot(dF_MPML[:,0], dF_MPML[:,1]*9.81,color="tab:purple", label="Large MPM")
# plt.plot(dF_MPMS[:,0], dF_MPMS[:,1]*9.81,color="tab:pink", label="Small MPM")
# plt.plot(dF_MPMS2[:,0], dF_MPMS2[:,1]*9.81,color="tab:blue", label="Small MPM 2")
plt.plot(dF_VG2[:,0], dF_VG2[:,1]*9.81,color="tab:olive", label="Standard VG")
# # plt.plot(dF_VG2[:,0], dF_VG2[:,1]*9.81,color="tab:red", label="Standard VG")
plt.plot(dF_VG_HT[:,0], dF_VG_HT[:,1]*9.81,color="tab:cyan", label="High Tension VG")
plt.plot(dF_VG_HT_ASM[:,0], dF_VG_HT_ASM[:,1]*9.81,color="tab:blue", label="ASM High Tension VG")
plt.grid()
plt.legend()
plt.xlabel("Displacement (mm)")
plt.ylabel("Force Feedback (N)")