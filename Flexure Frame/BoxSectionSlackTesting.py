import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np

os.chdir(r"C:\Users\angus.mcallister\Documents\TFF\Box Section Slack Testing")

F_RES = pd.read_csv("F_RES.csv")
U_RES = pd.read_csv("U_RES.csv")
U_RES *= -1

F_RES_2 = pd.read_csv("F_RES (160D, T).csv")
U_RES_2 = pd.read_csv("U_RES (160D, T).csv")

F_RES_3 = pd.read_csv("F_RES (150D, T).csv")
U_RES_3 = pd.read_csv("U_RES (150D, T).csv")

# S1 = np.loadtxt(fname="Sample1.txt")
# S2 = np.loadtxt(fname="Sample2.txt")
# S3 = np.loadtxt(fname="Sample3.txt")

# plt.plot(-U_RES.iloc[79:95,1]+U_RES.iloc[79,1], F_RES.iloc[79:95,4]/18.6, label="Simulated")
# plt.plot(-U_RES.iloc[79:95,1]+U_RES.iloc[79,1], F_RES.iloc[79:95,4]/18.6, label="150D, L")
# plt.plot(-U_RES_2.iloc[79:88,1]+U_RES_2.iloc[79,1], F_RES_2.iloc[79:88,4]/18.6, label="160D, T")
# plt.plot(-U_RES_3.iloc[79:88,1]+U_RES_3.iloc[79,1], F_RES_3.iloc[79:88,4]/18.6, label="150D, T")
plt.plot(U_RES_3.iloc[:,1], F_RES_3.iloc[:,4])
# plt.plot(S1[:,0], S1[:,1], label="Sample 1")
# plt.plot(S2[:,0], S2[:,1], label="Sample 2")
# plt.plot(S3[:,0], S3[:,1], label="Sample 3")
plt.xlabel("Slack Displacement (mm)")
plt.ylabel("Tension (N/mm)")

plt.legend()