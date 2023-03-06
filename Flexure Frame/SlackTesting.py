import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np

# Slack testing data analysis and visualisation

# Load all FEA data

os.chdir(r"R:\TFF\Testing Results\TFF\Slack Testing")

F_75D_csv = pd.read_csv("F_75D.csv")
F_75D = np.zeros((17,2))

F_75D[:,0] = 2000*(F_75D_csv.iloc[74:91,0]-0.75)
F_75D[:,1] = F_75D_csv.iloc[74:91,4]

F_70D_csv = pd.read_csv("F_70D.csv")
F_70D = np.zeros((21,2))

F_70D[:,0] = 2000*(F_70D_csv.iloc[74:95,0]-0.75)
F_70D[:,1] = F_70D_csv.iloc[74:95,4]

F_65D_csv = pd.read_csv("F_65D.csv")
F_65D = np.zeros((20,2))

F_65D[:,0] = (600/0.25)*(F_65D_csv.iloc[77:97,0]-0.75)
F_65D[:,1] = F_65D_csv.iloc[77:97,4]

F_60D_csv = pd.read_csv("F_60D.csv")
F_60D = np.zeros((20,2))

F_60D[:,0] = (600/0.25)*(F_60D_csv.iloc[77:97,0]-0.75)
F_60D[:,1] = F_60D_csv.iloc[77:97,4]

F_50D_csv = pd.read_csv("F_50D.csv")

F_50D = np.zeros((16,2))
F_50D[:,0] = (1000/0.25)*(F_50D_csv.iloc[74:90,0]-0.75)
F_50D[:,1] = F_50D_csv.iloc[74:90,4]

# Load all experimental data

F_75D_E1 = np.loadtxt("F_75D_E1.txt")
F_75D_E2 = np.loadtxt("F_75D_E2.txt")
F_75D_E3 = np.loadtxt("F_75D_E3.txt")

F_70D_E1 = np.loadtxt("F_70D_E1.txt")
F_70D_E2 = np.loadtxt("F_70D_E2.txt")
F_70D_E3 = np.loadtxt("F_70D_E3.txt")

F_65D_E1 = np.loadtxt("F_65D_E1.txt")
F_65D_E2 = np.loadtxt("F_65D_E2.txt")

F_60D_E1 = np.loadtxt("F_60D_E1.txt")
F_60D_E2 = np.loadtxt("F_60D_E2.txt")

plt.figure(1)
plt.plot(F_75D[:,0], F_75D[:,1],label="75 Degrees")
plt.plot(F_70D[:,0], F_70D[:,1],label="70 Degrees")
plt.plot(F_65D[:,0], F_65D[:,1],label="65 Degrees")
plt.plot(F_60D[:,0], F_60D[:,1], label="60 Degrees")
# plt.plot(F_50D[:,0], F_50D[:,1], label="50 Degrees")
plt.xlabel("Displacement (um)")
plt.ylabel("Tension (N/mm)")
plt.grid()
plt.legend()
plt.title("FEA Comparison")

plt.figure(2)
plt.scatter(F_75D_E1[:,1], F_75D_E1[:,0], label="Sample 1")
plt.scatter(F_75D_E2[:,1], F_75D_E2[:,0], label="Sample 2")
plt.scatter(F_75D_E3[:,1], F_75D_E3[:,0], label="Sample 3")
plt.scatter(F_75D[:,0], F_75D[:,1],label="FEA")
plt.xlabel("Displacement (um)")
plt.ylabel("Tension (N/mm)")
plt.grid()
plt.legend()
plt.title("75 Degree Results")

plt.figure(3)
plt.scatter(F_70D_E1[:,1], F_70D_E1[:,0], label="Sample 1")
plt.scatter(F_70D_E2[:,1], F_70D_E2[:,0], label="Sample 2")
plt.scatter(F_70D_E3[:,1], F_70D_E3[:,0], label="Sample 3")
plt.scatter(F_70D[:,0], F_70D[:,1],label="FEA")
plt.xlabel("Displacement (um)")
plt.ylabel("Tension (N/mm)")
plt.grid()
plt.legend()
plt.title("70 Degree Results")

plt.figure(4)
plt.scatter(F_65D_E1[:,1], F_65D_E1[:,0], label="Sample 1")
plt.scatter(F_65D_E2[:,1], F_65D_E2[:,0], label="Sample 2")
plt.scatter(F_65D[:,0], F_65D[:,1],label="FEA")
plt.xlabel("Displacement (um)")
plt.ylabel("Tension (N/mm)")
plt.grid()
plt.legend()
plt.title("65 Degree Results")

plt.figure(5)
plt.scatter(F_60D_E1[:,1], F_60D_E1[:,0], label="Sample 1")
plt.scatter(F_60D_E2[:,1], F_60D_E2[:,0], label="Sample 2")
plt.scatter(F_60D[:,0], F_60D[:,1],label="FEA")
plt.xlabel("Displacement (um)")
plt.ylabel("Tension (N/mm)")
plt.grid()
plt.legend()
plt.title("60 Degree Results")