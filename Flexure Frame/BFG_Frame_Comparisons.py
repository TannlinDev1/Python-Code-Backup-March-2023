import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd

os.chdir(r"R:\TFF\Testing\TFF\BFG Testing")

URES_FEA = pd.read_csv("URES_FEA.csv")
URES_E1 = np.loadtxt("URES_E1.txt")
URES_E2 = np.loadtxt("URES_E2.txt")
URES_E3 = np.loadtxt("URES_E3.txt")

URES_FEA.iloc[:,0] *= 8

URES_E1[:,0] /= 1000
URES_E1[:,1] /=25

URES_E2[:,0] /= 1000
URES_E2[:,1] /=10

URES_E3[:,0] /= 1000
URES_E3[:,1] /=25

plt.figure(1)
plt.scatter(URES_FEA.iloc[:,1], URES_FEA.iloc[:,0], label="FEA")
plt.scatter(URES_E1[:,0],URES_E1[:,1], label="Sample 1")
plt.scatter(URES_E2[:,0],URES_E2[:,1], label="Sample 2")
plt.scatter(URES_E3[:,0],URES_E3[:,1], label="Sample 3")
plt.grid()
plt.legend()