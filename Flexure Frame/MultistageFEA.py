import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

os.chdir(r"R:\TFF\FEA\Multistage Frame Sims")

F_RES_1 = pd.read_csv("F_RES_1.csv")
F_RES_2 = pd.read_csv("F_RES_2.csv")

plt.figure(1)
plt.plot(F_RES_1.iloc[:,0], F_RES_1.iloc[:,4]/325.6, label="Sim 1")
plt.plot(F_RES_2.iloc[:,0], F_RES_2.iloc[:,7]/325.6, label="Sim 2")
plt.xlabel("Time (s)")
plt.ylabel("Tension (N/mm)")
plt.grid()
plt.legend()