import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

os.chdir(r"R:\TFF\Back Radius Investigation\FEA\Data")

F_T16_csv = pd.read_csv("F_R15_T16.csv")
F_T20_csv = pd.read_csv("F_R15_T20.csv")
F_T24_csv = pd.read_csv("F_R15_T24.csv")

F_T16 = np.zeros((117-94, 2))
F_T16[:,1] = F_T16_csv.iloc[94:117,4]
F_T16[:,0] = (0.75/0.25)*(F_T16_csv.iloc[94:117, 0]-0.75)

F_T20 = np.zeros((110-89, 2))
F_T20[:,1] = F_T20_csv.iloc[89:110,4]
F_T20[:,0] = (0.7/0.25)*(F_T20_csv.iloc[89:110, 0]-0.75)

F_T24 = np.zeros((105-82, 2))
F_T24[:,1] = F_T24_csv.iloc[82:105,4]
F_T24[:,0] = (0.55/0.25)*(F_T24_csv.iloc[82:105, 0]-0.75)

F_T24_E = np.array([[0, 14.22],
                    [0.101, 12.45],
                    [0.206, 10.875],
                    [0.297, 9.055],
                    [0.401, 7.72],
                    [0.5, 6.135],
                    [0.6, 4.76],
                    [0.704, 3.365],
                    [0.802, 2.11],
                    [0.9, 0.815]])

F_T20_E = np.array([[0, 8.545],
                    [0.106, 7.3],
                    [0.217, 6.38],
                    [0.308, 5.37],
                    [0.406, 4.46],
                    [0.506, 4.12],
                    [0.614, 3.63],
                    [0.708, 3.31],
                    [0.809, 2.81],
                    [1.01, 2.06],
                    [1.107, 1.76],
                    [1.218, 1.34],
                    [1.4, 0.785]])

plt.figure(1)
plt.plot(F_T24[:,0], F_T24[:,1], label="0.6 mm FEA")
plt.plot(F_T20[:,0], F_T20[:,1], label="0.5 mm FEA")
plt.plot(F_T16[:,0], F_T16[:,1], label="0.4 mm FEA")
plt.scatter(F_T24_E[:,0], F_T24_E[:,1], label="0.6 mm Test")
plt.scatter(F_T20_E[:,0], F_T20_E[:,1], label="0.5 mm Test")
plt.legend()
plt.grid()
plt.xlabel("Displacement (mm)")
plt.ylabel("Tension (N/mm)")