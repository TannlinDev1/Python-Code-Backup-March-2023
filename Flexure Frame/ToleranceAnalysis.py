import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

os.chdir(r"R:\TFF\Testing\TFF\Tolerance Analysis")

F_65D_csv = pd.read_csv("F_65D.csv")
F_65D = np.zeros((50, 2))
F_65D[:,0] = F_65D_csv.iloc[1:51, 1]
F_65D[:,1] = F_65D_csv.iloc[1:51, 0]*10

F_75_csv = pd.read_csv("F_75D.csv")

F_75D = np.zeros((13,2))
F_75D[:,1] = F_75_csv.iloc[:,0]*10.2
F_75D[:,0] = F_75_csv.iloc[:,1]

d_tol = np.array([3.4,
                  3.16,
                2.86,
                2.57,
                2.28,
                1.98])

d_tol_75 = np.array([1.234,
                     0.965,
                     0.745,
                     0.555,
                     0.415,
                     0.315])

F_tol = np.interp(d_tol, F_65D[:,0], F_65D[:,1])
F_tol_75 = np.interp(d_tol_75, F_75D[:,0], F_75D[:,1])
plt.scatter(F_65D[:,0], F_65D[:,1])
plt.scatter(F_75D[:,0], F_75D[:,1])
