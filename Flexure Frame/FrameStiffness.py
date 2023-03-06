import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import pandas as pd
import os

# This script determines the theoretical deflection of a flexure/frame combination which generates a linearly
# distributed tension across the full length of a 736 mm frame

# Theoretical:

E = 193000 #Young's Modulus (MPa)
I = 11595 #2nd Moment of Area (mm^4)
L = 554 #Frame Length (736 mm section from start of mitre)
N_steps = 1000

x = np.linspace(0,L, N_steps)
K = np.zeros(N_steps)
F = np.zeros(N_steps)
K_FR = np.zeros(N_steps)
y = np.zeros(N_steps)
y_FR = np.zeros(N_steps)
y_FL = np.zeros(N_steps)
M = np.zeros(N_steps)
stress = np.zeros(N_steps)

K_FL = 5.79 #Flexure stiffness (N/mm_y/mm_x)
F_S = 5.5 #Desired tension on foil (N/mm_x)
d_NA = 15 #Max distance from neutral axis (mm)
s_y = 210 #Yield stress of frame material (MPa)

K_FR = np.zeros(N_steps)

for i in range(0,N_steps):
    K_FR[i] = (24*E*I/(x[i]**2*(L-x[i])**2)) #Frame Stiffness (N/mm)
    K[i] = 1/((1/K_FR[i]) + 1/K_FL) #Sectional spring constant for springs in series (N/mm/mm)
    y[i] = F_S/K[i] # Frame/Flexure Displacement in y Direction
    y_FR[i] = F_S/K_FR[i] #Frame Displacement in y direction
    y_FL[i] = y[i] - y_FR[i] #Flexure Displacement in y direction

    M[i] = F_S*(6*L*x[i] - 6*x[i]**2 - L**2)/12
    stress[i] = M[i]*d_NA/I

plt.figure(1)
plt.plot(x, y, label="Total Defelection")
plt.plot(x, y_FR, label="Frame Deflection")
plt.plot(x, y_FL, label="Flexure Deflection")
plt.legend()

# FEA Data Analysis:

# Import FEA data
#
# os.chdir(r"R:\TFF\Flexure Actuation\Deflection Data\736mm Frame")
#
# y1_FEA = pd.read_csv("UZ_736mm.csv")
#
# y2_FEA = y1_FEA*0
#
# y1_FEA.iloc[:,0] *= L/2
# y1_FEA.iloc[:,1] *= -1
#
# #mirror y_FEA
#
# for j in range(0,len(y2_FEA.iloc[:,0])):
#     j_inverse = len(y2_FEA.iloc[:,0]) - j
#     y2_FEA.iloc[j,0] = y1_FEA.iloc[j,0]+L/2
#     y2_FEA.iloc[j,1] = y1_FEA.iloc[j_inverse-1,1]
#
# y_FEA = y1_FEA.append(y2_FEA)
#
# plt.figure(1)
# # plt.plot(x,K)
# plt.plot(x, y, label="Frame")
# # plt.plot(y_FEA.iloc[:,0], y_FEA.iloc[:,1], label="FEA")
# plt.xlabel("Length (mm)")
# plt.ylabel("Frame Stiffness (N/mm)")
# plt.ylabel("Displacement (mm)")
# plt.title("Frame Stiffness - 736mm Frame, 8 N/mm")
# plt.grid()
#
# plt.figure(2)
# plt.plot(x, stress, label="Frame Stress")
# plt.hlines(s_y, x[0], x[-1], colors="tab:orange", linestyles="dashed", label="Yield Stress")
# plt.legend()
# plt.hlines(-s_y, x[0], x[-1], colors="tab:orange", linestyles="dashed")
# plt.xlabel("Length (mm)")
# plt.ylabel("Stress (MPa)")
# plt.title("Maximum Stress - 736mm Frame, 5.2 N/mm")
# plt.grid()
#
# # plt.figure(2)
# # plt.plot(x, y_FR)