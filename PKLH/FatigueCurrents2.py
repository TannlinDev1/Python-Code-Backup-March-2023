#Load in packages

import matplotlib.pyplot as plt
import os
import numpy as np

#Find Directory
os.chdir("C:/Users/Tannlin User/Documents/Parallel Kinematic Laser Head Profile/Testing/Testing up to 281220")

#Import 6 Thou Data

data1 = np.loadtxt(fname="LOG0.txt",delimiter=',')
data2 = np.loadtxt(fname="LOG1.txt",delimiter=',')

#Find Directory
os.chdir("C:/Users/Tannlin User/Documents/Parallel Kinematic Laser Head Profile/Testing/Testing up to 020121")

#Import 6 Thou Data

data3 = np.loadtxt(fname="LOG0.txt",delimiter=',')
data4 = np.loadtxt(fname="LOG1.txt",delimiter=',')
data5 = np.loadtxt(fname="LOG2.txt",delimiter=',')
data6 = np.loadtxt(fname="LOG3.txt",delimiter=',')

#Find Directory
os.chdir("C:/Users/Tannlin User/Documents/Parallel Kinematic Laser Head Profile/Testing/Testing up to 030121")

#Import 6 Thou Data

data7 = np.loadtxt(fname="LOG0.txt",delimiter=',')
data8 = np.loadtxt(fname="LOG1.txt",delimiter=',')

#Import More 6 Thou data

os.chdir("C:/Users/Tannlin User/Documents/Parallel Kinematic Laser Head Profile/Testing/Testing up to 060121")

data9 = np.loadtxt(fname="LOG0.txt",delimiter=',')
data10 = np.loadtxt(fname="LOG1.txt",delimiter=',')
data11 = np.loadtxt(fname="LOG2.txt",delimiter=',')
data12 = np.loadtxt(fname="LOG3.txt",delimiter=',')

#Append Data to Each Other

DATA_6Thou_tuple = (data1[:,1:3],data2[:,1:3],data3[:,1:3],data4[:,1:3],data5[:,1:3],data6[:,1:3],data7[:,1:3],data8[:,1:3],data9[:,1:3],data10[:,1:3],data11[:,1:3],data12[:,1:3])
DATA_6Thou = np.vstack(DATA_6Thou_tuple)
DATA_6Thou = abs(DATA_6Thou)

#Create vectors for number of reversals

N_6Thou = 100*np.linspace(1,len(DATA_6Thou[:,0]),num=len(DATA_6Thou[:,0]))

#Create coefficients of 3rd degree polynomial for curve-fitting data

Coefs_KL_6Thou = np.polyfit(N_6Thou,DATA_6Thou[:,0],deg=3)
Coefs_KR_6Thou = np.polyfit(N_6Thou,DATA_6Thou[:,1],deg=3)

#Evaluate Line of Best Fit for previously determined coefficients

LoBF_KL_6Thou = np.polyval(Coefs_KL_6Thou,N_6Thou)
LoBF_KR_6Thou = np.polyval(Coefs_KR_6Thou,N_6Thou)

#Determine variances for datasets

Var_KL_6Thou = np.var(DATA_6Thou[:,0])
Var_KL_6Thou = round(Var_KL_6Thou,5)
Var_KL_6Thou = str(Var_KL_6Thou)

Var_KR_6Thou = np.var(DATA_6Thou[:,1])
Var_KR_6Thou = round(Var_KR_6Thou,5)
Var_KR_6Thou = str(Var_KR_6Thou)

#Reformat variances for displaying in figure

label_KL_6Thou = "6 Thou Left (V = {})".format(Var_KL_6Thou)
label_KR_6Thou = "6 Thou Right (V = {})".format(Var_KR_6Thou)

#Plot with x as log axis

plt.semilogx(N_6Thou,LoBF_KL_6Thou,color='r',linestyle='dashdot',label=label_KL_6Thou)
plt.semilogx(N_6Thou,LoBF_KR_6Thou,color='b',linestyle='dashdot',label=label_KR_6Thou)
plt.legend()
plt.xlabel("Number of Cycles")
plt.ylabel("Current (A)")
plt.grid()
plt.show()