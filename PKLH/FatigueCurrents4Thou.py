#Load in packages

import matplotlib.pyplot as plt
import os
import numpy as np

#Find Directory
os.chdir("C:/Users/Tannlin User/Documents/Parallel Kinematic Laser Head Profile/Testing/Testing up to 090121")

#Import 4 Thou Data

data1 = np.loadtxt(fname="LOG0.txt",delimiter=',')

#Find Directory
os.chdir("C:/Users/Tannlin User/Documents/Parallel Kinematic Laser Head Profile/Testing/Testing up to 130121")

#Import 4 Thou Data

data2 = np.loadtxt(fname="LOG0.txt",delimiter=',')
data3 = np.loadtxt(fname="LOG1.txt",delimiter=',')
data4 = np.loadtxt(fname="LOG2.txt",delimiter=',')

#Find Directory
os.chdir("C:/Users/Tannlin User/Documents/Parallel Kinematic Laser Head Profile/Testing/Testing up to 150121")

data5 = np.loadtxt(fname="LOG0.txt",delimiter=",")
data6 = np.loadtxt(fname="LOG1.txt",delimiter=",")
data7 = np.loadtxt(fname="LOG2.txt",delimiter=",")

#Find Directory
os.chdir("C:/Users/Tannlin User/Documents/Parallel Kinematic Laser Head Profile/Testing/Testing up to 180121")

data8 = np.loadtxt(fname="LOG0.txt",delimiter=",")
data9 = np.loadtxt(fname="LOG1.txt",delimiter=",")
data10 = np.loadtxt(fname="LOG2.txt",delimiter=",")
data11 = np.loadtxt(fname="LOG3.txt",delimiter=",")

#Find Directory
os.chdir("C:/Users/Tannlin User/Documents/Parallel Kinematic Laser Head Profile/Testing/Testing up to 200121")

data12 = np.loadtxt(fname="LOG0.txt",delimiter=",")
data13 = np.loadtxt(fname="LOG1.txt",delimiter=",")

os.chdir("C:/Users/Tannlin User/Documents/Parallel Kinematic Laser Head Profile/Testing/Testing up to 210121")

data14 = np.loadtxt(fname="LOG0.txt",delimiter=",")
data15 = np.loadtxt(fname="LOG1.txt",delimiter=",")

#Append Data to Each Other

DATA_4Thou_tuple = (data1[:,1:3],data2[:,1:3],data3[:,1:3],data4[:,1:3],data5[:,1:3],data6[:,1:3],data7[:,1:3],data8[:,1:3],data9[:,1:3],data10[:,1:3],data11[:,1:3],data12[:,1:3],data13[:,1:3],data14[:,1:3],data15[:,1:3])

DATA_4Thou = np.vstack(DATA_4Thou_tuple)
DATA_4Thou = abs(DATA_4Thou)

#Create vectors for number of reversals

N_4Thou = 100*np.linspace(1,len(DATA_4Thou[:,0]),num=len(DATA_4Thou[:,0]))

#Create coefficients of 3rd degree polynomial for curve-fitting data

Coefs_KL_4Thou = np.polyfit(N_4Thou,DATA_4Thou[:,0],deg=3)
Coefs_KR_4Thou = np.polyfit(N_4Thou,DATA_4Thou[:,1],deg=3)

#Evaluate Line of Best Fit for previously determined coefficients

LoBF_KL_4Thou = np.polyval(Coefs_KL_4Thou,N_4Thou)
LoBF_KR_4Thou = np.polyval(Coefs_KR_4Thou,N_4Thou)

#Determine variances for datasets

Var_KL_4Thou = np.var(DATA_4Thou[:,0])
Var_KL_4Thou = round(Var_KL_4Thou,5)
Var_KL_4Thou = str(Var_KL_4Thou)

Var_KR_4Thou = np.var(DATA_4Thou[:,1])
Var_KR_4Thou = round(Var_KR_4Thou,5)
Var_KR_4Thou = str(Var_KR_4Thou)

#Reformat variances for displaying in figure

label_KL_4Thou = "4 Thou Left (V = {})".format(Var_KL_4Thou)
label_KR_4Thou = "4 Thou Right (V = {})".format(Var_KR_4Thou)

#Plot with x as log axis

plt.semilogx(N_4Thou,LoBF_KL_4Thou,color='r',linestyle='dashdot',label=label_KL_4Thou)
plt.semilogx(N_4Thou,LoBF_KR_4Thou,color='b',linestyle='dashdot',label=label_KR_4Thou)
plt.legend()
plt.xlabel("Number of Cycles")
plt.ylabel("Current (A)")
plt.grid()
plt.show()