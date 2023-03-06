#Load in packages

import matplotlib.pyplot as plt
import os
import numpy as np

#Find Directory
os.chdir("C:/Users/Tannlin User/Documents/Parallel Kinematic Laser Head Profile/Testing/SS304 Testing/Testing Round 1")

#Import 8 Thou Data

data1 = np.loadtxt(fname="LOG01200to1500.txt", delimiter=',')
data2 = np.loadtxt(fname="LOG0.txt",delimiter=',')
data3 = np.loadtxt(fname="LOG1.txt",delimiter=',')
data4 = np.loadtxt(fname="LOG2.txt",delimiter=',')
data5 = np.loadtxt(fname="LOG3.txt",delimiter=',')

#Append Data to Each Other

DATA_8Thou_tuple = (data1[:,1:3],data2[:,1:3],data3[:,1:3],data4[:,1:3],data5[:,1:3])
DATA_8Thou = np.vstack(DATA_8Thou_tuple)
DATA_8Thou = abs(DATA_8Thou)

#Find Directory
os.chdir("C:/Users/Tannlin User/Documents/Parallel Kinematic Laser Head Profile/Testing/SS304 Testing/Testing 251120")

#Import 6 Thou Data

data1 = np.loadtxt(fname="LOG0.txt",delimiter=',')
data2 = np.loadtxt(fname="LOG1.txt",delimiter=',')
data3 = np.loadtxt(fname="LOG2.txt",delimiter=',')
data4 = np.loadtxt(fname="LOG3.txt",delimiter=',')
data5 = np.loadtxt(fname="LOG4.txt",delimiter=',')
data6 = np.loadtxt(fname="LOG5.txt",delimiter=',')
data7 = np.loadtxt(fname="LOG6.txt",delimiter=',')
data8 = np.loadtxt(fname="LOG7.txt",delimiter=',')

#Find Directory
os.chdir("C:/Users/Tannlin User/Documents/Parallel Kinematic Laser Head Profile/Testing/SS304 Testing/Testing 011220")

#Import 6 Thou Data

data9 = np.loadtxt(fname="LOG0.txt",delimiter=',')
data10 = np.loadtxt(fname="LOG1.txt",delimiter=',')
data11 = np.loadtxt(fname="LOG2.txt",delimiter=',')
data12 = np.loadtxt(fname="LOG3.txt",delimiter=',')
data13 = np.loadtxt(fname="LOG4.txt",delimiter=',')
data14 = np.loadtxt(fname="LOG5.txt",delimiter=',')
data15 = np.loadtxt(fname="LOG6.txt",delimiter=',')

#Import More 6 Thou data

os.chdir("C:/Users/Tannlin User/Documents/Parallel Kinematic Laser Head Profile/Testing/SS304 Testing/Testing up to 091220")

data16 = np.loadtxt(fname="LOG0.txt",delimiter=',')
data17 = np.loadtxt(fname="LOG1.txt",delimiter=',')
data18 = np.loadtxt(fname="LOG2.txt",delimiter=',')
data19 = np.loadtxt(fname="LOG3.txt",delimiter=',')
data20 = np.loadtxt(fname="LOG4.txt",delimiter=',')
data21 = np.loadtxt(fname="LOG5.txt",delimiter=',')
data22 = np.loadtxt(fname="LOG6.txt",delimiter=',')
data23 = np.loadtxt(fname="LOG7.txt",delimiter=',')

#Import even more 6 thou data

os.chdir("C:/Users/Tannlin User/Documents/Parallel Kinematic Laser Head Profile/Testing/SS304 Testing/up to 111220")

data24 = np.loadtxt(fname="LOG0.txt",delimiter=',')
data25 = np.loadtxt(fname="LOG1.txt",delimiter=',')
data26 = np.loadtxt(fname="LOG2.txt",delimiter=',')

#Append Data to Each Other

DATA_6Thou_tuple = (data1[:,1:3],data2[:,1:3],data3[:,1:3],data4[:,1:3],data5[:,1:3],data6[:,1:3],data7[:,1:3],data8[:,1:3],data9[:,1:3],data10[:,1:3],data11[:,1:3],data12[:,1:3],data13[:,1:3],data14[:,1:3],data15[:,1:3],data16[:,1:3],data17[:,1:3],data18[:,1:3],data19[:,1:3],data20[:,1:3],data21[:,1:3],data22[:,1:3],data23[:,1:3],data24[:,1:3],data25[:,1:3],data26[:,1:3])
DATA_6Thou = np.vstack(DATA_6Thou_tuple)
DATA_6Thou = abs(DATA_6Thou)

#Create vectors for number of reversals

N_8Thou = 100*np.linspace(1,len(DATA_8Thou[:,0]),num=len(DATA_8Thou[:,0]))
N_6Thou = 100*np.linspace(1,len(DATA_6Thou[:,0]),num=len(DATA_6Thou[:,0]))

#Create coefficients of 3rd degree polynomial for curve-fitting data

Coefs_KL_8Thou = np.polyfit(N_8Thou,DATA_8Thou[:,0],deg=3)
Coefs_KR_8Thou = np.polyfit(N_8Thou,DATA_8Thou[:,1],deg=3)
Coefs_KL_6Thou = np.polyfit(N_6Thou,DATA_6Thou[:,0],deg=3)
Coefs_KR_6Thou = np.polyfit(N_6Thou,DATA_6Thou[:,1],deg=3)

#Evaluate Line of Best Fit for previously determined coefficients

LoBF_KL_8Thou = np.polyval(Coefs_KL_8Thou,N_8Thou)
LoBF_KR_8Thou = np.polyval(Coefs_KR_8Thou,N_8Thou)
LoBF_KL_6Thou = np.polyval(Coefs_KL_6Thou,N_6Thou)
LoBF_KR_6Thou = np.polyval(Coefs_KR_6Thou,N_6Thou)

#Determine variances for datasets

Var_KL_8Thou = np.var(DATA_8Thou[:,0])
Var_KL_8Thou = round(Var_KL_8Thou,5)
Var_KL_8Thou = str(Var_KL_8Thou)

Var_KR_8Thou = np.var(DATA_8Thou[:,1])
Var_KR_8Thou = round(Var_KR_8Thou,5)
Var_KR_8Thou = str(Var_KR_8Thou)

Var_KL_6Thou = np.var(DATA_6Thou[:,0])
Var_KL_6Thou = round(Var_KL_6Thou,5)
Var_KL_6Thou = str(Var_KL_6Thou)

Var_KR_6Thou = np.var(DATA_6Thou[:,1])
Var_KR_6Thou = round(Var_KR_6Thou,5)
Var_KR_6Thou = str(Var_KR_6Thou)

#Reformat variances for displaying in figure

label_KL_8Thou = "8 Thou Left (V = {})".format(Var_KL_8Thou)
label_KR_8Thou = "8 Thou Right (V = {})".format(Var_KR_8Thou)

label_KL_6Thou = "6 Thou Left (V = {})".format(Var_KL_6Thou)
label_KR_6Thou = "6 Thou Right (V = {})".format(Var_KR_6Thou)

#Plot with x as log axis

plt.semilogx(N_8Thou,LoBF_KL_8Thou,color='r',linestyle='dotted',label=label_KL_8Thou)
plt.semilogx(N_8Thou,LoBF_KR_8Thou,color='b',linestyle='dotted',label=label_KR_8Thou)
plt.semilogx(N_6Thou,LoBF_KL_6Thou,color='r',linestyle='dashdot',label=label_KL_6Thou)
plt.semilogx(N_6Thou,LoBF_KR_6Thou,color='b',linestyle='dashdot',label=label_KR_6Thou)
plt.legend()
plt.xlabel("Number of Cycles")
plt.ylabel("Current (A)")
plt.grid()
plt.show()