#Load in packages

import matplotlib.pyplot as plt
import os
import numpy as np

# Find Directory

os.chdir("C:/Users/Tannlin User/Documents/Parallel Kinematic Laser Head Profile/Testing/SS304 Testing/Testing 251120")

#Import 6 Thou SS304 Data

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

DATA_6Thou_tuple = (data1[:,1:3],data2[:,1:3],data3[:,1:3],data4[:,1:3],data5[:,1:3],data6[:,1:3],data7[:,1:3],data8[:,1:3],data9[:,1:3],data10[:,1:3],data11[:,1:3],data12[:,1:3],data13[:,1:3],data14[:,1:3],data15[:,1:3],data16[:,1:3],data17[:,1:3],data18[:,1:3],data19[:,1:3],data20[:,1:3],data21[:,1:3],data22[:,1:3],data23[:,1:3],data24[:,1:3],data25[:,1:3],data26[:,1:3])
DATA_6Thou = np.vstack(DATA_6Thou_tuple)
DATA_6Thou_304 = abs(DATA_6Thou)

#Import 6 Thou SS301 Data

os.chdir(r"C:\Users\Tannlin User\Documents\Parallel Kinematic Laser Head Profile\Testing\SS301 Testing\Testing up to 090221")
data1 = np.loadtxt(fname="LOG0.txt",delimiter=',')
data2 = np.loadtxt(fname="LOG1.txt",delimiter=',')

os.chdir(r"C:\Users\Tannlin User\Documents\Parallel Kinematic Laser Head Profile\Testing\SS301 Testing\Testing up to 150221")

data3 = np.loadtxt(fname="LOG0.txt",delimiter=',')
data4 = np.loadtxt(fname="LOG1.txt",delimiter=',')
data5 = np.loadtxt(fname="LOG2.txt",delimiter=',')
data6 = np.loadtxt(fname="LOG3.txt",delimiter=',')

os.chdir(r"C:\Users\Tannlin User\Documents\Parallel Kinematic Laser Head Profile\Testing\SS301 Testing\Testing up to 180221")

data7 = np.loadtxt(fname="LOG0.txt",delimiter=',')
data8 = np.loadtxt(fname="LOG1.txt",delimiter=',')
data9 = np.loadtxt(fname="LOG2.txt",delimiter=',')

os.chdir(r"C:\Users\Tannlin User\Documents\Parallel Kinematic Laser Head Profile\Testing\SS301 Testing\Testing up to 210221")

data10 = np.loadtxt(fname="LOG0.txt",delimiter=',')
data11 = np.loadtxt(fname="LOG1.txt",delimiter=',')
data12 = np.loadtxt(fname="LOG2.txt",delimiter=',')

os.chdir(r"C:\Users\Tannlin User\Documents\Parallel Kinematic Laser Head Profile\Testing\SS301 Testing\Testing up to 240221")

data13 = np.loadtxt(fname="LOG0.txt",delimiter=',')
data14 = np.loadtxt(fname="LOG1.txt",delimiter=',')
data15 = np.loadtxt(fname="LOG2.txt",delimiter=',')
data16 = np.loadtxt(fname="LOG3.txt",delimiter=',')

os.chdir(r"C:\Users\Tannlin User\Documents\Parallel Kinematic Laser Head Profile\Testing\SS301 Testing\Testing up to 010321")
data17 = np.loadtxt(fname="LOG0.txt",delimiter=",")
data18 = np.loadtxt(fname="LOG1.txt",delimiter=",")

os.chdir(r"C:\Users\Tannlin User\Documents\Parallel Kinematic Laser Head Profile\Testing\SS301 Testing\Testing up to 050321")
data19 = np.loadtxt(fname="LOG0.txt",delimiter=",")
data20 = np.loadtxt(fname="LOG1.txt",delimiter=",")
data21 = np.loadtxt(fname="LOG2.txt",delimiter=",")

os.chdir(r"C:\Users\Tannlin User\Documents\Parallel Kinematic Laser Head Profile\Testing\SS301 Testing\Testing up to 080321")
data22 = np.loadtxt(fname="LOG0.txt",delimiter=",")
data23 = np.loadtxt(fname="LOG1.txt",delimiter=",")

os.chdir(r"C:\Users\Tannlin User\Documents\Parallel Kinematic Laser Head Profile\Testing\SS301 Testing\Testing up to 140321")
data24 = np.loadtxt(fname="LOG0.txt",delimiter=",")
data25 = np.loadtxt(fname="LOG1.txt",delimiter=",")
data26 = np.loadtxt(fname="LOG2.txt",delimiter=",")

os.chdir(r"C:\Users\Tannlin User\Documents\Parallel Kinematic Laser Head Profile\Testing\SS301 Testing\Testing up to 180321")
data27 = np.loadtxt(fname="LOG0.txt",delimiter=",")
data28 = np.loadtxt(fname="LOG1.txt",delimiter=",")
data29 = np.loadtxt(fname="LOG2.txt",delimiter=",")
data30 = np.loadtxt(fname="LOG3.txt",delimiter=",")

DATA_6Thou_tuple = (data1[:,1:3],data2[:,1:3],data3[:,1:3],data4[:,1:3],data5[:,1:3],data6[:,1:3],data7[:,1:3],data8[:,1:3],data9[:,1:3],data10[:,1:3],data11[:,1:3],data12[:,1:3],data13[:,1:3],data14[:,1:3],data15[:,1:3],data16[:,1:3],data17[:,1:3],data18[:,1:3],data19[:,1:3],data20[:,1:3],data21[:,1:3],data22[:,1:3],data23[:,1:3],data24[:,1:3],data25[:,1:3],data26[:,1:3],data27[:,1:3],data28[:,1:3],data29[:,1:3],data30[:,1:3])
DATA_6Thou = np.vstack(DATA_6Thou_tuple)
DATA_6Thou_301 = abs(DATA_6Thou)

N_6Thou_304 = 100*np.linspace(1,len(DATA_6Thou_304[:,0]),num=len(DATA_6Thou_304[:,0]))

Coefs_KL_6Thou_304 = np.polyfit(N_6Thou_304,DATA_6Thou_304[:,0],deg=3)
Coefs_KR_6Thou_304 = np.polyfit(N_6Thou_304,DATA_6Thou_304[:,1],deg=3)

LoBF_KL_6Thou_304 = np.polyval(Coefs_KL_6Thou_304,N_6Thou_304)
LoBF_KR_6Thou_304 = np.polyval(Coefs_KR_6Thou_304,N_6Thou_304)

Var_KL_6Thou_304 = np.var(DATA_6Thou_304[:,0])
Var_KL_6Thou_304 = round(Var_KL_6Thou_304,5)
Var_KL_6Thou_304 = str(Var_KL_6Thou_304)

Var_KR_6Thou_304 = np.var(DATA_6Thou_304[:,1])
Var_KR_6Thou_304 = round(Var_KR_6Thou_304,5)
Var_KR_6Thou_304 = str(Var_KR_6Thou_304)

label_KL_6Thou_304 = "304 Left (V = {})".format(Var_KL_6Thou_304)
label_KR_6Thou_304 = "304 Right (V = {})".format(Var_KR_6Thou_304)

N_6Thou_301 = 100*np.linspace(1,len(DATA_6Thou[:,0]),num=len(DATA_6Thou[:,0]))

Coefs_KL_6Thou_301 = np.polyfit(N_6Thou_301,DATA_6Thou_301[:,0],deg=3)
Coefs_KR_6Thou_301 = np.polyfit(N_6Thou_301,DATA_6Thou_301[:,1],deg=3)

LoBF_KL_6Thou_301 = np.polyval(Coefs_KL_6Thou_301,N_6Thou_301)
LoBF_KR_6Thou_301 = np.polyval(Coefs_KR_6Thou_301,N_6Thou_301)

Var_KL_6Thou_301 = np.var(DATA_6Thou_301[:,0])
Var_KL_6Thou_301 = round(Var_KL_6Thou_301,5)
Var_KL_6Thou_301 = str(Var_KL_6Thou_301)

Var_KR_6Thou_301 = np.var(DATA_6Thou_301[:,1])
Var_KR_6Thou_301 = round(Var_KR_6Thou_301,5)
Var_KR_6Thou_301 = str(Var_KR_6Thou_301)

label_KL_6Thou_301 = "301 Left (V = {})".format(Var_KL_6Thou_301)
label_KR_6Thou_301 = "301 Right (V = {})".format(Var_KR_6Thou_301)

#Plot with x as log axis

plt.semilogx(N_6Thou_304,LoBF_KL_6Thou_304,color='r',linestyle='solid',label=label_KL_6Thou_304)
plt.semilogx(N_6Thou_304,LoBF_KR_6Thou_304,color='b',linestyle='solid',label=label_KR_6Thou_304)
plt.semilogx(N_6Thou_301,LoBF_KL_6Thou_301,color='r',linestyle='dashed',label=label_KL_6Thou_301)
plt.semilogx(N_6Thou_301,LoBF_KR_6Thou_301,color='b',linestyle='dashed',label=label_KR_6Thou_301)
plt.legend()
plt.xlabel("Number of Cycles")
plt.ylabel("Current (A)")
plt.grid()
plt.title("0.15mm Flexure Fatigue Comparison")
plt.show()