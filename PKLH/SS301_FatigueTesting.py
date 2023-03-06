import os
import numpy
import matplotlib.pyplot as plt
import numpy as np

os.chdir(r"E:\Users\Tannlin User\Documents\Parallel Kinematic Laser Head Profile\Testing\SS301 Testing\Testing up to 090221")

data0 = np.loadtxt(fname="LOG0.txt", delimiter=",")
data1 = np.loadtxt(fname="LOG1.txt", delimiter=",")

os.chdir(r"E:\Users\Tannlin User\Documents\Parallel Kinematic Laser Head Profile\Testing\SS301 Testing\Testing up to 150221")

data2 = np.loadtxt(fname="LOG0.txt", delimiter=",")
data3 = np.loadtxt(fname="LOG1.txt", delimiter=",")
data4 = np.loadtxt(fname="LOG2.txt", delimiter=",")
data5 = np.loadtxt(fname="LOG3.txt", delimiter=",")

os.chdir(r"E:\Users\Tannlin User\Documents\Parallel Kinematic Laser Head Profile\Testing\SS301 Testing\Testing up to 180221")

data6 = np.loadtxt(fname="LOG0.txt", delimiter=",")
data7 = np.loadtxt(fname="LOG1.txt", delimiter=",")
data8 = np.loadtxt(fname="LOG2.txt", delimiter=",")

os.chdir(r"E:\Users\Tannlin User\Documents\Parallel Kinematic Laser Head Profile\Testing\SS301 Testing\Testing up to 210221")

data9 = np.loadtxt(fname="LOG0.txt", delimiter=",")
data10 = np.loadtxt(fname="LOG1.txt", delimiter=",")
data11 = np.loadtxt(fname="LOG2.txt", delimiter=",")

os.chdir(r"E:\Users\Tannlin User\Documents\Parallel Kinematic Laser Head Profile\Testing\SS301 Testing\Testing up to 240221")

data12 = np.loadtxt(fname="LOG0.txt", delimiter=",")
data13 = np.loadtxt(fname="LOG1.txt", delimiter=",")
data14 = np.loadtxt(fname="LOG2.txt", delimiter=",")
data15 = np.loadtxt(fname="LOG3.txt", delimiter=",")

os.chdir(r"E:\Users\Tannlin User\Documents\Parallel Kinematic Laser Head Profile\Testing\SS301 Testing\Testing up to 010321")

data16 =  np.loadtxt(fname="LOG0.txt", delimiter=",")
data17 = np.loadtxt(fname="LOG1.txt", delimiter=",")

os.chdir(r"E:\Users\Tannlin User\Documents\Parallel Kinematic Laser Head Profile\Testing\SS301 Testing\Testing up to 050321")

data18 = np.loadtxt(fname="LOG0.txt", delimiter=",")
data19 = np.loadtxt(fname="LOG1.txt", delimiter=",")
data20 = np.loadtxt(fname="LOG2.txt", delimiter=",")

os.chdir(r"E:\Users\Tannlin User\Documents\Parallel Kinematic Laser Head Profile\Testing\SS301 Testing\Testing up to 080321")

data21 =  np.loadtxt(fname="LOG0.txt", delimiter=",")
data22 = np.loadtxt(fname="LOG1.txt", delimiter=",")

os.chdir(r"E:\Users\Tannlin User\Documents\Parallel Kinematic Laser Head Profile\Testing\SS301 Testing\Testing up to 140321")

data23 = np.loadtxt(fname="LOG0.txt", delimiter=",")
data24 = np.loadtxt(fname="LOG1.txt", delimiter=",")
data25 = np.loadtxt(fname="LOG2.txt", delimiter=",")

os.chdir(r"E:\Users\Tannlin User\Documents\Parallel Kinematic Laser Head Profile\Testing\SS301 Testing\Testing up to 180321")

data26 = np.loadtxt(fname="LOG0.txt", delimiter=",")
data27 = np.loadtxt(fname="LOG1.txt", delimiter=",")
data28 = np.loadtxt(fname="LOG2.txt", delimiter=",")
data29 = np.loadtxt(fname="LOG3.txt", delimiter=",")

os.chdir(r"E:\Users\Tannlin User\Documents\Parallel Kinematic Laser Head Profile\Testing\SS301 Testing\Testing up to 210321")

data30 = np.loadtxt(fname="LOG0.txt", delimiter=",")
data31 = np.loadtxt(fname="LOG1.txt", delimiter=",")
data32 = np.loadtxt(fname="LOG2.txt", delimiter=",")
data33 = np.loadtxt(fname="LOG3.txt", delimiter=",")

os.chdir(r"E:\Users\Tannlin User\Documents\Parallel Kinematic Laser Head Profile\Testing\SS301 Testing\Testing up to 310321")

data34 = np.loadtxt(fname="LOG0.txt", delimiter=",")
data35 = np.loadtxt(fname="LOG1.txt", delimiter=",")
data36 = np.loadtxt(fname="LOG2.txt", delimiter=",")

os.chdir(r"E:\Users\Tannlin User\Documents\Parallel Kinematic Laser Head Profile\Testing\SS301 Testing\Testing up to 070421")

data37 = np.loadtxt(fname="LOG0.txt", delimiter=",")
data38 = np.loadtxt(fname="LOG1.txt", delimiter=",")
data39 = np.loadtxt(fname="LOG2.txt", delimiter=",")
data40 = np.loadtxt(fname="LOG3.txt", delimiter=",")
data41 = np.loadtxt(fname="LOG4.txt", delimiter=",")
data42 = np.loadtxt(fname="LOG5.txt", delimiter=",")

os.chdir(r"E:\Users\Tannlin User\Documents\Parallel Kinematic Laser Head Profile\Testing\SS301 Testing\Testing up to 140421")

data43 = np.loadtxt(fname="LOG0.txt", delimiter=",")
data44 = np.loadtxt(fname="LOG1.txt", delimiter=",")
data45 = np.loadtxt(fname="LOG2.txt", delimiter=",")
data46 = np.loadtxt(fname="LOG3.txt", delimiter=",")
data47 = np.loadtxt(fname="LOG4.txt", delimiter=",")
data48 = np.loadtxt(fname="LOG5.txt", delimiter=",")

LeftCurrent = np.concatenate((data1[:,1],data2[:,1],data3[:,1],data4[:,1],data5[:,1],data6[:,1],data7[:,1],data8[:,1],data9[:,1],data10[:,1],data11[:,1],data12[:,1],data13[:,1],data14[:,1],data15[:,1],data16[:,1],data17[:,1],data18[:,1],data19[:,1],data20[:,1],data21[:,1],data22[:,1],data23[:,1],data24[:,1],data25[:,1],data26[:,1],data27[:,1],data28[:,1],data29[:,1],data30[:,1],data31[:,1],data32[:,1],data33[:,1],data34[:,1],data35[:,1],data36[:,1],data37[:,1],data38[:,1],data39[:,1],data40[:,1],data41[:,1],data42[:,1],data43[:,1],data44[:,1],data45[:,1],data46[:,1],data47[:,1],data48[:,1]), axis=0)
RightCurrent = np.concatenate((data1[:,2],data2[:,2],data3[:,2],data4[:,2],data5[:,2],data6[:,2],data7[:,2],data8[:,2],data9[:,2],data10[:,2],data11[:,2],data12[:,2],data13[:,2],data14[:,2],data15[:,2],data16[:,2],data17[:,2],data18[:,2],data19[:,2],data20[:,2],data21[:,2],data22[:,2],data23[:,2],data24[:,2],data25[:,2],data26[:,2],data27[:,2],data28[:,2],data29[:,2],data30[:,2],data31[:,2],data32[:,2],data33[:,2],data34[:,2],data35[:,2],data36[:,2],data37[:,2],data38[:,2],data39[:,2],data40[:,2],data41[:,2],data42[:,2],data43[:,2],data44[:,2],data45[:,2],data46[:,2],data47[:,2],data48[:,2]), axis=0)
N = np.linspace(100,np.size(LeftCurrent)*100, num = np.size(LeftCurrent))

Coefs_KL = np.polyfit(N, LeftCurrent, deg=4)
Coefs_KR = np.polyfit(N, RightCurrent, deg=4)

LoBF_KL = np.polyval(Coefs_KL, N)
LoBF_KR = np.polyval(Coefs_KR, N)

plt.figure(1)
plt.semilogx(N, abs(LoBF_KL),color="r",label="Left")
plt.semilogx(N, abs(LoBF_KR), color="g", label="Right")
plt.legend()
plt.xlabel("Number of Cycles")
plt.ylabel("Current (A)")
plt.grid()
plt.title("SS301 Testing")
plt.show()