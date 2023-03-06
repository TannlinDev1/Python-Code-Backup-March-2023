#Following is for Z400 at 50

import matplotlib.pyplot as plt
import os
import numpy as np

os.chdir("C:/Users/Tannlin User/Documents/Tannlin Internship 2020/Side Profiles/MicroVue Measurements/Between Batches/Z400At50")
os.getcwd()

step = 0.1
N_step = 60

XY_H100 = np.zeros((4,122))
XY_H50 = np.zeros((4,122))

XY_H50[0,:] = np.loadtxt(fname = "XY_H50_1.txt", delimiter=',')
XY_H50[1,:] = np.loadtxt(fname = "XY_H50_2.txt", delimiter=',')
XY_H50[2,:] = np.loadtxt(fname = "XY_H50_3.txt", delimiter=',')
XY_H50[3,:] = np.loadtxt(fname = "XY_H50_4.txt", delimiter=',')

XY_H100[0,:] = np.loadtxt(fname = "XY_H100_1.txt", delimiter=',')
XY_H100[1,:] = np.loadtxt(fname = "XY_H100_2.txt", delimiter=',')
XY_H100[2,:] = np.loadtxt(fname = "XY_H100_3.txt", delimiter=',')
XY_H100[3,:] = np.loadtxt(fname = "XY_H100_4.txt", delimiter=',')

XYC_H50 = np.zeros((4,61))
XYF_H50 = np.zeros((4,61))

XYC_H100 = np.zeros((4,61))
XYF_H100 = np.zeros((4,61))

for i in range(0,4):

    XYC_H50[i] = XY_H50[i,0:61]
    XYF_H50[i] = XY_H50[i,61:122]

    XYC_H100[i] = XY_H100[i,0:61]
    XYF_H100[i] = XY_H100[i,61:122]

Range_H50 = np.zeros((4,len(XYC_H50[0,:])))
Range_H100 = np.zeros((4,len(XYC_H100[0,:])))

for j in range(0,4):
    for i in range(0,len(XYC_H50[j,:])):
        
        Range_H50[j,i] = XYC_H50[j,i]-XYF_H50[j,i]
        Range_H100[j,i] = XYC_H100[j,i]-XYF_H100[j,i]

AveRange_H50 = np.zeros(len(Range_H50[0,:]))
AveRange_H100 = np.zeros(len(Range_H100[0,:]))
                    
for i in range(0,len(Range_H50[0,:])):
               
    AveRange_H50[i] = (Range_H50[0,i]+Range_H50[1,i]+Range_H50[2,i]+Range_H50[3,i])/4
    AveRange_H100[i] = (Range_H100[0,i]+Range_H100[1,i]+Range_H100[2,i]+Range_H100[3,i])/4
           
Y = np.linspace(0, len(XYC_H50[0,:]), num = len(XYC_H50[0,:]))

plot1 = plt.figure(1)

for i in range(0,4):
    plt.scatter(Y,XYC_H50[i,:],color='r')
    plt.scatter(Y,XYC_H100[i,:],color='g')
    
plt.xlabel('Cut Distance (um)')
plt.ylabel('Furthest Distance (um)')
plt.grid()

plot2 = plt.figure(2)

for i in range(0,4):
    plt.scatter(Y,XYF_H50[i,:],color='r')
    plt.scatter(Y,XYF_H100[i,:],color='g')

plt.xlabel('Cut Distance (um)')
plt.ylabel('Closest Distance (um)')
plt.grid()

plot3 = plt.figure(3)

for i in range(0,4):
    plt.plot(Y,Range_H50[i,:],color='r')
    plt.plot(Y,Range_H100[i,:],color='g')

plt.xlabel('Cut Distance (um)')
plt.ylabel('Range (um)')
plt.grid()

plot4 = plt.figure(4)

plt.plot(Y,AveRange_H50,label='50 um')
plt.plot(Y,AveRange_H100,label='100 um')

plt.xlabel('Cut Distance (um)')
plt.ylabel('Range (um)')
plt.grid()
plt.legend()
plt.show()
