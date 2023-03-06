import matplotlib.pyplot as plt
import os
import numpy as np

os.chdir("C:/Users/Tannlin User/Documents/Tannlin Internship 2020/Side Profiles/MicroVue Measurements/Nozzle Diameter Middle Focus")
os.getcwd()

step = 0.1
N_step = 60

#Load Data 50 | 100 | 150

XY_N800 = np.zeros((3,122))
XY_N1000 = np.zeros((3,122))
XY_N1200 = np.zeros((4,122))

XY_N800[0,:] = np.loadtxt(fname='XY_N800_H50_1.txt',delimiter=',')
XY_N800[1,:] = np.loadtxt(fname='XY_N800_H100_1.txt',delimiter=',')
XY_N800[2,:] = np.loadtxt(fname='XY_N800_H150_1.txt',delimiter=',')

XY_N1000[0,:] = np.loadtxt(fname='XY_N1000_H50_1.txt',delimiter=',')
XY_N1000[1,:] = np.loadtxt(fname='XY_N1000_H100_1.txt',delimiter=',')
XY_N1000[2,:] = np.loadtxt(fname='XY_N1000_H150_1.txt',delimiter=',')

XY_N1200[0,:] = np.loadtxt(fname='XY_N1200_H50_1.txt',delimiter=',')
XY_N1200[1,:] = np.loadtxt(fname='XY_N1200_H100_1.txt',delimiter=',')
XY_N1200[2,:] = np.loadtxt(fname='XY_N1200_H150_1.txt',delimiter=',')
XY_N1200[3,:] = np.loadtxt(fname='XY_N1200_H200_1.txt',delimiter=',')

XYC_N800 = np.zeros((3,61))
XYC_N1000 = np.zeros((3,61))
XYC_N1200 = np.zeros((4,61))

XYF_N800 = np.zeros((3,61))
XYF_N1000 = np.zeros((3,61))
XYF_N1200 = np.zeros((4,61))

for i in range(0,3):

    XYC_N800[i] = XY_N800[i,0:61]
    XYF_N800[i] = XY_N800[i,61:122]

    XYC_N1000[i] = XY_N1000[i,0:61]
    XYF_N1000[i] = XY_N1000[i,61:122]

    XYC_N1200[i] = XY_N1200[i,0:61]
    XYF_N1200[i] = XY_N1200[i,61:122]

XYC_N1200[3] = XY_N1200[3,0:61]
XYC_N1200[3] = XY_N1200[3,0:61]

Range_N800 = np.zeros((3,len(XYC_N800[0,:])))
Range_N1000 = np.zeros((3,len(XYC_N1000[0,:])))
Range_N1200 = np.zeros((4,len(XYC_N1200[0,:])))


for j in range(0,3):
    for i in range(0,len(XYC_N800[j,:])):
        Range_N800[j,i] = XYC_N800[j,i] - XYF_N800[j,i]
        Range_N1000[j,i] = XYC_N1000[j,i] - XYF_N1000[j,i]
        Range_N1200[j,i] = XYC_N1200[j,i] - XYF_N1200[j,i]

Range_N1200[3,:] = XYC_N1200[3,:]-XYF_N1200[3,:]

deg = 1
Coefs_N800 = np.zeros((3,deg+1))
Coefs_N1000 = np.zeros((3,deg+1))
Coefs_N1200 = np.zeros((3,deg+1))

Y = np.linspace(0, len(Range_N1200[0,:]), num = len(Range_N1200[0,:]))

for i in range(0,3):
    
    Coefs_N800[i,:] = np.polyfit(Y,Range_N800[i,:],deg)
    Coefs_N1000[i,:] = np.polyfit(Y,Range_N1000[i,:],deg)
    Coefs_N1200[i,:] = np.polyfit(Y,Range_N1200[i,:],deg)

LoBF_N800 = np.zeros((3,len(Range_N800[0,:])))
LoBF_N1000 = np.zeros((3,len(Range_N1000[0,:])))
LoBF_N1200 = np.zeros((3,len(Range_N1200[0,:])))

for i in range(0,3):

    LoBF_N800[i,:] = np.polyval(Coefs_N800[i,:],Y)
    LoBF_N1000[i,:] = np.polyval(Coefs_N1000[i,:],Y)   
    LoBF_N1200[i,:] = np.polyval(Coefs_N1200[i,:],Y)

Y = np.linspace(0, len(Range_N1200[0,:]), num = len(Range_N1200[0,:]))

labels = ['50 um', '100 um','150 um', '200 um']
colors = ['r', 'g','b']

plot1 = plt.figure(1)

for i in range(0,3):
    plt.scatter(Y,Range_N800[i,:],label=labels[i],color=colors[i])
    plt.plot(Y,LoBF_N800[i,:],color=colors[i],linestyle='--')
    
plt.legend()
plt.grid()
plt.xlabel('Cut Distance (um)')
plt.ylabel('Edge Range (um)')
plt.title('Roughness Variation with Run Height, D = 0.8 mm')

plot2 = plt.figure(2)

for i in range(0,3):
    plt.scatter(Y,Range_N1000[i,:],label=labels[i],color=colors[i])
    plt.plot(Y,LoBF_N1000[i,:],color=colors[i],linestyle='--')

plt.legend()
plt.grid()
plt.xlabel('Cut Distance (um)')
plt.ylabel('Edge Range (um)')
plt.title('Roughness Variation with Run Height, D = 1.0 mm')

plot3 = plt.figure(3)

for i in range(0,3):
    plt.scatter(Y,Range_N1200[i,:],label=labels[i],color=colors[i])
    plt.plot(Y,LoBF_N1200[i,:],color=colors[i],linestyle='--')
    
plt.legend()
plt.grid()
plt.xlabel('Cut Distance (um)')
plt.ylabel('Edge Range (um)')
plt.title('Roughness Variation with Run Height, D = 1.2 mm')

plot4 = plt.figure(4)

plt.scatter(Y,Range_N800[0,:],color='r',label='0.8 mm')
plt.plot(Y,LoBF_N800[0,:],color='r',linestyle='--')

plt.scatter(Y,Range_N1000[0,:],color='g',label='1.0 mm')
plt.plot(Y,LoBF_N1000[0,:],color='g',linestyle='--')

plt.scatter(Y,Range_N1200[0,:],color='b',label='1.2 mm')
plt.plot(Y,LoBF_N1200[0,:],color='b',linestyle='--')

plt.xlabel('Cut Distance (um)')
plt.ylabel('Edge Range (um)')
plt.grid()
plt.legend()
plt.title('Roughness Variation with Nozzle Diameter, H = 50 um')

plot5 = plt.figure(5)

plt.scatter(Y,Range_N800[1,:],color='r',label='0.8 mm')
plt.plot(Y,LoBF_N800[1,:],color='r',linestyle='--')

plt.scatter(Y,Range_N1000[1,:],color='g',label='1.0 mm')
plt.plot(Y,LoBF_N1000[1,:],color='g',linestyle='--')

plt.scatter(Y,Range_N1200[1,:],color='b',label='1.2 mm')
plt.plot(Y,LoBF_N1200[1,:],color='b',linestyle='--')

plt.xlabel('Cut Distance (um)')
plt.ylabel('Edge Range (um)')
plt.grid()
plt.legend()
plt.title('Roughness Variation with Nozzle Diameter, H = 100 um')

plot6 = plt.figure(6)

plt.scatter(Y,Range_N800[2,:],color='r',label='0.8 mm')
plt.plot(Y,LoBF_N800[2,:],color='r',linestyle='--')

plt.scatter(Y,Range_N1000[2,:],color='g',label='1.0 mm')
plt.plot(Y,LoBF_N1000[2,:],color='g',linestyle='--')

plt.scatter(Y,Range_N1200[2,:],color='b',label='1.2 mm')
plt.plot(Y,LoBF_N1200[2,:],color='b',linestyle='--')

plt.xlabel('Cut Distance (um)')
plt.ylabel('Edge Range (um)')
plt.grid()
plt.legend()
plt.title('Roughness Variation with Nozzle Diameter, H = 150 um')

plt.show()
