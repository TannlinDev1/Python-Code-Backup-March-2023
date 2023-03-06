import matplotlib.pyplot as plt
import os
import numpy as np

os.chdir("C:/Users/Tannlin User/Documents/Tannlin Internship 2020/Side Profiles/MicroVue Measurements/Nozzle Diameters Middle Focus 2")
os.getcwd()

step = 0.1
N_step = 60
Y = np.linspace(0,6,num=61)
#Load Data 50 | 100 | 150

XY_N800 = np.zeros((3,122))
XY_N1000 = np.zeros((3,122))
XY_N1200 = np.zeros((4,122))
XY_Test = np.zeros((2,122))

XY_N800[0,:] = np.loadtxt(fname='XY_H50_N800_1.txt',delimiter=',')
XY_N800[1,:] = np.loadtxt(fname='XY_H100_N800_2.txt',delimiter=',')
XY_N800[2,:] = np.loadtxt(fname='XY_H150_N800_2.txt',delimiter=',')

XY_N1000[0,:] = np.loadtxt(fname='XY_H50_N1000_1.txt',delimiter=',')
XY_N1000[1,:] = np.loadtxt(fname='XY_H100_N1000_2.txt',delimiter=',')
XY_N1000[2,:] = np.loadtxt(fname='XY_H150_N1000_2.txt',delimiter=',')

XY_N1200[0,:] = np.loadtxt(fname='XY_H50_N1200_1.txt',delimiter=',')
XY_N1200[1,:] = np.loadtxt(fname='XY_H100_N1200_2.txt',delimiter=',')
XY_N1200[2,:] = np.loadtxt(fname='XY_H150_N1200_2.txt',delimiter=',')

XYC_N800 = np.zeros((3,61))
XYC_N1000 = np.zeros((3,61))
XYC_N1200 = np.zeros((3,61))

for i in range(0,3):

    XYC_N800[i,:] = XY_N800[i,0:61]

    XYC_N1000[i,:] = XY_N1000[i,0:61]

    XYC_N1200[i,:] = XY_N1200[i,0:61]

var_N800 = np.zeros(3)
var_N1000 = np.zeros(3)
var_N1200 = np.zeros(3)

deg = 3
Coef_N800 = np.zeros((3, deg + 1))
Coef_N1000 = np.zeros((3, deg + 1))
Coef_N1200 = np.zeros((3, deg + 1))

LoBF_N800 = np.zeros((len(Y), 3))
LoBF_N1000 = np.zeros((len(Y), 3))
LoBF_N1200 = np.zeros((len(Y), 3))


for j in range(0, 3):
    Coef_N800[j, :] = np.polyfit(Y, XYC_N800[j, :], deg)
    Coef_N1000[j, :] = np.polyfit(Y, XYC_N1000[j, :], deg)
    Coef_N1200[j, :] = np.polyfit(Y, XYC_N1200[j, :], deg)

    LoBF_N800[:,j] = np.polyval(Coef_N800[j, :], Y)
    LoBF_N1000[:,j] = np.polyval(Coef_N1000[j, :], Y)
    LoBF_N1200[:,j] = np.polyval(Coef_N1200[j, :], Y)

    var_N800[j] = np.sqrt(np.var(XYC_N800[j,:]))
    var_N1000[j] = np.sqrt(np.var(XYC_N1000[j,:]))
    var_N1200[j] = np.sqrt(np.var(XYC_N1200[j,:]))

Z = np.linspace(50,150,num=3)

plot7 = plt.figure(7)

plt.scatter(Y,XYC_N800[0,:],color='r',label='0.8 mm')
plt.scatter(Y,XYC_N1000[0,:],color='g',label='1.0 mm')
plt.scatter(Y,XYC_N1200[0,:],color='b',label='1.2 mm')

plt.legend()

plt.plot(Y, LoBF_N800[:, 0], color='r', linestyle='dashed', label='0.8 mm')
plt.plot(Y, LoBF_N1000[:, 0], color='g', linestyle='dashed', label='1.0 mm')
plt.plot(Y, LoBF_N1200[:, 0], color='b', linestyle='dashed', label='1.2 mm')

plt.grid()
plt.xlabel('Cut Distance(um)')
plt.ylabel('Edge Distance (mm)')
plt.title('Rougness Variation with Nozzle Diameter, H = 50 um')

plot8 = plt.figure(8)

plt.scatter(Y,XYC_N800[1,:],color='r',label='0.8 mm')
plt.scatter(Y,XYC_N1000[1,:],color='g',label='1.0 mm')
plt.scatter(Y,XYC_N1200[1,:],color='b',label='1.2 mm')

plt.legend()

plt.plot(Y, LoBF_N800[:, 1], color='r', linestyle='dashed', label='0.8 mm')
plt.plot(Y, LoBF_N1000[:, 1], color='g', linestyle='dashed', label='1.0 mm')
plt.plot(Y, LoBF_N1200[:, 1], color='b', linestyle='dashed', label='1.2 mm')

plt.grid()
plt.xlabel('Cut Distance(um)')
plt.ylabel('Edge Distance (mm)')
plt.title('Rougness Variation with Nozzle Diameter, H = 100 um')

plot9 = plt.figure(9)

plt.scatter(Y,XYC_N800[2,:],color='r',label='0.8 mm')
plt.scatter(Y,XYC_N1000[2,:],color='g',label='1.0 mm')
plt.scatter(Y,XYC_N1200[2,:],color='b',label='1.2 mm')

plt.legend()

plt.plot(Y, LoBF_N800[:, 2], color='r', linestyle='dashed', label='0.8 mm')
plt.plot(Y, LoBF_N1000[:, 2], color='g', linestyle='dashed', label='1.0 mm')
plt.plot(Y, LoBF_N1200[:, 2], color='b', linestyle='dashed', label='1.2 mm')

plt.grid()
plt.xlabel('Cut Distance(mm)')
plt.ylabel('Edge Distance (um)')
plt.title('Rougness Variation with Nozzle Diameter, H = 150 um')

plot10 = plt.figure(10)

plt.plot(Z,var_N800,label = "0.8 mm")
plt.plot(Z,var_N1000, label= "1.0 mm")
plt.plot(Z,var_N1200, label="1.2 mm")

plt.legend()
plt.xlabel("Run Height (mm)")
plt.ylabel("Variance")
plt.grid()

plt.title("Variance Variation with Run Height")
plt.show()
