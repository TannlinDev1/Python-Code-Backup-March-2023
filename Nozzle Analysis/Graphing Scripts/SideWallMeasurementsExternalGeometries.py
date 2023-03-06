import matplotlib.pyplot as plt
import os
import numpy as np

os.chdir("C:/Users/Tannlin User/Documents/Tannlin Internship 2020/Side Profiles/MicroVue Measurements/External Geometries")
os.getcwd()

step = 0.1
N_step = 60

#Load Data 50 | 100 | 150 | 200 | 250

XY_90 = np.zeros((8,61,2))
XY_R15 = np.zeros((8,61,2))
XY_R5 = np.zeros((8,61,2))

XY_90[0,:,0] = np.loadtxt(fname="XY_90D_H50.txt",delimiter=',')
XY_90[1,:,0] = np.loadtxt(fname="XY_90D_H100.txt",delimiter=',')
XY_90[2,:,0] = np.loadtxt(fname="XY_90D_H150.txt",delimiter=',')
XY_90[3,:,0] = np.loadtxt(fname="XY_90D_H200.txt",delimiter=',')
XY_90[4,:,0] = np.loadtxt(fname="XY_90D_H250.txt",delimiter=',')
XY_90[5,:,0] = np.loadtxt(fname="XY_90D_H300.txt",delimiter=',')
XY_90[6,:,0] = np.loadtxt(fname="XY_90D_H350.txt",delimiter=',')
XY_90[7,:,0] = np.loadtxt(fname="XY_90D_H400.txt",delimiter=',')

XY_90[0,:,1] = np.loadtxt(fname="XY_90D_H50_2.txt",delimiter=',')
XY_90[1,:,1] = np.loadtxt(fname="XY_90D_H100_2.txt",delimiter=',')
XY_90[2,:,1] = np.loadtxt(fname="XY_90D_H150_2.txt",delimiter=',')
XY_90[3,:,1] = np.loadtxt(fname="XY_90D_H200_2.txt",delimiter=',')
XY_90[4,:,1] = np.loadtxt(fname="XY_90D_H250_2.txt",delimiter=',')
XY_90[5,:,1] = np.loadtxt(fname="XY_90D_H300_2.txt",delimiter=',')
XY_90[6,:,1] = np.loadtxt(fname="XY_90D_H350_2.txt",delimiter=',')
XY_90[7,:,1] = np.loadtxt(fname="XY_90D_H400_2.txt",delimiter=',')

XY_R15[0,:,0] = np.loadtxt(fname="XY_R15_H50.txt",delimiter=',')
XY_R15[1,:,0] = np.loadtxt(fname="XY_R15_H100.txt",delimiter=',')
XY_R15[2,:,0] = np.loadtxt(fname="XY_R15_H150.txt",delimiter=',')
XY_R15[3,:,0] = np.loadtxt(fname="XY_R15_H200.txt",delimiter=',')
XY_R15[4,:,0] = np.loadtxt(fname="XY_R15_H250.txt",delimiter=',')
XY_R15[5,:,0] = np.NaN
XY_R15[6,:,0] = np.NaN
XY_R15[7,:,0] = np.NaN

XY_R15[0,:,1] = np.loadtxt(fname="XY_R15_H50_2.txt",delimiter=',')
XY_R15[1,:,1] = np.loadtxt(fname="XY_R15_H100_2.txt",delimiter=',')
XY_R15[2,:,1] = np.loadtxt(fname="XY_R15_H150_2.txt",delimiter=',')
XY_R15[3,:,1] = np.loadtxt(fname="XY_R15_H200_2.txt",delimiter=',')
XY_R15[4,:,1] = np.loadtxt(fname="XY_R15_H250_2.txt",delimiter=',')
XY_R15[5,:,1] = np.NaN
XY_R15[6,:,1] = np.NaN
XY_R15[7,:,1] = np.NaN

XY_R5[0,:,0] = np.loadtxt(fname="XY_R5_H50.txt",delimiter=',')
XY_R5[1,:,0] = np.loadtxt(fname="XY_R5_H100.txt",delimiter=',')
XY_R5[2,:,0] = np.loadtxt(fname="XY_R5_H150.txt",delimiter=',')
XY_R5[3,:,0] = np.loadtxt(fname="XY_R5_H200.txt",delimiter=',')
XY_R5[4,:,0] = np.loadtxt(fname="XY_R5_H250.txt",delimiter=',')
XY_R5[5,:,0] = np.loadtxt(fname="XY_R5_H300.txt",delimiter=',')
XY_R5[6,:,0] = np.loadtxt(fname="XY_R5_H350.txt",delimiter=',')
XY_R5[7,:,0] = np.loadtxt(fname="XY_R5_H400.txt",delimiter=',')

XY_R5[0,:,1] = np.loadtxt(fname="XY_R5_H50_2.txt",delimiter=',')
XY_R5[1,:,1] = np.loadtxt(fname="XY_R5_H100_2.txt",delimiter=',')
XY_R5[2,:,1] = np.loadtxt(fname="XY_R5_H150_2.txt",delimiter=',')
XY_R5[3,:,1] = np.loadtxt(fname="XY_R5_H200_2.txt",delimiter=',')
XY_R5[4,:,1] = np.loadtxt(fname="XY_R5_H250_2.txt",delimiter=',')
XY_R5[5,:,1] = np.loadtxt(fname="XY_R5_H300_2.txt",delimiter=',')
XY_R5[6,:,1] = np.loadtxt(fname="XY_R5_H350_2.txt",delimiter=',')
XY_R5[7,:,1] = np.loadtxt(fname="XY_R5_H400_2.txt",delimiter=',')

Y = np.linspace(0, 6, num = len(XY_90[0,:]))
Z = np.linspace(50, 400, num=8)

##
Var_90 = np.zeros((2,8))
Var_R15 = np.zeros((2,8))
Var_R5 = np.zeros((2,8))

Ave_90 = np.zeros((2,8))
Ave_R15 = np.zeros((2,8))
Ave_R5 = np.zeros((2,8))

for i in range(0,8):
    for j in range(0,2):
            
        Var_90[j,i] = np.var(XY_90[i,:,j])
        Var_R15[j,i] = np.var(XY_R15[i,:,j])
        Var_R5[j,i] = np.var(XY_R5[i,:,j])

        Ave_90[j,i] = np.mean(XY_90[i,:,j])
        Ave_R15[j,i] = np.mean(XY_R15[i,:,j])
        Ave_R5[j,i] = np.mean(XY_R5[i,:,j])

deg = 3

Z_sample = np.linspace(50,400,num=100)

Coef_90 = np.polyfit(Z, Var_90[0, :], deg)
Coef_R15 = np.polyfit(Z, Var_R15[0, :], deg)
Coef_R5 = np.polyfit(Z, Var_R5[0, :], deg)

LoBF_90 = np.polyval(Coef_90, Z_sample)
LoBF_R15= np.polyval(Coef_R15, Z_sample)
LoBF_R5 = np.polyval(Coef_R5, Z_sample)
##    
##plot1 = plt.figure(1)
##plt.scatter(Y,1000*XY_90[0,:],color='r',label='90 deg Chamfer')
##plt.scatter(Y,1000*XY_R15[0,:],color='g',label='1.5 mm Radius')
##plt.scatter(Y,1000*XY_R5[0,:],color='b',label='5 mm Radius')
##plt.xlabel('Distance (mm)')
##plt.ylabel('Surface Roughness (um)')
##plt.legend()
##plt.grid()
##plt.title('Outer Surface Measurements for H = 50 um')
##                        
##plot2 = plt.figure(2)
##plt.scatter(Y,1000*XY_90[1,:],color='r',label='90 deg Chamfer')
##plt.scatter(Y,1000*XY_R15[1,:],color='g',label='1.5 mm Radius')
##plt.scatter(Y,1000*XY_R5[1,:],color='b',label='5 mm Radius')
##plt.xlabel('Distance (mm)')
##plt.ylabel('Surface Roughness (um)')
##plt.legend()
##plt.grid()
##plt.title('Outer Surface Measurements for H = 100 um')
##                        
##plot3 = plt.figure(3)
##plt.scatter(Y,1000*XY_90[2,:],color='r',label='90 deg Chamfer')
##plt.scatter(Y,1000*XY_R15[2,:],color='g',label='1.5 mm Radius')
##plt.scatter(Y,1000*XY_R5[2,:],color='b',label='5 mm Radius')
##plt.xlabel('Distance (mm)')
##plt.ylabel('Surface Roughness (um)')
##plt.legend()
##plt.grid()
##plt.title('Outer Surface Measurements for H = 150 um')
##
##plot4 = plt.figure(4)
##plt.scatter(Y,1000*XY_90[3,:],color='r',label='90 deg Chamfer')
##plt.scatter(Y,1000*XY_R15[3,:],color='g',label='1.5 mm Radius')
##plt.scatter(Y,1000*XY_R5[3,:],color='b',label='5 mm Radius')
##plt.xlabel('Distance (mm)')
##plt.ylabel('Surface Roughness (um)')
##plt.legend()
##plt.grid()
##plt.title('Outer Surface Measurements for H = 200 um')
##
##plot5 = plt.figure(5)
##plt.scatter(Y,1000*XY_90[4,:],color='r',label='90 deg Chamfer')
##plt.scatter(Y,1000*XY_R15[4,:],color='g',label='1.5 mm Radius')
##plt.scatter(Y,1000*XY_R5[4,:],color='b',label='5 mm Radius')
##plt.xlabel('Distance (mm)')
##plt.ylabel('Surface Roughness (um)')
##plt.legend()
##plt.grid()
##plt.title('Outer Surface Measurements for H = 250 um')
##
##plot6 = plt.figure(6)
##plt.scatter(Y,1000*XY_90[5,:],color='r',label='90 deg Chamfer')
##plt.scatter(Y,1000*XY_R5[5,:],color='b',label='5 mm Radius')
##plt.xlabel('Distance (mm)')
##plt.ylabel('Surface Roughness (um)')
##plt.legend()
##plt.grid()
##plt.title('Outer Surface Measurements for H = 300 um')
##
##plot7 = plt.figure(7)
##plt.scatter(Y,1000*XY_90[6,:],color='r',label='90 deg Chamfer')
##plt.scatter(Y,1000*XY_R5[6,:],color='b',label='5 mm Radius')
##plt.xlabel('Distance (mm)')
##plt.ylabel('Surface Roughness (um)')
##plt.legend()
##plt.grid()
##plt.title('Outer Surface Measurements for H = 350 um')
##
##plot8 = plt.figure(8)
##plt.scatter(Y,1000*XY_90[7,:],color='r',label='90 deg Chamfer')
##plt.scatter(Y,1000*XY_R15[7,:],color='g',label='1.5 mm Radius')
##plt.scatter(Y,1000*XY_R5[7,:],color='b',label='5 mm Radius')
##plt.xlabel('Distance (mm)')
##plt.ylabel('Surface Roughness (um)')
##plt.legend()
##plt.grid()
##plt.title('Outer Surface Measurements for H = 400 um')

plot9 = plt.figure(9)

plt.plot(Z,Var_90[1,:],color='r',label='90 deg Chamfer')
plt.plot(Z,Var_R15[1,:],color='g',label='1.5 mm Radius')
plt.plot(Z,Var_R5[1,:],color='b',label='5 mm Radius')
plt.xlabel('Run Height (um)')
plt.ylabel('Variance')
plt.legend()
plt.grid()
plt.title('Variance of Outer Surface Measurements')

plt.show()
##
##plot10 = plt.figure(10)
##plt.plot(Z,Ave_90*1000,color='r',label='90 deg Chamfer')
##plt.plot(Z,Ave_R15*1000,color='g',label='1.5 mm Radius')
##plt.plot(Z,Ave_R5*1000,color='b',label='5 mm Radius')
##plt.xlabel('Run Height (um)')
##plt.ylabel('Mean Surface Roughness (um)')
##plt.legend()
##plt.grid()
##plt.title('Mean of Outer Surface Measurements')
##plt.show()
##
