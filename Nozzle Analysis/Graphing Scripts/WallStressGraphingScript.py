import matplotlib.pyplot as plt
import os
import numpy as np

os.chdir("C:/Users/Tannlin User/Documents/Tannlin Internship 2020/CFD/Wall Stress Data/Nominal Nozzle, Pressure and Run Height Sensitivity")
os.getcwd()

#Load Data

ss_H150_P18 = np.loadtxt(fname = "ShearStress_H150_P18.txt")
ss_H100_P18 = np.loadtxt(fname = "ShearStress_H100_P18.txt")
ss_H50_P18  = np.loadtxt(fname = "ShearStress_H50_P18.txt")

ss_H150_P15 = np.loadtxt(fname = "ShearStress_H150_P15.txt")
ss_H100_P15 = np.loadtxt(fname = "ShearStress_H100_P15.txt")
ss_H50_P15  = np.loadtxt(fname = "ShearStress_H50_P15.txt")

ss_H150_P12 = np.loadtxt(fname = "ShearStress_H150_P12.txt")
ss_H100_P12 = np.loadtxt(fname = "ShearStress_H100_P12.txt")
ss_H50_P12  = np.loadtxt(fname = "ShearStress_H50_P12.txt")

dp_H150_P18 = np.loadtxt(fname = "DynamicPressure_H150_P18.txt")
dp_H100_P18 = np.loadtxt(fname = "DynamicPressure_H100_P18.txt")
dp_H50_P18  = np.loadtxt(fname = "DynamicPressure_H50_P18.txt")

dp_H150_P15 = np.loadtxt(fname = "DynamicPressure_H150_P15.txt")
dp_H100_P15 = np.loadtxt(fname = "DynamicPressure_H100_P15.txt")
dp_H50_P15  = np.loadtxt(fname = "DynamicPressure_H50_P15.txt")

dp_H150_P12 = np.loadtxt(fname = "DynamicPressure_H150_P12.txt")
dp_H100_P12 = np.loadtxt(fname = "DynamicPressure_H100_P12.txt")
dp_H50_P12  = np.loadtxt(fname = "DynamicPressure_H50_P12.txt")

sp_H300_P18 = np.loadtxt(fname = "StaticPressure_H300_P18.txt")
sp_H200_P18 = np.loadtxt(fname = "StaticPressure_H200_P18.txt")
sp_H175_P18 = np.loadtxt(fname = "StaticPressure_H175_P18.txt")
sp_H150_P18 = np.loadtxt(fname = "StaticPressure_H150_P18.txt")
sp_H125_P18 = np.loadtxt(fname = "StaticPressure_H125_P18.txt")
sp_H100_P18 = np.loadtxt(fname = "StaticPressure_H100_P18.txt")
sp_H75_P18  = np.loadtxt(fname = "StaticPressure_H75_P18.txt")
sp_H50_P18  = np.loadtxt(fname = "StaticPressure_H50_P18.txt")

sp_H150_P15 = np.loadtxt(fname = "StaticPressure_H150_P15.txt")
sp_H100_P15 = np.loadtxt(fname = "StaticPressure_H100_P15.txt")
sp_H50_P15  = np.loadtxt(fname = "StaticPressure_H50_P15.txt")

sp_H150_P12 = np.loadtxt(fname = "StaticPressure_H150_P12.txt")
sp_H100_P12 = np.loadtxt(fname = "StaticPressure_H100_P12.txt")
sp_H50_P12  = np.loadtxt(fname = "StaticPressure_H50_P12.txt")

#Arrange Data
# X (mm) | H = 50 mm | H = 100 mm | H = 150 mm

ss_P18 = np.zeros((len(ss_H150_P18),4))
ss_P18[:,0] = 1000000*(ss_H150_P18[:,0]-ss_H150_P18[0,0])
ss_P18[:,1] = ss_H50_P18[:,1]/1000
ss_P18[:,2] = ss_H100_P18[:,1]/1000
ss_P18[:,3] = ss_H150_P18[:,1]/1000

ss_P15 = np.zeros((len(ss_H150_P15),4))
ss_P15[:,0] = 1000000*(ss_H150_P15[:,0]-ss_H150_P15[0,0])
ss_P15[:,1] = ss_H50_P15[:,1]/1000
ss_P15[:,2] = ss_H100_P15[:,1]/1000
ss_P15[:,3] = ss_H150_P15[:,1]/1000

ss_P12 = np.zeros((len(ss_H150_P12),4))
ss_P12[:,0] = 1000000*(ss_H150_P12[:,0]-ss_H150_P12[0,0])
ss_P12[:,1] = ss_H50_P12[:,1]/1000
ss_P12[:,2] = ss_H100_P12[:,1]/1000
ss_P12[:,3] = ss_H150_P12[:,1]/1000

dp_P18 = np.zeros((len(dp_H150_P18),4))
dp_P18[:,0] = 1000000*(dp_H150_P18[:,0]-dp_H150_P18[0,0])
dp_P18[:,1] = dp_H50_P18[:,1]/1000
dp_P18[:,2] = dp_H100_P18[:,1]/1000
dp_P18[:,3] = dp_H150_P18[:,1]/1000

dp_P15 = np.zeros((len(dp_H150_P15),4))
dp_P15[:,0] = 1000000*(dp_H150_P15[:,0]-dp_H150_P15[0,0])
dp_P15[:,1] = dp_H50_P15[:,1]/1000
dp_P15[:,2] = dp_H100_P15[:,1]/1000
dp_P15[:,3] = dp_H150_P15[:,1]/1000

dp_P12 = np.zeros((len(dp_H150_P12),4))
dp_P12[:,0] = 1000000*(dp_H150_P12[:,0]-dp_H150_P12[0,0])
dp_P12[:,1] = dp_H50_P12[:,1]/1000
dp_P12[:,2] = dp_H100_P12[:,1]/1000
dp_P12[:,3] = dp_H150_P12[:,1]/1000

sp_P18 = np.zeros((len(sp_H150_P18),9))
sp_P18[:,0] = 1000000*(sp_H150_P18[:,0]-sp_H150_P18[0,0])
sp_P18[:,1] = sp_H50_P18[:,1]/1000000
sp_P18[:,2] = sp_H75_P18[:,1]/1000000
sp_P18[:,3] = sp_H100_P18[:,1]/1000000
sp_P18[:,4] = sp_H125_P18[:,1]/1000000
sp_P18[:,5] = sp_H150_P18[:,1]/1000000
sp_P18[:,6] = sp_H175_P18[:,1]/1000000
sp_P18[:,7] = sp_H200_P18[:,1]/1000000
sp_P18[:,8] = sp_H300_P18[:,1]/1000000

sp_P15 = np.zeros((len(sp_H150_P15),4))
sp_P15[:,0] = 1000000*(sp_H150_P15[:,0]-sp_H150_P15[0,0])
sp_P15[:,1] = sp_H50_P15[:,1]/1000000
sp_P15[:,2] = sp_H100_P15[:,1]/1000000
sp_P15[:,3] = sp_H150_P15[:,1]/1000000

sp_P12 = np.zeros((len(sp_H150_P12),4))
sp_P12[:,0] = 1000000*(sp_H50_P12[:,0]-sp_H50_P12[0,0])
sp_P12[:,1] = sp_H50_P12[:,1]/1000000
sp_P12[:,2] = sp_H100_P12[:,1]/1000000
sp_P12[:,3] = sp_H150_P12[:,1]/1000000

#Cut off ends (discontinuity at workpiece edge)

N_slice = 1

ss_P18 = ss_P18[N_slice:-N_slice, :]
dp_P18 = dp_P18[N_slice:-N_slice, :]
sp_P18 = sp_P18[N_slice:-N_slice, :]

ss_P15 = ss_P15[N_slice:-N_slice, :]
dp_P15 = dp_P15[N_slice:-N_slice, :]
sp_P15 = sp_P15[N_slice:-N_slice, :]

ss_P12 = ss_P12[N_slice:-N_slice, :]
dp_P12 = dp_P12[N_slice:-N_slice, :]
sp_P12 = sp_P12[N_slice:-N_slice, :]

#Pressure Gradients
# H = 50 mm | H = 100 mm | H = 150 mm

#Fit polynomial to data

dPdx_P18 = np.zeros((len(sp_P18[:,0]),3))
dPdx_P18[:,0] = np.gradient((sp_P18[:,1]),sp_P18[:,0])
dPdx_P18[:,1] = np.gradient((sp_P18[:,2]),sp_P18[:,0])
dPdx_P18[:,2] = np.gradient((sp_P18[:,3]),sp_P18[:,0])

dPdx_P15 = np.zeros((len(sp_P15[:,0]),3))
dPdx_P15[:,0] = np.gradient((sp_P15[:,1]),sp_P15[:,0])
dPdx_P15[:,1] = np.gradient((sp_P15[:,2]),sp_P15[:,0])
dPdx_P15[:,2] = np.gradient((sp_P15[:,3]),sp_P15[:,0])

dPdx_P12 = np.zeros((len(sp_P12[:,0]),3))
dPdx_P12[:,0] = np.gradient((sp_P12[:,1]),sp_P12[:,0])
dPdx_P12[:,1] = np.gradient((sp_P12[:,2]),sp_P12[:,0])
dPdx_P12[:,2] = np.gradient((sp_P12[:,3]),sp_P12[:,0])


#Define Find Nearest Function

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

#Boundary Layer Separation Point

BLSx_P18 = np.zeros(3)
BLSx_P15 = np.zeros(3)
BLSx_P12 = np.zeros(3)

for i in range(0,3):

    BLSx_P18[i] = sp_P18[find_nearest(dPdx_P18[:,i], 0),0]
    BLSx_P15[i] = sp_P15[find_nearest(dPdx_P15[:,i], 0),0]
    BLSx_P12[i] = sp_P12[find_nearest(dPdx_P12[:,i], 0),0]


#Maximum Pressures

Z = [50, 100, 150, 200]

PMax_P18 = np.zeros(4)

PMax_P18[0] = np.max(sp_P18[:,1])# H=50
PMax_P18[1] = np.max(sp_P18[:,3])# H=100
PMax_P18[2] = np.max(sp_P18[:,5])# H=150
PMax_P18[3] = np.max(sp_P18[:,7])# H=200

Z_H = np.linspace(50,200,num=100)

Coef_PMax = np.polyfit(Z, PMax_P18, 3)
P_PMax = np.polyval(Coef_PMax, Z_H)

#Plot Static/Dynamic Pressure and Shear Stress

#plot1 = plt.figure(1)
#plt.plot(ss_P18[:,0],ss_P18[:,1],label = "0.05 mm")
#plt.plot(ss_P18[:,0],ss_P18[:,2],label = "0.1 mm")
#plt.plot(ss_P18[:,0],ss_P18[:,3],label = "0.15 mm")
#plt.xlabel('Distance (mm)')
#plt.ylabel('Shear Stress (kPa)')
#plt.title('Shear Stress Along Kerf Varying Run Height for P = 18 bar')
#plt.legend()
#plt.grid()

#plot2 = plt.figure(2)
#plt.plot(dp_P18[:,0],dp_P18[:,1],label = "0.05 mm")
#plt.plot(dp_P18[:,0],dp_P18[:,2],label = "0.1 mm")
#plt.plot(dp_P18[:,0],dp_P18[:,3],label = "0.15 mm")
#plt.ylabel('Dynamic Pressure(kPa)')
#plt.title('Dynamic Pressure Along Kerf Varying Run Height for P = 18 bar')
#plt.legend()
#plt.grid()

plot3 = plt.figure(3)
plt.plot(sp_P18[:,0],sp_P18[:,1],label = "0.05 mm")
plt.plot(sp_P18[:,0],sp_P18[:,3],label = "0.1 mm")
plt.plot(sp_P18[:,0],sp_P18[:,5],label = "0.15 mm")
plt.xlabel('Distance (mm)')
plt.ylabel('Stress (MPa)')
plt.title('Static Pressure Along Kerf Varying Run Height for P = 18 bar')
plt.legend()
plt.grid()


#plot4 = plt.figure(4)
#plt.plot(ss_P12[:,0],ss_P12[:,1],label = "0.05 mm")
#plt.plot(ss_P12[:,0],ss_P12[:,2],label = "0.1 mm")
#plt.plot(ss_P12[:,0],ss_P12[:,3],label = "0.15 mm")
#plt.xlabel('Distance (mm)')
#plt.ylabel('Shear Stress (kPa)')
#plt.title('Shear Stress Along Kerf Varying Run Height for P = 12 bar')
#plt.legend()
#plt.grid()

#plot5 = plt.figure(5)
#plt.plot(dp_P12[:,0],dp_P12[:,1],label = "0.05 mm")
#plt.plot(dp_P12[:,0],dp_P12[:,2],label = "0.1 mm")
#plt.plot(dp_P12[:,0],dp_P12[:,3],label = "0.15 mm")
#plt.xlabel('Distance (mm)')
#plt.ylabel('Dynamic Pressure(kPa)')
#plt.title('Dynamic Pressure Along Kerf Varying Run Height for P = 12 bar')
#plt.legend()
#plt.grid()

plot6 = plt.figure(6)
plt.plot(sp_P12[:,0],sp_P12[:,1],label = "0.05 mm")
plt.plot(sp_P12[:,0],sp_P12[:,2],label = "0.1 mm")
plt.plot(sp_P12[:,0],sp_P12[:,3],label = "0.15 mm")
plt.xlabel('Distance (mm)')
plt.ylabel('Stress (MPa)')
plt.title('Static Pressure Along Kerf Varying Run Height for P = 12 bar')
plt.legend()
plt.grid()

#plot7 = plt.figure(7)
#plt.plot(ss_P15[:,0],ss_P15[:,1],label = "0.05 mm")
#plt.plot(ss_P15[:,0],ss_P15[:,2],label = "0.1 mm")
#plt.plot(ss_P15[:,0],ss_P15[:,3],label = "0.15 mm")
#plt.xlabel('Distance (mm)')
#plt.ylabel('Shear Stress (kPa)')
#plt.title('Shear Stress Along Kerf Varying Run Height for P = 15 bar')
#plt.legend()
#plt.grid()

#plot8 = plt.figure(8)
#plt.plot(dp_P15[:,0],dp_P15[:,1],label = "0.05 mm")
#plt.plot(dp_P15[:,0],dp_P15[:,2],label = "0.1 mm")
#plt.plot(dp_P15[:,0],dp_P15[:,3],label = "0.15 mm")
#plt.xlabel('Distance (mm)')
#plt.ylabel('Dynamic Pressure(kPa)')
#plt.title('Dynamic Pressure Along Kerf Varying Run Height for P = 15 bar')
#plt.legend()
#plt.grid()

plot9 = plt.figure(9)
plt.plot(sp_P15[:,0],sp_P15[:,1],label = "0.05 mm")
plt.plot(sp_P15[:,0],sp_P15[:,2],label = "0.1 mm")
plt.plot(sp_P15[:,0],sp_P15[:,3],label = "0.15 mm")
plt.xlabel('Distance (mm)')
plt.ylabel('Stress (MPa)')
plt.title('Static Pressure Along Kerf Varying Run Height for P = 15 bar')
plt.legend()
plt.grid()

#Plot Pressure Gradients

#plot10 = plt.figure(10)
#plt.plot(sp_P18[:,0],dPdx_P18[:,0],label = "0.05 mm")
#plt.plot(sp_P18[:,0],dPdx_P18[:,1],label = "0.1 mm")
#plt.plot(sp_P18[:,0],dPdx_P18[:,2],label = "0.15 mm")
#plt.xlabel("Distance (um)")
#plt.ylabel("Pressure Gradient (N/m^2/m)")
#plt.title("Pressure Gradient Along Kerf Varying Run Height for P = 18 bar")
#plt.grid()
#plt.legend()

#plot11 = plt.figure(11)
#plt.plot(sp_P15[:,0],dPdx_P15[:,0],label = "0.05 mm")
#plt.plot(sp_P15[:,0],dPdx_P15[:,1],label = "0.1 mm")
#plt.plot(sp_P15[:,0],dPdx_P15[:,2],label = "0.15 mm")
#plt.xlabel("Distance (um)")
#plt.ylabel("Pressure Gradient (N/m^2/m)")
#plt.title("Pressure Gradient Along Kerf Varying Run Height for P = 15 bar")
#plt.grid()
#plt.legend()

#plot13 = plt.figure(13)
#plt.plot(sp_P12[:,0],sp_P12[:,1],label = "12 bar")
#plt.plot(sp_P15[:,0],sp_P15[:,1],label = "15 bar")
#plt.plot(sp_P18[:,0],sp_P18[:,1],label = "18 bar")
#plt.xlabel("Distance (mm)")
#plt.ylabel("Static Pressure (MPa)")
#plt.title("Static Pressure Along Kerf Varying Supply Pressure for H = 0.05 mm")
#plt.grid()
#plt.legend()

#plot13 = plt.figure(14)
#plt.plot(sp_P12[:,0],sp_P12[:,2],label = "12 bar")
#plt.plot(sp_P15[:,0],sp_P15[:,2],label = "15 bar")
#plt.plot(sp_P18[:,0],sp_P18[:,2],label = "18 bar")
#plt.xlabel("Distance (mm)")
#plt.ylabel("Static Pressure (MPa)")
#plt.title("Static Pressure Along Kerf Varying Supply Pressure for H = 0.1 mm")
#plt.grid()
#plt.legend()

#plot13 = plt.figure(15)
#plt.plot(sp_P12[:,0],sp_P12[:,3],label = "12 bar")
#plt.plot(sp_P15[:,0],sp_P15[:,3],label = "15 bar")
#plt.plot(sp_P18[:,0],sp_P18[:,3],label = "18 bar")
#plt.xlabel("Distance (mm)")
#plt.title("Static Pressure Along Kerf Varying Supply Pressure for H = 0.15 mm")
#plt.grid()
#plt.legend()

plt.show()
