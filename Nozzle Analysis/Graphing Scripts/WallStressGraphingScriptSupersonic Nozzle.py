import matplotlib.pyplot as plt
import os
import numpy as np

os.chdir("C:/Users/Tannlin User/Documents/Tannlin Internship 2020/CFD/Wall Stress Data/Supersonic Nozzle")
os.getcwd()

sp_D05_H2000 = np.loadtxt(fname="StaticPressure_D05_H2000.txt")
ss_D05_H2000 = np.loadtxt(fname="ShearStress_D05_H2000.txt")

sp_D05_H1000 = np.loadtxt(fname="StaticPressure_D05_H1000.txt")
ss_D05_H1000 = np.loadtxt(fname="ShearStress_D05_H1000.txt")

sp_D05_H100 = np.loadtxt(fname="StaticPressure_D05_H100.txt")
ss_D05_H100 = np.loadtxt(fname="ShearStress_D05_H100.txt")

sp_D05_H100 = np.loadtxt(fname="StaticPressure_D05_H100.txt")
ss_D05_H100= np.loadtxt(fname="ShearStress_D05_H100.txt")


#Clip Structures

min_clip = np.zeros(3)
max_clip = np.zeros(3)
min_clip[2] = 0.012
max_clip[2] = 0.0122
min_clip[1] = 0.0110001
max_clip[1] = 0.0112
min_clip[0] = 0.0101001
max_clip[0] = 0.0103

#P_0 structure:
# X (mm) | 0.1 mm | 1 mm | 2 mm
P_n = np.zeros((149,4))
P_n[:,0] = np.linspace(0,200,num=149)

#Tau structure:
tau = np.zeros((149,4))
tau[:,0] = np.linspace(0,200,num=149)

#Index structure:
#      | 0.1 mm | 1 mm | 2 mm
# P_n  |
# P_d  |
# Tau  |



def ClipToWorkpiece():
    P_n[:,3] = sp_D05_H2000[:,1][(sp_D05_H2000[:,0]>min_clip[2]) & (sp_D05_H2000[:,0]<max_clip[2])]
    tau[:,3] = ss_D05_H2000[:,1][(ss_D05_H2000[:,0]>min_clip[2]) & (ss_D05_H2000[:,0]<max_clip[2])]
    P_n[:,2] = sp_D05_H1000[:,1][(sp_D05_H1000[:,0]>min_clip[1]) & (sp_D05_H1000[:,0]<max_clip[1])]
    tau[:,2] = ss_D05_H1000[:,1][(ss_D05_H1000[:,0]>min_clip[1]) & (ss_D05_H1000[:,0]<max_clip[1])]
    P_n[:,1] = sp_D05_H100[:,1][(sp_D05_H100[:,0]>min_clip[0]) & (sp_D05_H100[:,0]<max_clip[0])]
    tau[:,1] = ss_D05_H100[:,1][(ss_D05_H100[:,0]>min_clip[0]) & (ss_D05_H100[:,0]<max_clip[0])]
    return[P_n]
            
ClipToWorkpiece()    

plot1 = plt.figure(1)
plt.plot(P_n[:,0],P_n[:,3]/1000000,'r',label='2 mm')
plt.plot(P_n[:,0],P_n[:,2]/1000000,'b',label='1 mm')
plt.plot(P_n[:,0],P_n[:,1]/1000000,'g',label='0.1 mm')
plt.xlabel(" Cut Distance (mm)")
plt.ylabel(" Static Pressure (MPa)")
plt.legend()
plt.grid()

plot2 = plt.figure(2)

plt.plot(tau[:,0],tau[:,3],'r',label='2 mm')
plt.plot(tau[:,0],tau[:,2],'b',label='1 mm')
plt.plot(tau[:,0],tau[:,1],'g',label='0.1 mm')
plt.xlabel(" Cut Distance (mm)")
plt.ylabel(" Static Pressure (MPa)")
plt.legend()
plt.grid()

plt.show()
