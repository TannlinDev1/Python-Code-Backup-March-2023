import matplotlib.pyplot as plt
import os
import numpy as np

os.chdir("C:/Users/Tannlin User/Documents/Tannlin Internship 2020/CFD/Wall Stress Data/External Geometry Variation")
os.getcwd()

#Load Data

sp_H50_90D = np.loadtxt(fname='StaticPressure_H50_90D.txt')
sp_H150_90D = np.loadtxt(fname='StaticPressure_H150_90D.txt')
sp_H250_90D = np.loadtxt(fname='StaticPressure_H250_90D.txt')

sp_90D = np.zeros((len(sp_H50_90D),4))

sp_90D[:,0] = 1000000*(sp_H50_90D[:,0]-sp_H50_90D[0,0])
sp_90D[:,1] = sp_H50_90D[:,1]/1000000
sp_90D[:,2] = sp_H150_90D[:,1]/1000000
sp_90D[:,3] = sp_H250_90D[:,1]/1000000

N_slice = 10

sp_90D = sp_90D[N_slice:-N_slice, :]

plot1 = plt.figure(1)

plt.plot(sp_90D[:,0],sp_90D[:,1],label='50 um')
plt.plot(sp_90D[:,0],sp_90D[:,2],label='150 um')
plt.plot(sp_90D[:,0],sp_90D[:,3],label='250 um')

plt.grid()
plt.xlabel('Cut Depth (um)')
plt.ylabel('Static Pressure (MPa)')
plt.legend()

plt.show()

