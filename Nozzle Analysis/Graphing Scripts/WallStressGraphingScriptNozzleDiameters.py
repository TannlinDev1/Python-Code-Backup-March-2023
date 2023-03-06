import matplotlib.pyplot as plt
import os
import numpy as np

os.chdir("C:/Users/Tannlin User/Documents/Tannlin Internship 2020/CFD/Wall Stress Data/Nozzle Diameter Variation")
os.getcwd()

#Load Data

sp_H50_N800 = np.loadtxt(fname='StaticPressure_H50_N800.txt')
sp_H50_N1000 = np.loadtxt(fname='StaticPressure_H50_N1000.txt')
sp_H50_N1200 = np.loadtxt(fname='StaticPressure_H50_N1200.txt')

sp_H100_N800 = np.loadtxt(fname='StaticPressure_H100_N800.txt')
sp_H100_N1000 = np.loadtxt(fname='StaticPressure_H100_N1000.txt')
sp_H100_N1200 = np.loadtxt(fname='StaticPressure_H100_N1200.txt')

sp_H50 = np.zeros((len(sp_H50_N1000),4))
sp_H100 = np.zeros((len(sp_H100_N1000),4))

sp_H50[:,0] = 1000000*(sp_H50_N1000[:,0]-sp_H50_N1000[0,0])
sp_H50[:,1] = sp_H50_N800[:,1]/1000000
sp_H50[:,2] = sp_H50_N1000[:,1]/1000000
sp_H50[:,3] = sp_H50_N1200[:,1]/1000000

sp_H100[:,0] = 1000000*(sp_H100_N1000[:,0]-sp_H100_N1000[0,0])
sp_H100[:,1] = sp_H100_N800[:,1]/1000000
sp_H100[:,2] = sp_H100_N1000[:,1]/1000000
sp_H100[:,3] = sp_H100_N1200[:,1]/1000000

N_slice = 10

sp_H50 = sp_H50[N_slice:-N_slice, :]
sp_H100 = sp_H100[N_slice:-N_slice, :]

plot1 = plt.figure(1)

plt.plot(sp_H50[:,0],sp_H50[:,1],label='0.8 mm')
plt.plot(sp_H50[:,0],sp_H50[:,2],label='1 mm')
plt.plot(sp_H50[:,0],sp_H50[:,3],label='1.2 mm')

plt.grid()
plt.xlabel('Cut Depth (um)')
plt.ylabel('Static Pressure (MPa)')
plt.legend()

plot2 = plt.figure(2)
plt.plot(sp_H100[:,0],sp_H100[:,1],label='0.8 mm')
plt.plot(sp_H100[:,0],sp_H100[:,2],label='1 mm')
plt.plot(sp_H100[:,0],sp_H100[:,3],label='1.2 mm')

plt.grid()
plt.xlabel('Cut Depth (um)')
plt.ylabel('Static Pressure (MPa)')
plt.legend()

plt.show()

