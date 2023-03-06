#Polyplot script

import matplotlib.pyplot as plt
import os
import numpy as np

os.chdir("C:/Users/Tannlin User/Documents/Tannlin Internship 2020/CFD/Wall Stress Data")
os.getcwd()

sp_H150_P15 = np.loadtxt(fname = "StaticPressure_H150_P15.txt")
sp_H100_P15 = np.loadtxt(fname = "StaticPressure_H100_P15.txt")
sp_H50_P15  = np.loadtxt(fname = "StaticPressure_H50_P15.txt")

sp_P15 = np.zeros((len(sp_H150_P15),4))
sp_P15[:,0] = 1000000*(sp_H150_P15[:,0]-sp_H150_P15[0,0])
sp_P15[:,1] = sp_H50_P15[:,1]/1000000
sp_P15[:,2] = sp_H100_P15[:,1]/1000000
sp_P15[:,3] = sp_H150_P15[:,1]/1000000

N_slice = 10

sp_P15 = sp_P15[N_slice:-N_slice, :]

sp_P15_H100_poly = np.polyfit(sp_P15[:,0],sp_P15[:,2],15)
sp_P15_H100_polyplot = np.polyval(sp_P15_H100_poly,sp_P15[:,0])

plot1 = plt.figure(1)
plt.plot(sp_P15[:,0],sp_P15_H100_polyplot,label="Polynomial")
plt.plot(sp_P15[:,0],sp_P15[:,2],label="Real Data")
plt.grid()
plt.legend()
plt.show()
