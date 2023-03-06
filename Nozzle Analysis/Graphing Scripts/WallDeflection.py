import matplotlib.pyplot as plt
import os
import numpy as np

os.chdir("C:/Users/Tannlin User/Documents/Tannlin Internship 2020/CFD/Workpiece Deflection")
os.getcwd()

#Load Data

sp_H100 = np.loadtxt(fname = "StaticPressure_P18_H100.txt")
sp_H100[:,1] -= 101325

radius = np.linspace(0,0.03,num=len(sp_H100[:,1]))

area = np.pi*radius**2
force = sp_H100[:,1]*area

plot1 = plt.figure(1)
plt.plot(sp_H100[:,0]*1000,force)
plt.xlabel("Distance (mm)")
plt.ylabel("Force (N)")
plt.grid()
plt.show()
