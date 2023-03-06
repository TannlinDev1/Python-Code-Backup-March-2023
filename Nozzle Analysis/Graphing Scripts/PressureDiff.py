import matplotlib.pyplot as plt
import os
import numpy as np

os.chdir("C:/Users/Tannlin User/Documents/Tannlin Internship 2020/CFD/Bernoulli Pressure Difference")
os.getcwd()

#Load Data

dP_H50 = np.loadtxt(fname="dP_H50.txt")
dP_H100 = np.loadtxt(fname="dP_H100.txt")
dP_H150 = np.loadtxt(fname="dP_H150.txt")
dP_H200 = np.loadtxt(fname="dP_H200.txt")

X = dP_H50[:,0]-dP_H50[0,0]

plt.plot(X*1000,dP_H50[:,1]/1000000,label="50 um")
plt.plot(X*1000,dP_H100[:,1]/1000000,label="100 um")
plt.plot(X*1000,dP_H150[:,1]/1000000,label="150 um")
plt.plot(X*1000,dP_H200[:,1]/1000000,label="200 um")

plt.grid()
plt.legend()
plt.xlabel("Distance (mm)")
plt.ylabel("Static Pressure (MPa)")

plt.show()
