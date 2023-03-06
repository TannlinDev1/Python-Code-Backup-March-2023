import matplotlib.pyplot as plt
import os
import numpy as np

os.chdir("C:/Users/Tannlin User/Documents/Parallel Kinematic Laser Head Profile/Literature/SN Curves/Bakri et al")
os.getcwd()
stress = 631.25
ZeroPC = np.loadtxt(fname="0PCStrain.txt", delimiter=',')
TwoPC = np.loadtxt(fname="2PCStrain.txt", delimiter=',')
FivePC = np.loadtxt(fname="5PCStrain.txt", delimiter=',')
Hard = np.loadtxt(fname="HardStrain.txt", delimiter=',')

Coefs_0PC = np.polyfit(np.log(ZeroPC[:,0]), ZeroPC[:,1], 1)
Coefs_2PC = np.polyfit(np.log(TwoPC[:,0]), TwoPC[:,1], 1)
Coefs_5PC = np.polyfit(np.log(FivePC[:,0]), FivePC[:,1], 1)
Coefs_Hard = np.polyfit(np.log(Hard[:,0]), Hard[:,1], 1)

LoBF_0PC = np.zeros(len(ZeroPC[:,0]))
LoBF_2PC = np.zeros(len(TwoPC[:,0]))
LoBF_5PC = np.zeros(len(FivePC[:,0]))
LoBF_Hard = np.zeros(len(Hard[:,0]))

for i in range(0,len(ZeroPC[:,0])):
    LoBF_0PC[i] = Coefs_0PC[0]*np.log(ZeroPC[i,0]) + Coefs_0PC[1]

for i in range(0,len(TwoPC[:,0])):
    LoBF_2PC[i] = Coefs_2PC[0] * np.log(TwoPC[i, 0]) + Coefs_2PC[1]

for i in range(0,len(FivePC[:,0])):
    LoBF_5PC[i] = Coefs_5PC[0] * np.log(FivePC[i, 0]) + Coefs_5PC[1]

for i in range(0,len(Hard[:,0])):
    LoBF_Hard[i] = Coefs_Hard[0] * np.log(Hard[i, 0]) + Coefs_Hard[1]

C2F_0PC = np.exp((stress-Coefs_0PC[1])/Coefs_0PC[0])
C2F_2PC = np.exp((stress-Coefs_2PC[1])/Coefs_2PC[0])
C2F_5PC = np.exp((stress-Coefs_5PC[1])/Coefs_5PC[0])
C2F_Hard = np.exp((stress-Coefs_Hard[1])/Coefs_Hard[0])

fig1 = plt.figure()
ax = plt.gca()
ax.plot(ZeroPC[:,0],LoBF_0PC,color='r',linestyle='dashed',label='0 %')
ax.plot(TwoPC[:,0],LoBF_2PC,color='g',linestyle='dashed',label='2 %')
ax.plot(FivePC[:,0],LoBF_5PC,color='b',linestyle='dashed',label='5 %')
ax.plot(Hard[:,0],LoBF_Hard,color='m',linestyle='dashed',label='Hardened')
plt.hlines(stress,ZeroPC[0,0],ZeroPC[-1,0],color='k',label="Maximum Stress")
plt.legend()
ax.scatter(ZeroPC[:,0],ZeroPC[:,1],color='r')
ax.scatter(TwoPC[:,0],TwoPC[:,1],color='g')
ax.scatter(FivePC[:,0],FivePC[:,1],color='b')
ax.scatter(Hard[:,0],Hard[:,1],color='m')
ax.set_xscale('log')
plt.grid()
plt.xlabel("Number of Cycles")
plt.ylabel("Stress (MPa)")
plt.title("S-N Curve For Varying Strained SS304")
plt.show()
