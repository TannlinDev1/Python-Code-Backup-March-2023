import numpy as np
import matplotlib.pyplot as plt

P = 1000
D = 8e-3
A = 0.25*np.pi*D**2 
N = 1000
t_spec = 10e-9
LDT = 7e4

f_min = 1e3
f_max = 10e3

f = np.linspace(f_min, f_max, N)

t_min = 1e-6
t_max = 100e-6

t = np.linspace(t_min, t_max, N)
DAM = np.zeros((N,N))
res = np.zeros((N,N))

for i in range(0, N):
    for j in range(0, N):
        LHS = P/(f[i]*A)
        RHS = np.sqrt(t[j]/t_spec)*LDT
        res[i,j] = RHS - LHS
        if LHS > RHS:
            DAM[i,j] = 1
        else:
            DAM[i,j] = 0


f, t = np.meshgrid(f, t)

fig1, ax1 = plt.subplots(constrained_layout=True)
CS1 = ax1.contourf(f/1e3, t*1e6, np.transpose(res))
ax1.set_xlabel("Frequency (kHz)")
ax1.set_ylabel("Pulse Time (us)")
cbar1 = fig1.colorbar(CS1, location="bottom")
