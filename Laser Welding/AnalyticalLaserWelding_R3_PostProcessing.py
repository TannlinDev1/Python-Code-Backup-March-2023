import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

N_GLOB = 25 #number of spot radius iterations

P_L_min = 500 #minimum power (W)
P_L_max = 1000 #maximum power (W)
P_L_arr = np.linspace(P_L_min, P_L_max, N_GLOB) #power array

v_min = 5e-3 #minimum weld speed (m/s)
v_max = 50e-3 #maximum weld speed (m/s)
v_arr = np.linspace(v_min, v_max, N_GLOB) #velocity array

F = 150/8  # Focal ratio
M = 1.07  # Output beam quality
wavelength = 1.064E-6  # Central Emission Wavelength (m)

r_f0 = 2 * wavelength * F * M / np.pi  # focal radius (m)
z_r = 2 * r_f0 * F  # Rayleigh Length (m)
r_f = 0.625e-3 #focal radius (m)

os.chdir(r"C:\Users\angus.mcallister\Documents\TFF\Welding Sim Results\Sim Results R4")

D_GLOB = np.loadtxt(fname="Penetration Depth.txt", delimiter=" ")
W_T_GLOB = np.loadtxt(fname="Top Melt Width.txt", delimiter=" ")
W_B_GLOB = np.loadtxt(fname="Bottom Melt Width.txt", delimiter=" ")

for i in range(0, N_GLOB):
    for j in range(0, N_GLOB):
        if D_GLOB[i,j] == 0:
            D_GLOB[i,j] = None
            W_T_GLOB[i,j] = None
            W_B_GLOB[i,j] = None

v_arr, P_L_arr= np.meshgrid(v_arr, P_L_arr)

fig1, ax1 = plt.subplots(constrained_layout=True)
clev = np.arange(np.nanmin(D_GLOB), np.nanmax(D_GLOB), 0.01)
CS1 = ax1.contourf(v_arr*1000, P_L_arr, np.transpose(D_GLOB), clev, cmap = plt.cm.coolwarm)
ax1.set_xlabel("Welding Speed (mm/s)")
ax1.set_ylabel("Laser Power (W)")
ax1.set_title("CW Nd:YAG Laser Welding Analysis \n ($r_f$ = " +str(r_f*1000)+ " mm, F = " +str(np.round(F,2))+ ")")
cbar1 = fig1.colorbar(CS1, location="bottom")
cbar1.ax.set_xlabel("Penetration Depth (mm)")