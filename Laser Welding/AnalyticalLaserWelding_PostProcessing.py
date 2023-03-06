import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

r_f_min = 0.275e-3 #minimum spot radius (m)
r_f_max = (1.65e-3)/2 #maximum spot radius (m)
N_GLOB_RF = 100 #number of spot radius iterations
r_f_arr = np.linspace(r_f_min, r_f_max, N_GLOB_RF) #focal radius array

v_min = 1e-3 #minimum weld speed (m/s)
v_max = 30e-3 #maximum weld speed (m/s)
N_GLOB_V = 100
v_arr = np.linspace(v_min, v_max, N_GLOB_V) #velocity array

F = 5.56  # Focal ratio
M = 1.07  # Output beam quality
wavelength = 1.064E-6  # Central Emission Wavelength (m)
P_L = 1000  # Nominal output power (W)

r_f0 = 2 * wavelength * F * M / np.pi  # focal radius (m)
z_r = 2 * r_f0 * F  # Rayleigh Length (m)

z_arr = np.zeros(N_GLOB_RF)

for i in range(0, N_GLOB_RF):
    z_arr[i] = z_r*np.sqrt((r_f_arr[i]/r_f0)**2 - 1)

os.chdir(r"C:\Users\angus.mcallister\Documents\TFF\Welding Sim Results\Sim Results")

D_GLOB = np.loadtxt(fname="Penetration Depth.txt", delimiter=" ")
R_K_GLOB = np.loadtxt(fname="Keyhole Radius.txt", delimiter=" ")
W_T_GLOB = np.loadtxt(fname="Top Melt Width.txt", delimiter=" ")
W_B_GLOB = np.loadtxt(fname="Bottom Melt Width.txt", delimiter=" ")

zmax = np.zeros(N_GLOB_RF)
zmin = np.zeros(N_GLOB_RF)
dz = np.zeros(N_GLOB_RF)

for j in range(0, N_GLOB_RF):
    index_zmax = np.where(D_GLOB[j,:] == np.nanmax(D_GLOB[j,:]))
    zmax[j] = z_arr[index_zmax]
    index_zmin = np.where(D_GLOB[j,:] == np.nanmin(D_GLOB[j,:]))
    zmin[j] = z_arr[index_zmin]
    dz[j] = zmin[j] - zmax[j]

Coefs_dz = np.polyfit(v_arr, dz, 3)
v_linspace = np.linspace(np.min(v_arr), np.max(v_arr), 100)
LoBF_dz = np.polyval(Coefs_dz, v_linspace)

corr_matrix = np.corrcoef(dz, LoBF_dz)
r_sq = (corr_matrix[0,1])**2

plt.figure(1)
plt.scatter(v_arr*1000, dz*500,label="Simulated", color="tab:blue")
plt.plot(v_linspace*1000, LoBF_dz*500, label="Best Fit (R = "+str(np.round(r_sq,2))+ ")", color="tab:orange")
plt.xlabel("Welding Speed (mm/s)")
plt.ylabel("Bilateral Defocus Height Tolerance (mm)")
plt.legend()
plt.title("CW Nd:YAG Laser Welding Tolerance Analysis \n (P = 1000W, F = 5.56)")

vr_arr,  r_f_arr= np.meshgrid(v_arr, r_f_arr)
vz_arr, z_arr = np.meshgrid(v_arr, z_arr)

fig1, ax1 = plt.subplots(constrained_layout=True)
CS1 = ax1.contourf(vr_arr*1000, r_f_arr*2000, np.transpose(D_GLOB)*1000)
ax1.set_xlabel("Welding Speed (mm/s)")
ax1.set_ylabel("Spot Diameter (mm)")
ax1.set_title("CW Nd:YAG Laser Welding Analysis \n (P = 1000W, F = 5.56)")

ax2 = ax1.twinx()

ax2.contour(vz_arr*1000, z_arr*1000, np.transpose(D_GLOB)*1000)
ax2.set_ylabel("Defocus Height (mm)")

cbar1 = fig1.colorbar(CS1, location="bottom")
cbar1.ax.set_xlabel("Penetration Depth (mm)")

fig2, ax3 = plt.subplots(constrained_layout=True)
CS2 = ax3.contourf(vr_arr*1000, r_f_arr*2000, np.transpose(R_K_GLOB)*1000)
ax3.set_xlabel("Welding Speed (mm/s)")
ax3.set_ylabel("Spot Diameter (mm)")
ax3.set_title("CW Nd:YAG Laser Welding Analysis \n (P = 1000W, F = 5.56)")

ax4 = ax3.twinx()

ax4.contour(vz_arr*1000, z_arr*1000, np.transpose(R_K_GLOB)*1000)
ax4.set_ylabel("Defocus Height (mm)")

cbar2 = fig2.colorbar(CS2, location="bottom")
cbar2.ax.set_xlabel("Keyhole Radius (mm)")

fig3, ax5 = plt.subplots(constrained_layout=True)
CS3 = ax5.contourf(vr_arr*1000, r_f_arr*2000, np.transpose(W_T_GLOB)*1000)
ax5.set_xlabel("Welding Speed (mm/s)")
ax5.set_ylabel("Spot Diameter (mm)")
ax5.set_title("CW Nd:YAG Laser Welding Analysis \n (P = 1000W, F = 5.56)")

ax6 = ax5.twinx()

ax6.contour(vz_arr*1000, z_arr*1000, np.transpose(W_T_GLOB)*1000)
ax6.set_ylabel("Defocus Height (mm)")

cbar3 = fig3.colorbar(CS3, location="bottom")
cbar3.ax.set_xlabel("Top Melt Width (mm)")

fig4, ax7 = plt.subplots(constrained_layout=True)
CS4 = ax7.contourf(vr_arr*1000, r_f_arr*2000, np.transpose(W_B_GLOB)*1000)
ax7.set_xlabel("Welding Speed (mm/s)")
ax7.set_ylabel("Spot Diameter (mm)")
ax7.set_title("CW Nd:YAG Laser Welding Analysis \n (P = 1000W, F = 5.56)")

ax8 = ax7.twinx()

ax8.contour(vz_arr*1000, z_arr*1000, np.transpose(W_B_GLOB)*1000)
ax8.set_ylabel("Defocus Height (mm)")

cbar4 = fig4.colorbar(CS4, location="bottom")
cbar4.ax.set_xlabel("Bottom Melt Width (mm)")
