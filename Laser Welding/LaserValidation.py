import numpy as np
import matplotlib.pyplot as plt

#Laser Parameters

F =150/8   #Focal ratio (ratio of focal distance to collimated beam diameter)
M = 1.07   #Output beam quality
wavelength = 1.064E-6 #Central Emission Wavelength (m)
P_L = 1000 #Nominal output power (W)

r_f0 = 2*wavelength*F*M/np.pi #focal radius (m)
z_r = 2*r_f0*F #Rayleigh Length (m)
r_ft = 6e-4 #Spot radius at top of material (m)
I_0 = 2*P_L/(np.pi*r_f0**2) #Beam laser intensity at focal radius (W/m^2)
angular_freq = 2*np.pi*299792458/wavelength #angular frequency of laser wave (rad/s)
z_0 = -0.02249

dz = 1e-6
z = 1500e-6
N_z_steps = int(z/dz)
z_arr = np.linspace(0, z, N_z_steps)
r_f = np.zeros(N_z_steps)

for i in range(0, N_z_steps):
    r_f[i] = r_f0 *np.sqrt(1 + ((z_arr[i] - z_0 ) /z_r )**2)

plt.plot(z_arr, r_f)
