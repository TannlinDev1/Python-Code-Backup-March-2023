import numpy as np
import matplotlib.pyplot as plt

#Laser Parameters

F = 80/5  #Focal ratio (ratio between clear aperture and focal distance)
M = 1.07   #Output beam quality
wavelength = 1.064E-6 #Central Emission Wavelength (m)
P_L = 400 #Nominal output power (W)

r_f0 = 2*wavelength*F*M/np.pi #focal radius (m)
z_r = 2*r_f0*F #Rayleigh Length (m)
r_f = r_f0*1.5 #Spot radius (m)
z = z_r * np.sqrt((r_f / r_f0) ** 2 - 1)

# z = np.linspace(-80e-3, 1.2e-3, 1000)
# rad = np.zeros(1000)
#
# for i in range(0, 1000):
#     rad[i] = r_f0*np.sqrt(1 + (z[i]/z_r)**2)
#
# plt.plot(z*1000, rad*1000)


def intensity(P, w_0, z_r, z, r):
    w_f0 = w_0**2 * (1 + (z/z_r)**2)
    I = 2*P/(np.pi*w_f0)*np.exp((-2*r**2)/w_f0)
    return I

P1 = 1.467*P_L
P2 = 0.467*P_L
w1 = r_f0
w2 = 0.7*r_f0
z_1r = 1.42*z_r
z_2r = 1.136*z_r

rad_min = -r_f
rad_max = r_f
N_steps = 1000
rad = np.linspace(rad_min, rad_max, N_steps)
I_TH = np.zeros(N_steps)
I_nom = np.zeros(N_steps)

for i in range(0, N_steps):
    I_TH[i] = intensity(P1, w1, z_1r, z, rad[i]) - intensity(P2, w2, z_2r, z, rad[i])
    I_nom[i] = intensity(P_L, r_f0, z_r, z, rad[i])

plt.plot(rad*1000, I_TH, label="Top Hat")
plt.plot(rad*1000, I_nom, label="Gaussian")
plt.legend()