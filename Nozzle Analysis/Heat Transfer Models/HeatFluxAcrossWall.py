import scipy as sp
from scipy.special import kv
from scipy.special import iv
import scipy.integrate as integrate
import numpy as np
import matplotlib.pyplot as plt

#Processing Parameters

V = 18E-3 #Translational Cutting Speed (m/s)
T_0 = 300 #Initial Temperature
Z_0 = 0.05E-3 #Laser Beam Focal Position From Top of Material (m)

#Laser Parameters

F = 2  #Focal ratio
M = 1.07   #Output beam quality
wavelength = 1.07E-6 #Central Emission Wavelength (m)
P_L = 150 #Nominal output power (W)

r_f0 = 2*wavelength*F*M/np.pi #focal radius (m)
z_r = 2*r_f0*F #Rayleigh Length (m)
I_0 = 2*P_L/(r_f0**2*np.pi) #Peak Intensity

psi_min = 0
psi_max = np.pi/2
psi = np.linspace(psi_min,psi_max,num=100)

#Material Parameters

k = 7.6E-6 #Thermal Diffusivity (m^2/s)
L = 39 #Thermal Conductivity (W/mK)
T_m = 1450 #Melting Temperature

R_K = 23E-6
Pe_dash = V/(2*k)
Pe = Pe_dash*R_K

K = 5

q1 = np.zeros((len(psi),K))
q = np.zeros(len(psi))

for i in range(0,len(psi)):

    for j in range(1,K):
            
        q1[i,j] = (sp.special.iv(j,Pe)/sp.special.kv(j,Pe))*(sp.special.kv(j-1,Pe_dash*R_K)-2*np.cos(psi[i])*sp.special.kv(j,Pe_dash*R_K)+sp.special.kv(j+1,Pe_dash*R_K))*np.cos(j*psi[i])

    q[i] = L*(T_m-T_0)*Pe_dash*np.exp(-Pe_dash*R_K*np.cos(psi[i]))*((sp.special.iv(0,Pe)/sp.special.kv(0,Pe))*(-np.cos(psi[i])*sp.special.kv(0,Pe_dash*R_K)+sp.special.kv(1,Pe_dash*R_K))+np.sum(q1[i,:]))

P_1 = sp.integrate.simps(q,psi)

plot1 = plt.figure(1)
plt.plot(psi/(np.pi/180),q/1000000)
plt.xlabel('Angle (deg)')
plt.ylabel('Heat Flux (MW/m^2)')
plt.grid()

plt.show()
