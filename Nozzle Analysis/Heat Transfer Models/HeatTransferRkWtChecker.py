import scipy as sp
from scipy.special import kv
from scipy.special import iv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter


#Please see "An Analytical Model of Metal Cutting with a Laser Beam" for an explanation of the underlying physics
#Analysis conducted for nominal cutting setup on mild steel, 8 thou (0.2 mm) thick

#Processing Parameters

V = 18E-3 #Translational Cutting Speed (m/s)
T_0 = 300 #Initial Temperature
Z_0 = 0.05E-3 #Laser Beam Focal Position From Top of Material (m)

#Laser Parameters

F = 2  #Focal ratio
M = 1.07   #Output beam quality
wavelength = 1.07E-6 #Central Emission Wavelength (m)
P_L = 140 #Nominal output power (W)

r_f0 = 2*wavelength*F*M/np.pi #focal radius (m)
z_r = 2*r_f0*F #Rayleigh Length (m)

w = r_f0*np.sqrt(1+(-Z_0/z_r)**2)
I_0 = 2*P_L/(w**2*np.pi) #Peak Intensity

#Material Parameters

k = 7.6E-6 #Thermal Diffusivity (m^2/s)
L = 39 #Thermal Conductivity (W/mK)
T_m = 1450 #Melting Temperature

r_k = 2.2525252525E-5
x_t = 1.712121212E-5
Pe_dash = V/(2*k)
Pe = Pe_dash*r_k
K=5
q_front1 = np.zeros(K)
q_side1 = np.zeros(K)

for i in range(1,K):
    
        q_front1[i-1] = (sp.special.iv(i,Pe)/sp.special.kv(i,Pe))*(sp.special.kv(i-1,Pe)-2*sp.special.kv(i,Pe)+sp.special.kv(i+1,Pe))
        q_side1[i-1] = (sp.special.iv(2*i,Pe)/sp.special.kv(2*i,Pe))*(sp.special.kv(2*i-1,Pe)+sp.special.kv(2*i+1,Pe))*np.cos(i*np.pi)
    
q_front2 = L*(T_m-T_0)*Pe_dash*np.exp(-Pe)*((sp.special.iv(0,Pe)/sp.special.kv(0,Pe))*(-sp.special.kv(0,Pe)+sp.special.kv(1,Pe))+np.sum(q_front1))
q_side2 = L*(T_m-T_0)*Pe_dash*((sp.special.iv(0,Pe)/sp.special.kv(0,Pe))+np.sum(q_side1))

R = np.sqrt((x_t-r_k)**2+r_k**2)
I_L_side= I_0*np.exp((-2*R**2)/w**2)

LHS_side = q_side2/(0.41*np.tan(3*np.pi/180))

I_L_front = I_0*np.exp((-2*x_t**2)/w**2)
LHS_front = q_front2/(0.37*np.tan(6*np.pi/180))

Error_Side = abs(1-LHS_side/I_L_side)
Error_Front = abs(1-LHS_front/I_L_front)
