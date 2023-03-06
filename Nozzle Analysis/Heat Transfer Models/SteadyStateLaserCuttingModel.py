import scipy as sp
from scipy.special import kv
from scipy.special import iv
import scipy.integrate as integrate
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import math
import time

#Please see "An Analytical Model of Metal Cutting with a Laser Beam" for an explanation of the underlying physics
#Analysis conducted for nominal cutting setup on mild steel, 8 thou (0.2 mm) thick

#Processing Parameters
t = time.time()

V = 20E-3 #Translational Cutting Speed (m/s)

T_0 = 300 #Initial Temperature
Z_0 = 0.1E-3 #Laser Beam Focal Position From Top of Material (m)
d = 0.2E-3 #Depth of material (m)
P_a = 101325 #Ambient Pressure (Pa)

#Laser Parameters

F = 2  #Focal ratio
M = 1.07   #Output beam quality
wavelength = 1.07E-6 #Central Emission Wavelength (m)
P_L = 140 #Nominal output power (W)

r_f0 = 2*wavelength*F*M/np.pi #focal radius (m)
z_r = 2*r_f0*F #Rayleigh Length (m)

w = r_f0*np.sqrt(1+(-Z_0/z_r)**2)
w_b = r_f0*np.sqrt(1+((d-Z_0)/z_r)**2)

I_0 = P_L/(np.pi*w**2) #Peak Intensity
I_0_b = P_L/(np.pi*w_b**2)
alpha_max = 0.41 #Max Fresnel Absorption

#Material Parameters


k = 7.6E-6 #Thermal Diffusivity (m^2/s)
L = 39 #Thermal Conductivity (W/mK)
T_0 = 300 #Ambient Temperature (K)

T_m = 1800 #Melting Temperature (K)
delta = 1.8 #Surface Tension of steel (N/m)
D_eff = 2e-7 #Diffusion constant of oxygen in mild steel (m^2/s)
Ar_Fe = 55.845 #Relative atomic mass of iron
rho_St = 7800 #Density of steel (kg/m^3)
rho_St_m = 6822.56 #Density of molten steel
H_FeO = 2.57e5 #Reaction heat of oxidation to FeO (J/mol)
H_m = 2.7e5 #Enthalpy of melting 
C_p = 502 #Specific heat capacity (J/kgK)
C_pm = 890 #Specific heat capacity of molten material (scaled by iron)
L_f = 260 #Latent heat of fusion of steel

#Gas Parameters

V_g = 300 #Gas Velocity (m/s)
P_g = 1.51E6 + P_a #Gas Pressure (Pa)
rho_g = 1.28 #Gas Density (kg/m^3)
mu_g = 1.82e-5 #Gas Viscosity (kg/ms)

#Setup matrices to iterate on r_k and x_t

r_k_min = 1E-6
r_k_max = 100e-6
r_k = np.linspace(r_k_min,r_k_max,num=500)

x_t_min = 1E-6
x_t_max = 100e-6
x_t = np.linspace(x_t_min,x_t_max,num=500)

K = 5

Pe_dash = V/(2*k)# Adjusted Peclet number
W_K = 30E-6
theta_min = 0
theta_max = 30*np.pi/180
theta = np.linspace(theta_min,theta_max,num=100)
V_m = np.zeros(len(theta))

def find_Vm(theta):
        F_0 = d*W_K*(np.pi/2)*P_g
        F_n = d*W_K*(np.pi/2)*rho_g*V_g**2*np.tan(theta)
        F_t = np.sqrt(d)*W_K*(np.pi/2)*np.sqrt(rho_g*mu_g)*2*np.sqrt(V_g**3)
        F_st = d*W_K*(np.pi/2)*(2*delta/W_K)

        Coef_Vm3 = W_K*(np.pi/2)*mu_g/V_g
        Coef_Vm2 = d*W_K*(np.pi/2)*rho_g*V_g/3
        Coef_Vm1 = F_st-F_0-F_n-F_t
        Coef_Vm0 = W_K*P_a*V_g*d
        Coefs_Vm = [Coef_Vm3, Coef_Vm2, Coef_Vm1, Coef_Vm0]

        Vm = np.roots(Coefs_Vm)
        Vm = min([n for n in Vm  if n>0])
        return(Vm)

Gamma = 1 + (rho_St_m*C_pm*(T_s[i]-T_m))/(rho_St_m*L_f+rho_St*C_p*(T_m-T_0)
S_m = (k/V_d[i])*np.log(Gamma)
V_v = 5000*np.exp(-U/T_s[i])

