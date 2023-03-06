#Heat Transfer Script

import numpy as np
import matplotlib.pyplot as plt

#Material Constants

C_s = 0.628 #Specific heat of solid iron (J/gK)
C_m = 0.748 #Specific heat of liquid iron (J/gK)
L_m = 276 #Latent heat of melt (J/g)
L_v = 6088 #Latent heat of evaporation (J/g)
T_m = 1808 #Melting point of iron (K)
T_0 = 300 #Initial temperature (K)
a_s = 0.14 #Thermal diffusivity (cm^2/s)
a_m = 0.07 #Thermal diffusivity (cm^2/s)
rho_s = 7.8 #Density of solid (g/cm^3)
rho_m = 6.98 #Density of melt (g/cm^3)
B_0 = 3.9e13 #Vaporisation constant (g/cm/s^2)
M_a = 55.847 #Atomic mass of iron (g/mol)
N_a = 6.02214086e23 #Avogadro's constant
k_b = 1.38064852e-22 #Boltzmann's constant
U = M_a*L_v/(N_a*k_b) #Calculation Constant
#Experimental constants

alpha = 0.5 #given constant
A = 0.55 #given constant
T_s_min = 1850 #Minimum surface temperature (K)
T_s_max = 7000 #Maximum surface temperature (K)
N_step = 100
V_0 = 3.8e5 #speed of sound in iron in condensed phase (cm/s)

r_l = 1.9e-2

#Matrix Setup

T_s = np.linspace(T_s_min, T_s_max, num = N_step)

V_v = np.zeros((len(T_s),1))
V_dv = np.zeros((len(T_s),1))
P_r = np.zeros((len(T_s),1))
V_m = np.zeros((len(T_s),1))
V_d = np.zeros((len(T_s),1))
V_dm = np.zeros((len(T_s),1))
T_star = np.zeros((len(T_s),1))
I_conv = np.zeros((len(T_s),1))
I_cond = np.zeros((len(T_s),1))
I_evap = np.zeros((len(T_s),1))
I_abs = np.zeros((len(T_s),1))
CONV = np.zeros((len(T_s),1))
COND = np.zeros((len(T_s),1))
EVAP = np.zeros((len(T_s),1))

for i in range(len(T_s)):

    V_v[i,0] = V_0 * np.exp(-U / T_s[i])
    V_dv[i,0] = (rho_m / rho_s) * V_v[i,0]

    P_r[i,0] = A * B_0 * (1 / np.sqrt(T_s[i])) * np.exp(-U / T_s[i])
    V_m[i,0] = np.sqrt(2 * P_r[i,0] / rho_m)
    V_d[i,0] = 0.5 * ((rho_m * V_v[i,0] / rho_s) + np.sqrt(
        (rho_m * V_v[i,0] / rho_s) ** 2 + 4 * (rho_m * a_m * V_m[i,0] / (rho_s * r_l))))
    V_dm[i,0] = V_d[i,0] - V_dv[i,0]

    T_star[i,0] = T_m + alpha * (T_s[i] - T_m)

    I_conv[i,0] = rho_m * (C_m * T_star[i,0] + L_m) * V_m[i,0] * (a_m / (V_d[i,0] * r_l)) - rho_s * C_s * T_0 * \
                   V_d[i,0]
    I_cond[i,0] = (rho_s * C_s * (T_m - T_0) * V_d[i,0]) / np.sqrt((a_m / a_s) + (V_d[i,0] * r_l / a_s))
    I_evap[i,0] = rho_m * V_d[i,0] * L_v
    I_abs[i,0] = I_conv[i,0] + I_cond[i,0] + I_evap[i,0]

    CONV[i,0] = I_conv[i,0] / I_abs[i,0]
    COND[i,0] = I_cond[i,0] / I_abs[i,0]
    EVAP[i,0] = I_evap[i,0] / I_abs[i,0]

                  
#Plotting

plot1 = plt.figure(1)
plt.plot(I_abs/1e6,T_s)
plt.xlabel("I_abs (MW/cm^2)")
plt.ylabel("T_s (K)")
plt.title("Melt Surface Temperature vs Absorbed Laser Intensity for Z = 0.1 mm ")
plt.grid()

plot3 = plt.figure(3)
plt.plot(I_abs/1e6,V_d/100,label = "Total Drilling Velocity")
plt.plot(I_abs/1e6,V_dv/100,label = "Evaporation Component")
plt.plot(I_abs/1e6,V_dm/100,label = "Melt Ejection Component")
plt.xlabel("I_abs (MW/cm^2)")
plt.ylabel("V (m/s)")
plt.title("Drilling Velocity vs Absorbed Laser Intensity for Z = 0.1 mm")
plt.grid()
plt.legend()
