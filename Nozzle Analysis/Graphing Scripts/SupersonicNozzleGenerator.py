# Supersonic Nozzle Generator
# This script is based on "The Analytical Design of an Axially Symmetric Laval Nozzle for a Parallel and Uniform Jet" by Kuno Foelsch
import numpy as np
import matplotlib.pyplot as plt
import os

# Parameters:

P_0 = 0.5E6  # Inlet Pressure (Pa)
P_e = 101325  # Ambient Pressure (Pa)
D_C = 5  # Throat Diameter (mm)
M_C = 1  # Choked Mach No
gamma = 1.66  # Gas constant (c_p/c_v)
L_N = 15  # Length of Nozzle (mm)
R_0 = 8  # Inlet Radius (mm)
A_C = 0.25*np.pi*(D_C/1000)**2 #Choked area (m^2)
T_0 = 300 #Total temperature (K)
R_air = 208.13 #Gas constant for air (J/kgK)

# 1D Supersonic Flow:

M_E = np.sqrt((2 / (gamma - 1)) * ((P_0 / P_e) ** ((gamma - 1) / gamma) - 1)) # Exit Mach No

Ratio_EtoC = np.sqrt(
    (1 / M_E) * ((1 + M_E ** 2 * (gamma - 1) / 2) / (1 + (gamma - 1) / 2)) ** ((gamma + 1) / (2 * (gamma - 1))))

D_E = Ratio_EtoC * D_C #Exit Diameter

# Foelsch's Method Step 1:

psi_E = 0.5*(np.sqrt((gamma+1)/(gamma-1))*np.arctan(np.sqrt(((gamma-1)/(gamma+1))*(M_E**2 - 1))) - np.arctan(np.sqrt(M_E**2 - 1)))
tau_E = np.sqrt((((2/(gamma+1)) + ((gamma-1)/(gamma+1))*M_E**2)**((gamma+1)/(2*(gamma-1))))/M_E)

# Foeslch's Method Step 2:

psi_A = psi_E/2

# Need to implement numerical solver to find tau_A and M_A:

M = np.linspace(M_C, M_E, num=100)
K_1 = np.sqrt((gamma + 1) / (gamma - 1))
K_2 = np.sqrt((gamma - 1) / (gamma + 1))

def find_A_Te():
    Eqn_1 = np.zeros(len(M))
    Eqn_2 = np.zeros(len(M))
    A = np.zeros(len(M))

    for i in range(0, len(M)):

        A[i] = np.sqrt(M[i] ** 2 - 1)
        Eqn_1[i] = 0.5 * (K_1 * np.arctan(K_2 * A[i]) - np.arctan(A[i])) - psi_A
        Eqn_2[i] = np.sqrt(
            (1 / M[i]) * ((2 / (gamma + 1)) + (K_2 ** 2) * M[i] ** 2) ** ((gamma + 1) / (2 * (gamma - 1))))

        if Eqn_1[i] > 0:
            sol_A = A[i]
            sol_Te = Eqn_2[i]
            return [sol_A, sol_Te]


sol = find_A_Te()

A_start = sol[0] #This is sqrt(M_A^2 -1) in literature
A_end = np.sqrt(M_E ** 2 - 1)
tau_A = sol[1]

# Foelsch's Method Part 3:

A = np.linspace(A_start, A_end, num=100)
x = np.zeros(len(A))
y = np.zeros(len(A))
M = np.zeros(len(A))
psi_P = np.zeros(len(A))

for i in range(0, len(A)):
    M[i] = np.sqrt(A[i] ** 2 + 1)
    psi_P[i] = 0.5 * (K_1 * np.arctan(K_2 * A[i]) - np.arctan(A[i]))
    theta_P = psi_E - psi_P[i]
    F = np.sqrt(
        (np.sin(theta_P) ** 2) + 2 * (np.cos(theta_P) - np.cos(psi_A)) * (A[i] * np.sin(theta_P) + np.cos(theta_P)))
    tau_P = np.sqrt(
        (1 / M[i]) * ((2 / (gamma + 1)) + ((gamma - 1) / (gamma + 1)) * M[i] ** 2) ** ((gamma + 1) / (2 * (gamma - 1))))
    y[i] = (D_E / (4 * np.sin(psi_A / 2))) * (tau_P / tau_E) * F
    x_1 = (D_E / (4 * np.sin(psi_A / 2))) * (tau_P / tau_E) * (
                (1 + F * (np.cos(theta_P) * A[i] - np.sin(theta_P))) / (np.sin(theta_P) * A[i] + np.cos(theta_P)))

    #Foelsch's Method Part 4:
    x_0 = (D_E / (2 * tau_E)) * ((1 / np.tan(psi_A)) - (tau_A * np.cos(psi_A / 2) - 1) / (
                2 * np.cos(psi_A / 2) * (np.sin(psi_A / 2) + np.cos(psi_A / 2))))
    x[i] = x_1 - x_0

#Foeslch's Method Part 5:
R = (D_E / (4 * tau_E * np.sin(psi_A / 2))) * ((tau_A * np.cos(psi_A / 2)) - 1) / (
            np.cos(psi_A / 2) + np.sin(psi_A / 2))
x_D = R * np.sin(psi_A)
y_D = (D_E / 2) + R * (1 - np.cos(psi_A))
L = (D_E / (4 * np.sin(psi_A / 2))) + (D_E / 2) * (np.sqrt(M_E ** 2 - 1)) - x_0
x += (L_N - x[-1])

# Calculate Converging Section Co-ordinates (Based on "Design and Characteristic Analysis of Supersonic Nozzles for
# High Gas Pressure Laser Cutting")

L_0 = L_N - L #Length of converging section (chosen arbitrarily, unclear exactly how this impacts sonic flow field)
x_c = np.linspace(0, L_0, num=100)
alpha = np.zeros(len(x_c))
radius_convergent = np.zeros(len(x_c))
R_C = D_C / 2 # Throat radius

for i in range(len(x_c)):
    alpha[i] = x_c[i] / L_0
    radius_convergent[i] = R_C / np.sqrt(
        1 - (1 - (R_C / R_0) ** 2) * ((1 - alpha[i] ** 2) ** 2 / (1 + alpha[i] ** 2 / 3)))

plot1 = plt.figure(1)
plt.plot(x, y, 'b', label="Diverging")
plt.plot(x_c, radius_convergent, 'r', label="Converging")
plt.xlabel("Axial (mm)")
plt.ylabel("Radial (mm)")
plt.grid()
plt.legend()
plt.show()

XY_div = np.zeros((len(x), 3))
XY_conv = np.zeros((len(x_c), 3))

XY_div[:, 0] = np.round(x, 4)
XY_div[:, 1] = np.round(y, 4)
XY_div[:, 2] = np.zeros(len(x))

XY_conv[:, 0] = np.round(x_c, 4)
XY_conv[:, 1] = np.round(radius_convergent, 4)
XY_conv[:, 2] = np.zeros(len(x_c))

mdot = (A_C*P_0/np.sqrt(T_0))*np.sqrt(gamma/R_air)*((gamma+1)/2)**(-(gamma+1)/(2*(gamma-1)))
mdot_str = str(round(mdot*3600,2))

P_crit = P_0*((gamma+1)/2)**(-gamma/(gamma-1))
T_crit = T_0/((gamma+1)/2)
rho_crit = P_crit/(R_air*T_crit)

T_e = T_0*(1 + ((gamma-1)/2)*M_E**2)**(-1)

#The following is taken from "Numerical Study of Characteristics of Underexpanded Supersonic Jets" by Murugeessan et al.
# Can be accessed at "https://www.scielo.br/j/jatm/a/frWwNFxSBjtrGkkfkNLmDCJ/?format=pdf&lang=en#:~:text=They%20considered%20supersonic%20jets%20having,exit%20diameter%20of%20the%20nozzle."

f_y = -0.0276*gamma**2 + 0.0448*gamma + 0.113
NPR = P_0/P_e
L_C = D_E*np.exp(-f_y*M_E**2)*(-0.0303*NPR**2 + 3.5002*NPR + 4.085)#equation 6

print("Mass Flow Rate = " +mdot_str+ " kg/hr")
print("Exit Mach Number = " +str(round(M_E,2))+ "" )
print("Supersonic Core Length = " +str(round(L_C, 2))+ "mm")

os.chdir(r"R:\Nozzle Analysis\Supersonic Nozzle\Data Points")
np.savetxt("Divergent Section XY Points.txt", XY_div, delimiter=" ", fmt='%f')
np.savetxt("Convergent Section XY Points.txt", XY_conv, delimiter=" ", fmt='%f')