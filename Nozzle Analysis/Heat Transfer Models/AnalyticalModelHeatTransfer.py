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

V = 35E-3 #Translational Cutting Speed (m/s)

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

I_0 = P_L/(w**2*np.pi) #Peak Intensity
I_0_b = P_L/(w_b**2*np.pi) #Peak Intensity at bottom of kerf

alpha_max = 0.41 #Max Fresnel Absorption

#Material Parameters


k = 7.6E-6 #Thermal Diffusivity (m^2/s)
L = 39 #Thermal Conductivity (W/mK)
T_m = 1800 #Melting Temperature (K)
delta = 1.8 #Surface Tension of steel (N/m)
rho_St = 7800 #Density of steel (kg/m^3)
H_m = 2.7e5 #Enthalpy of melting 
C_p = 670 #Specific heat capacity (J/kgK)

#Gas Parameters

V_g = 126.8 #Gas Velocity (m/s)
P_g = 1.6E6 + P_a #Gas Pressure (Pa)
rho_g = 18.8 #Gas Density (kg/m^3)
mu_g = 1.912e-5 #Gas Viscosity (kg/ms)


#Setup matrices to iterate on r_k and x_t

r_k_min = 1E-6
r_k_max = 100e-6
r_k = np.linspace(r_k_min,r_k_max,num=500)

x_t_min = 1E-6
x_t_max = 100e-6
x_t = np.linspace(x_t_min,x_t_max,num=500)

K = 5

Pe_dash = V/(2*k)# Adjusted Peclet number

#Setup zero matrices

q_side1 = np.zeros((len(r_k),K))
q_front1 = np.zeros((len(r_k),K))
q_side2 = np.zeros(len(r_k))
q_front2 = np.zeros(len(r_k))

Pe = np.zeros(len(r_k))
I_L = np.zeros(len(r_k))
LHS_side = np.zeros(len(r_k))

R = np.zeros((len(x_t),len(r_k)))
I_L_side = np.zeros((len(x_t),len(r_k)))
I_L_front = np.zeros(len(x_t))
LHS_front = np.zeros(len(x_t))
sol_R = np.zeros(len(x_t))
LHS_front2 = np.zeros(len(x_t))
sol_Xt = np.zeros(len(x_t))

#Define tolerance which rk and xt can be within

tol = 0.05

##Brute force calculating r and xt

def find_geometry():
        for k in range(len(x_t)):
                for j in range(len(r_k)):
                        
                        Pe[j] = Pe_dash*r_k[j] #Peclet number

                        for i in range(1,K):
                            
                                q_front1[j,i-1] = (sp.special.iv(i,Pe[j])/sp.special.kv(i,Pe[j]))*(sp.special.kv(i-1,Pe[j])-2*sp.special.kv(i,Pe[j])+sp.special.kv(i+1,Pe[j]))#Heat flux for moving cylinder at front
                                q_side1[j,i-1] = (sp.special.iv(2*i,Pe[j])/sp.special.kv(2*i,Pe[j]))*(sp.special.kv(2*i-1,Pe[j])+sp.special.kv(2*i+1,Pe[j]))*np.cos(i*np.pi)#Heat flux for moving cylinder at side
                            
                        q_front2[j] = L*(T_m-T_0)*Pe_dash*np.exp(-Pe[j])*((sp.special.iv(0,Pe[j])/sp.special.kv(0,Pe[j]))*(-sp.special.kv(0,Pe[j])+sp.special.kv(1,Pe[j]))+np.sum(q_front1[j,:]))
                        q_side2[j] = L*(T_m-T_0)*Pe_dash*((sp.special.iv(0,Pe[j])/sp.special.kv(0,Pe[j]))+np.sum(q_side1[j,:]))

                        R[k,j] = np.sqrt((x_t[k]-r_k[j])**2+r_k[j]**2)#Radius of side of cylinder
                        I_L_side[k,j] = I_0*np.exp((-2*R[k,j]**2)/w**2)#Laser power intensity at side

                        LHS_side[j] = q_side2[j]/(0.41*np.tan(3*np.pi/180))#LHS of equation 5 in paper

                        I_L_front[k] = I_0*np.exp((-2*x_t[k]**2)/w**2)#Laser power intensity at front
                        LHS_front[j] = q_front2[j]/(0.37*np.tan(6*np.pi/180))#LHS of equation 4 in paper

                        Error_Side = abs(1-LHS_side[j]/I_L_side[k,j])#Define error at side
                        Error_Front = abs(1-LHS_front[j]/I_L_front[k])#Define error at front
                        
                        if Error_Side<tol and Error_Front<tol:#Conditional statements to find approximate value of rk and xt
                                return[r_k[j], x_t[k], Error_Front, Error_Side]

sol_geo = find_geometry()


R_K = sol_geo[0]
W_K = R_K*2
X_t = sol_geo[1]

P_Error = 1.01
                
theta = np.arctan(X_t/d)# Defines starting cut angle such that x_b at bottom starts at x = 0
theta_step = 0.001

while P_Error>1:

        k = 1
        theta += theta_step
        X_b = X_t - d*np.tan(theta)

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
        Vm = min(abs(Vm))

        S_m = V*d/Vm

        Pe = Pe_dash*R_K
        q_front1 = np.zeros(K)

        for i in range(1,K):
                q_front1[i-1] = (sp.special.iv(i,Pe)/sp.special.kv(i,Pe))*(sp.special.kv(i-1,Pe)-2*sp.special.kv(i,Pe)+sp.special.kv(i+1,Pe))#Heat flux for moving cylinder at front

        q_front2 = L*(T_m-T_0)*Pe_dash*np.exp(-Pe)*((sp.special.iv(0,Pe)/sp.special.kv(0,Pe))*(-sp.special.kv(0,Pe)+sp.special.kv(1,Pe))+np.sum(q_front1))

        T_L = T_m + (S_m/L)*q_front2

        A_1 = 2/(w**2)
        A_2 = 2/(w_b**2)
        P_ab_1 = alpha_max*I_0*np.pi*(1-np.exp(-A_1*X_t**2))/(2*A_1)
        P_ab_2 = alpha_max*I_0_b*np.pi*(np.exp(-A_2*X_b**2)-1)/(2*A_2)
        
        psi_min = 0
        psi_max = np.pi/2
        psi = np.linspace(psi_min,psi_max,num=100)

        q1 = np.zeros((len(psi),K))
        q = np.zeros(len(psi))

        for i in range(0,len(psi)):

            for j in range(1,K):
                    
                q1[i,j] = (sp.special.iv(j,Pe)/sp.special.kv(j,Pe))*(sp.special.kv(j-1,Pe_dash*R_K)-2*np.cos(psi[i])*sp.special.kv(j,Pe_dash*R_K)+sp.special.kv(j+1,Pe_dash*R_K))*np.cos(j*psi[i])

            q[i] = L*(T_m-T_0)*Pe_dash*np.exp(-Pe_dash*R_K*np.cos(psi[i]))*((sp.special.iv(0,Pe)/sp.special.kv(0,Pe))*(-np.cos(psi[i])*sp.special.kv(0,Pe_dash*R_K)+sp.special.kv(1,Pe_dash*R_K))+np.sum(q1[i,:]))

        P_1 = sp.integrate.simps(q,psi)
        P_mcs = 2*R_K*d*P_1

        P_m = rho_St*V*d*W_K*H_m
        P_La = rho_St*V*d*W_K*C_p*0.5*(T_L-T_m)
        P_LHS = P_ab_1+abs(P_ab_2)
        P_RHS = P_mcs + P_m + P_La

        P_Error = P_RHS/P_LHS

        
We = rho_St*Vm**2*S_m/delta

print(P_m)
print(P_La)
print(P_mcs)

elapsed = time.time() - t
