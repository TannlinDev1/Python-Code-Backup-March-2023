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

T_0 = 300 #Initial Temperature
Z_0 = 0.05E-3 #Laser Beam Focal Position From Top of Material (m)
d = 0.2E-3 #Depth of material (m)
P_a = 101325 #Ambient Pressure (Pa)

#Laser Parameters

F = 2  #Focal ratio
M = 1.07   #Output beam quality
wavelength = 1.07E-6 #Central Emission Wavelength (m)
P_L = 12.5 #Nominal output power (W)

r_f0 = 2*wavelength*F*M/np.pi #focal radius (m)
z_r = 2*r_f0*F #Rayleigh Length (m)

w = r_f0*np.sqrt(1+(-Z_0/z_r)**2)
w_b = r_f0*np.sqrt(1+((d-Z_0)/z_r)**2)

I_0 = 2*P_L/(w**2*np.pi) #Peak Intensity

alpha_max = 0.41 #Max Fresnel Absorption

#Material Parameters

k = 7.6E-6 #Thermal Diffusivity (m^2/s)
L = 39 #Thermal Conductivity (W/mK)
T_m = 1450 #Melting Temperature (K)
delta = 1.8 #Surface Tension of steel (N/m)
D_eff = 2e-7 #Diffusion constant of oxygen in mild steel (m^2/s)
Ar_Fe = 55.845 #Relative atomic mass of iron
rho_St = 7800 #Density of steel (kg/m^3)
H_FeO = 2.57e5 #Reaction heat of oxidation to FeO (J/mol)
H_m = 4e5 #Enthalpy of melting 
C_p = 670 #Specific heat capacity (J/kgK)

#Gas Parameters

V_g = 290 #Gas Velocity (m/s)
rho_g = 1.28 #Gas Density (kg/m^3)
mu_g = 1.82e-5 #Gas Viscosity (kg/ms)

P_g_min = 1E6#Min Gas Pressure (Pa)
P_g_max = 1.8E6#Max Gas Pressure (Pa)
P_g = np.linspace(P_g_min,P_g_max,num=9)

#Translational Velocity Parameter Sweep

V = 20E-3

We = np.zeros(len(P_g))
angle = np.zeros(len(P_g))
Vm = np.zeros(len(P_g))
T_L = np.zeros(len(P_g))
S_m = np.zeros(len(P_g))


#Setup matrices to iterate on r_k and x_t
t = time.time()

r_k_min = 1E-6
r_k_max = 50e-6
r_k = np.linspace(r_k_min,r_k_max,num=500)

x_t_min = 1E-6
x_t_max = 50e-6
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
Error_Side = np.zeros((len(x_t),len(r_k)))
Error_Front = np.zeros((len(x_t),len(r_k)))

#Define tolerance which rk and xt can be within

tol = 0.05

#Brute force calculating r and rt

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

                        Error_Side[k,j] = abs(1-LHS_side[j]/I_L_side[k,j])#Define error at side
                        Error_Front[k,j] = abs(1-LHS_front[j]/I_L_front[k])#Define error at front
                        
                        if Error_Side[k,j]<tol and Error_Front[k,j]<tol:#Conditional statements to find approximate value of rk and xt
                                sol_R[j] = r_k[j]
                                sol_Xt[k] = x_t[k]
                                print('Front Energy Balance Error = ' +str(round(100*Error_Front[k,j],2))+'%')
                                print('Rear Energy Balance Error = '+str(round(100*Error_Side[k,j],2))+'%')
                                return

find_geometry()
                
R_K = float(sol_R[sol_R != 0])
X_t = float(sol_Xt[sol_Xt != 0])

W_K = R_K*2



for N in range(len(P_g)):
        P_Error = 1.01
        theta = 0
        theta_step = 0.0001
        k = 1
        while P_Error>1:
            
            theta += theta_step

            X_b = X_t - d*np.tan(theta)

            F_0 = d*W_K*(np.pi/2)*P_g[N]
            F_n = d*W_K*(np.pi/2)*rho_g*V_g**2*np.tan(theta)
            F_t = np.sqrt(d)*W_K*(np.pi/2)*np.sqrt(rho_g*mu_g)*2*np.sqrt(V_g**3)
            F_st = d*W_K*(np.pi/2)*(2*delta/W_K)

            Coef_Vm3 = W_K*(np.pi/2)*mu_g/V_g
            Coef_Vm2 = d*W_K*(np.pi/2)*rho_g*V_g/3
            Coef_Vm1 = F_st-F_0-F_n-F_t
            Coef_Vm0 = W_K*P_a*V_g*d
            Coefs_Vm = [Coef_Vm3, Coef_Vm2, Coef_Vm1, Coef_Vm0]

            V_m = np.roots(Coefs_Vm)
            Vm[N] = min([n for n in V_m  if n>0])

            S_m[N] = V*d/Vm[N]

            Pe = Pe_dash*R_K
            q_front1 = np.zeros(K)

            for i in range(1,K):
                    q_front1[i-1] = (sp.special.iv(i,Pe)/sp.special.kv(i,Pe))*(sp.special.kv(i-1,Pe)-2*sp.special.kv(i,Pe)+sp.special.kv(i+1,Pe))#Heat flux for moving cylinder at front

            q_front2 = L*(T_m-T_0)*Pe_dash*np.exp(-Pe)*((sp.special.iv(0,Pe)/sp.special.kv(0,Pe))*(-sp.special.kv(0,Pe)+sp.special.kv(1,Pe))+np.sum(q_front1))

            T_L[N] = T_m + (S_m[N]/L)*q_front2

            #S_FeO = np.sqrt(2*D_eff*d/Vm)

            #Ne_FeO = Vm*W_K*S_FeO*rho_St/Ar_Fe
            #I_0_b = 2*P_L/(w_b**2*np.pi)
            P_ab_1 = alpha_max*I_0*(w/r_f0)**2*np.pi*(-(w**2/4)*np.exp(-2*X_t**2/w**2)+w**2/4)
            P_ab_2 = abs(alpha_max*I_0*(w_b/r_f0)**2*np.pi*((w_b**2/4)*(-1+np.exp(-2*X_b**2/w_b**2))))
    ##        P_r = Ne_FeO*H_FeO
            P_r = 0
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
            P_mcs = 2*R_K*P_1

            P_m = rho_St*V*d*W_K*H_m
            P_La = rho_St*V*d*W_K*C_p*0.5*(T_L[N]-T_m)

            P_LHS = P_ab_1+P_ab_2+P_r
            P_RHS = P_mcs + P_m + P_La

            P_Error = P_LHS/P_RHS

        angle[N] = theta
        We[N] = rho_St*Vm[N]**2*S_m[N]/delta

        elapsed = time.time() - t
    
        Pg_str = round(P_g[N]/1000000,2)
        t_str = round(elapsed,2)
    
        print('Global Energy Balance Error = '+str(round(100*abs(P_Error-1),2))+'%')
        print('Iteration for V = ' +str(Pg_str)+' MPa took ' +str(t_str)+ 's')
        print('-----------------------------------')
    
plot1 = plt.figure(1)
plt.plot(P_g/1000000,We)
plt.xlabel('Gas Pressure (MPa)')
plt.ylabel('Weber Number')
plt.grid()


plot3 = plt.figure(3)
plt.plot(P_g/1000000,S_m*1000000)
plt.xlabel('Gas Pressure (MPa)')
plt.ylabel('Melt Film Thickness (um)')
plt.grid()

plot4 = plt.figure(4)
plt.plot(P_g/1000000,T_L)
plt.xlabel('Gas Pressure (MPa)')
plt.ylabel('Surface Temperature (K)')
plt.grid()

plot5 = plt.figure(5)
plt.plot(P_g/1000000,angle*57.3)
plt.xlabel('Gas Pressure (MPa)')
plt.ylabel('Kerf Angle (deg)')
plt.grid()

plot6 = plt.figure(6)
plt.plot(P_g/1000000,Vm)
plt.xlabel('Gas Pressure (MPa)')
plt.ylabel('Melt Velocity (m/s)')
plt.grid()

plt.show()
