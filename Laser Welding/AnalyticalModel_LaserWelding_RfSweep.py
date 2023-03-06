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
import os

#This script is an analytical model of CW Nd YAG laser welding of 1 mm stainless steel
#Based off of "An Analytical Thermodynamic Model of Laser Welding" by John Powell with some minor changes

#This simulation performs a sensitivity analysis of the model w.r.t welding speed and spot radius

#Processing Parameters

T_a = 300  # Ambient Temperature (K)

r_f_min = 0.275e-3 #minimum spot radius (m)
r_f_max = (1.65e-3)/2 #maximum spot radius (m)
N_GLOB_RF = 100 #number of spot radius iterations
r_f_arr = np.linspace(r_f_min, r_f_max, N_GLOB_RF) #focal radius array

v_min = 1e-3 #minimum weld speed (m/s)
v_max = 30e-3 #maximum weld speed (m/s)
N_GLOB_V = 100
v_arr = np.linspace(v_min, v_max, N_GLOB_V) #velocity array

D_GLOB = np.zeros((N_GLOB_V,N_GLOB_RF))
R_K_GLOB = np.zeros((N_GLOB_V,N_GLOB_RF))
W_T_GLOB = np.zeros((N_GLOB_V,N_GLOB_RF))
W_B_GLOB = np.zeros((N_GLOB_V,N_GLOB_RF))

# Laser Parameters

F = 5.56  # Focal ratio
M = 1.07  # Output beam quality
wavelength = 1.064E-6  # Central Emission Wavelength (m)
P_L = 1000  # Nominal output power (W)

r_f0 = 2 * wavelength * F * M / np.pi  # focal radius (m)
z_r = 2 * r_f0 * F  # Rayleigh Length (m)

I_0 = 2 * P_L / (np.pi * r_f0 ** 2)  # Beam laser intensity at focal radius (W/m^2)
angular_freq = 2 * np.pi * 299792458 / wavelength  # angular frequency of laser wave (rad/s)

# Material Parameters

epsilon_0 = 8.854e-12  # material constant (see "The Laser Welding of Thin Metal Sheets: An Integrated Keyhole and Weld Pool Model with Supporting Experiments" Pg 1622)
sigma_st = 1.45e6  # electrical conductance per unit depth (/Ohm/m)
epsilon = np.sqrt(2 / (1 + np.sqrt(
    (1 + sigma_st / (angular_freq * epsilon_0)) ** 2)))  # material constant for determining Fresnel absorption

T_v = 3143  # vaporisation temperature of iron (K)
T_m = 1673  # melting temperature of steel (K)
rho = 7750  # density of steel (kg/m^3)
c_p = 420  # specific heat capacity of steel (J/kgK)
lambda_th = 31  # thermal conductivity of high chromium (10%) steel (W/mK)

# Numerical Parameters

N_rk_err = 1000  # number of iterations for r_k secant method
N_rm_err = 1000  # number of iterations for r_m secant method

phi_min = 0
phi_max = 2 * np.pi
N_phi = 100
phi = np.linspace(phi_min, phi_max, N_phi)  # define azimuthal angle array (rad)
d_phi = (phi_max - phi_min) / (N_phi - 1)  # step size for azimuthal angle array (rad)

z_arr = np.zeros(N_GLOB_RF) #defocus height (m)

for i in range(0, N_GLOB_RF):
    z_arr[i] = z_r*np.sqrt((r_f_arr[i]/r_f0)**2 - 1)

#Print important parameters to console

print("-----------------------------------------------------------------------------------------")
print(" ______               ___        __                      _      __    __   ___          ")
print("/_  __/__ ____  ___  / (_)__    / /  ___ ____ ___ ____  | | /| / /__ / /__/ (_)__  ___ _ ")
print(" / / / _ `/ _ \/ _ \/ / / _ \  / /__/ _ `(_-</ -_) __/  | |/ |/ / -_) / _  / / _ \/ _ `/ ")
print("/_/  \_,_/_//_/_//_/_/_/_//_/ /____/\_,_/___/\__/_/     |__/|__/\__/_/\_,_/_/_//_/\_, /  ")
print("                                                                                 /___/  ")

print("VELOCITY AND SPOT RADIUS SWEEP")
print("-----------------------------------------------------------------------------------------")
print("Laser Power = " +str(np.round(P_L,2))+ " W")
print("-----------------------------------------------------------------------------------------")

tic = time.time()

for glob_I in range(0, N_GLOB_V): #for loop which iterates across welding speed
    print("-----------------------------------------------------------------------------------------")
    print("Step " +str(glob_I+1)+ " Out of " +str(N_GLOB_V))
    print("-----------------------------------------------------------------------------------------")

    for glob_J in range(0, N_GLOB_RF): #for loop which iterates across spot radius

        print("" +str(np.round(glob_J*100/N_GLOB_RF, 2))+ " %")
        d_star_err = 1e-5  # initial error to initiate while loop (mm)
        dstar = 0.5e-3  # initial guess for penetration depth (mm)

        v = v_arr[glob_I] #Welding speed (m/s)
        r_f = r_f_arr[glob_J] #Spot radius (m)

        th = np.deg2rad(45) #angle of keyhole wall at top of weld (rad)

        def fresnel_absorption(th): #Fresnel absorption calculation (see Handbook of Laser Welding Technologies Pg 113 Eqn 5.21)
            alpha_fr = 1 - 0.5*(((1+(1 - epsilon*np.cos(th))**2)/(1+(1 + epsilon*np.cos(th))**2)) + ((epsilon**2 - 2 * epsilon*np.cos(th) + 2 * (np.cos(th))**2)/(epsilon**2 + 2 * epsilon*np.cos(th) + 2 * (np.cos(th))**2)))
            return alpha_fr

        def laser_intensity(r_f, r): #Laser intensity assuming Gaussian distribution (W/m^2)
            I = I_0*(r_f0/r_f)**2 * np.exp(-(2*r**2)/(r_f**2))
            return I

        def heat_flow(r, T, v, lambda_th, phi):  #conductive radial heat flow equation for moving line source (W)
            kappa = lambda_th/(rho*c_p) #thermal diffusivity (m^2/s)
            Pe = r*v/(2*kappa) #Peclet number
            q = (1/r)*(T - T_a)*lambda_th*Pe*(np.cos(phi) + sp.special.kv(1, Pe)/sp.special.kv(0, Pe))
            return q

        def rk_error_fcn(r, th, r_f, T, v, lambda_th, phi): #error function for calculating r_k
            rk_err = heat_flow(r, T, v, lambda_th, phi) - laser_intensity(r_f, r)*fresnel_absorption(th)*np.tan(th)
            return rk_err

        def secant_method_rk(phi, th, r_f, T, v, lambda_th, N_rk_err): #secant method for solving rk error fcn == 0

            #See "https://personal.math.ubc.ca/~pwalls/math-python/roots-optimization/secant/" for full details

            r_0 = r_f/4 #initial lower estimate for rk
            r_1 = r_f #initial upper estimate for rk

            if rk_error_fcn(r_0, th, r_f, T, v, lambda_th, phi)*rk_error_fcn(r_1, th, r_f, T, v, lambda_th, phi) >= 0:
                # print("Could Not Solve For Melt Radius")
                return None

            for j in range(1, N_rk_err+1):
                m_j = r_0 - rk_error_fcn(r_0, th, r_f, T, v, lambda_th, phi)*(r_1-r_0)/(rk_error_fcn(r_1, th, r_f, T, v, lambda_th, phi) - rk_error_fcn(r_0, th, r_f, T, v, lambda_th, phi))
                f_m_j = rk_error_fcn(m_j, th, r_f, T, v, lambda_th, phi)

                if rk_error_fcn(r_0, th, r_f, T, v, lambda_th, phi)*f_m_j<0:
                    r_0 = r_0
                    r_1 = m_j

                elif rk_error_fcn(r_1, th, r_f, T, v, lambda_th, phi)*f_m_j<0:
                    r_0 = m_j
                    r_1 = r_1

                elif f_m_j == 0:
                    # print("Found Exact Solution")
                    return m_j

                else:
                    # print("Could Not Solve For Melt Radius")
                    return None

            return r_0 - rk_error_fcn(r_0, th, r_f, T, v, lambda_th, phi)*(r_1-r_0)/(rk_error_fcn(r_1, th, r_f, T, v, lambda_th, phi) - rk_error_fcn(r_0, th, r_f, T, v, lambda_th, phi))

        def line_source_strength(r, T, lambda_th, phi): #Power dissipated by moving line source (W)
            kappa = lambda_th/(rho*c_p) #thermal diffusivity (m^2/s)
            Pe = r*v/(2*kappa) #Peclet number
            Pdash = (T - T_a)*2*np.pi*lambda_th*np.exp(Pe*np.cos(phi))/sp.special.kv(0, Pe)
            return Pdash

        def rm_error_fcn(r_k, phi, T_v, lambda_th, T_m, r_m): #error function for calculating r_m
            rm_err = line_source_strength(r_k, T_v, lambda_th, phi) - line_source_strength(r_m, T_m, lambda_th, phi)
            return rm_err

        def secant_method_rm(r_k, phi, T_v, lambda_th, T_m, N_rm_err): #secant method for solving rm error fcn == 0

            r_0 = r_k #inital lower estimate
            r_1 = r_k*50 #initial upper estimate

            if rm_error_fcn(r_k, phi, T_v, lambda_th, T_m, r_0)*rm_error_fcn(r_k, phi, T_v, lambda_th, T_m, r_1) >= 0:
                return None

            for j in range(1, N_rm_err+1):
                m_j = r_0 - rm_error_fcn(r_k, phi, T_v, lambda_th, T_m, r_0)*(r_1-r_0)/(rm_error_fcn(r_k, phi, T_v, lambda_th, T_m, r_1) - rm_error_fcn(r_k, phi, T_v, lambda_th, T_m, r_0))
                f_m_j = rm_error_fcn(r_k, phi, T_v, lambda_th, T_m, m_j)

                if rm_error_fcn(r_k, phi, T_v, lambda_th, T_m, r_0)*f_m_j<0:
                    r_0 = r_0
                    r_1 = m_j

                elif rm_error_fcn(r_k, phi, T_v, lambda_th, T_m, r_1)*f_m_j<0:
                    r_0 = m_j
                    r_1 = r_1

                elif f_m_j == 0:
                    # print("Found Exact Solution")
                    return m_j

                else:
                    return None

            return r_0 - rm_error_fcn(r_k, phi, T_v, lambda_th, T_m, r_0)*(r_1-r_0)/(rm_error_fcn(r_k, phi, T_v, lambda_th, T_m, r_1) - rm_error_fcn(r_k, phi, T_v, lambda_th, T_m, r_0))

        def keyhole_power(rad): #determines power deposited by laser across keyhole radius (W/rad)

            #see "https://www.wolframalpha.com/input?i=integrate+r*A*%28B%2FC%29%5E2*exp%28-2*r%5E2%2FC%5E2%29+with+respect+to+r+from+0+to+D" for derivation
            # A = I_0 , B = r_f0 , C = r_f, D = rad

            if rad>r_f: #if keyhole radius is outside of focal spot then laser can only deposit at maximum I_0
                P_KH = (0.25 * I_0 * r_f0 ** 2) * (1 - np.exp(-(2 * r_f**2) / (r_f ** 2)))
            else:
                P_KH = (0.25 * I_0 * r_f0 ** 2) * (1 - np.exp(-(2 * rad**2) / (r_f ** 2))) #integral of gaussian beam w.r.t r (W/rad)
            return P_KH

        def fresnel_absorption_mr(n_mr): #fresnel absorption coefficient for number of multiple reflections
            alpha_mr = 1 - (1 - fresnel_absorption(np.pi/2))**(n_mr-1)
            return alpha_mr

        def penetration_depth(P_T, P_B, P_abs): #penetration depth of laser (m)
            P_av = (P_T + P_B)/2 #average power dissipated across keyhole depth (W)
            d = P_abs/P_av #penetration depth (m)
            return d

        def radtocart(R, angle): #converts from radial to cartesian co-ordinates
            X = R*np.cos(angle)
            Y = R*np.sin(angle)
            return X, Y

        #memory preallocation
        r_k_arr = np.zeros(N_phi)
        r_m_b_arr = np.zeros(N_phi)
        r_m_t_arr = np.zeros(N_phi)
        x_k = np.zeros(N_phi)
        y_k = np.zeros(N_phi)
        x_m_t = np.zeros(N_phi)
        y_m_t = np.zeros(N_phi)
        x_m_b = np.zeros(N_phi)
        y_m_b = np.zeros(N_phi)
        x_l = np.zeros(N_phi)
        y_l = np.zeros(N_phi)
        P_KH = np.zeros(N_phi)
        d_r = np.zeros(N_phi)
        d = np.zeros(N_phi)
        P_Fr1 = np.zeros(N_phi)
        P_Fr2 = np.zeros(N_phi)
        P_abs = np.zeros(N_phi)
        N_mr = np.zeros(N_phi)
        P_T = np.zeros(N_phi)
        P_B = np.zeros(N_phi)

        solution = True

        for J in range(0, N_phi): #Solve for keyhole dimensions across full revolution and convert to cartesian co-ordinates

            r_k_arr[J] = secant_method_rk(phi[J], th, r_f, T_v, v, lambda_th*2.5, N_rk_err) #solve for keyhole entrance radius (m)

            if math.isnan(r_k_arr[J]) == True: #checks if r_k solver has failed
                solution = False #flag that solution has failed and exit from for loop
                break

            r_m_b_arr[J] = secant_method_rm(r_k_arr[J], phi[J], T_v, lambda_th, T_m, N_rm_err)*0.5 #solve for bottom weld width (m)
            r_m_t_arr[J] = secant_method_rm(r_k_arr[J], phi[J], T_v, lambda_th*2.5, T_m, N_rm_err)#solve for top weld width (m)

            P_KH[J] = keyhole_power(r_k_arr[J])*d_phi #laser power absorbed by keyhole (W)
            P_T[J] = line_source_strength(r_m_t_arr[J], T_m, lambda_th*2.5, phi[J]) #power dissipated at top of weld (W)
            P_B[J] = line_source_strength(r_m_b_arr[J], T_m, lambda_th, phi[J]) #power dissipated at bottom of weld (W)

        if solution == False: #assign global solution variables to NaN for unsolved r_k
            D_GLOB[glob_I, glob_J] = None
            R_K_GLOB[glob_I, glob_J] = None
            W_T_GLOB[glob_I, glob_J] = None
            W_B_GLOB[glob_I, glob_J] = None

        else: #otherwise then solve for keyhole dimensions (takes average across full revolution)
            R_K = np.mean(r_k_arr)
            W_T = np.mean(r_m_t_arr)*2
            W_B = np.mean(r_m_b_arr)*2

            P_KH = np.sum(P_KH)
            P_T = np.mean(P_T)
            P_B = np.mean(P_B)

            while d_star_err>1e-6: #loop which solves for penetration depth to accuracy of 1e-6 m

                th_av = np.arctan(R_K/dstar) #average keyhole inclination in lower section of weld (rad)
                P_Fr1 = P_KH*fresnel_absorption(th_av) #power lost by first fresnel absorption (W)
                n_mr = np.pi/(4*th_av) #average number of multiple reflections
                P_Fr2 = (P_KH - P_Fr1)*fresnel_absorption_mr(n_mr) #power lost by second fresnel absorption (W)

                P_abs = P_Fr1 + P_Fr2 #total absorbed power from laser (W)

                d_star_err = abs(penetration_depth(P_T, P_B, P_abs) - dstar) #determine error from previous iteration of dstar
                dstar = penetration_depth(P_T, P_B, P_abs) #define new dstar

            D_GLOB[glob_I,glob_J] = dstar #save important keyhole dimensions to global variable
            R_K_GLOB[glob_I,glob_J] = R_K
            W_T_GLOB[glob_I,glob_J] = W_T
            W_B_GLOB[glob_I,glob_J] = W_B

toc = time.time() - tic

print("Simulation Took " +str(np.round(toc,2))+ "s")
print("-----------------------------------------------------------------------------------------")

#Save results

os.chdir(r"C:\Users\angus.mcallister\Documents\TFF\Welding Sim Results\Sim Results")
np.savetxt("Penetration Depth.txt", D_GLOB, delimiter=" ",fmt='%f')
np.savetxt("Keyhole Radius.txt", R_K_GLOB, delimiter=" ",fmt='%f')
np.savetxt("Top Melt Width.txt", W_T_GLOB, delimiter=" ",fmt='%f')
np.savetxt("Bottom Melt Width.txt", W_B_GLOB, delimiter=" ",fmt='%f')
