import scipy as sp
import scipy.special as scsp
import scipy.optimize as scopt
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import math
import time
import os

#This script is an analytical model of CW Nd YAG laser welding of 1 mm stainless steel
#Based off of "An Analytical Thermodynamic Model of Laser Welding" by John Powell with some minor changes

#Processing Parameters

P_L = 550#Nominal output power (W)
r_ft = 1.5e-3 #Spot radius at top of material (m)
v = 5e-3 #welding velocity (m/s)

T_a = 300 #Ambient Temperature (K)
th_0 = np.deg2rad(45) #vertical angle of keyhole wall at top of weld (rad)
phi = np.pi/2 #azimuthal angle from direction of travel (rad)

t = 1e-3 #material thickness (m)
dz = 0.01e-3 #mesh step size (m)
z = 1.1*t #mesh total size (m)

N_z_steps = int(z/dz - 1) #number of nodes
z_arr = np.linspace(0, z, N_z_steps) #create mesh

#Laser Parameters

F = 150/8   #Focal ratio (ratio of focal distance to collimated beam diameter)
M = 1.07   #Output beam quality
wavelength = 1.064E-6 #Central Emission Wavelength (m)

r_f0 = 2*wavelength*F*M/np.pi #focal radius (m)
z_r = 2*r_f0*F #Rayleigh Length (m)
I_0 = 2*P_L/(np.pi*r_f0**2) #Beam laser intensity at focal radius (W/m^2)
c = 299792458 #speed of laser wave (m/s)
angular_freq = 2*np.pi*c/wavelength #angular frequency of laser wave (rad/s)

#Material Parameters

epsilon_0 = 8.854e-12 #permittivity of free space (see "The Laser Welding of Thin Metal Sheets: An Integrated Keyhole and Weld Pool Model with Supporting Experiments" Pg 1622)
sigma_st = 1.45e6 # electrical conductance per unit depth (/Ohm/m)
epsilon = np.sqrt(2 / (1 + np.sqrt((1 + sigma_st / (angular_freq * epsilon_0)) ** 2)))  # material constant for determining Fresnel absorption
R_0 = 4*np.sqrt((angular_freq/2*np.pi)*np.pi*epsilon_0/sigma_st) #reflection at normal incidence for steel (https://eng.libretexts.org/Bookshelves/Materials_Science/Supplemental_Modules_(Materials_Science)/Optical_Properties/Metallic_Reflection)

T_v = 3143 #vaporisation temperature of iron (K)
T_m = 1510 #melting temperature of steel (K)
rho = 7722 #density of steel (kg/m^3)
c_p = 460#specific heat capacity of steel (J/kgK)
lambda_th = 13.2 #thermal conductivity of 310 stainless steel(https://www.engineeringtoolbox.com/stainless-steel-310-properties-d_2167.html)

#rho = 7400 #density of steel at 1000 deg C (https://www.engineeringtoolbox.com/stainless-steel-310-properties-d_2167.html)
#c_p = 660 #specific heat capacity of steel at 1000 deg C(https://www.engineeringtoolbox.com/stainless-steel-310-properties-d_2167.html)

#Numerical Parameters

N_rk_err = 1000 #number of iterations for r_k secant method
N_th_err = 1000 #number of iterations for th secant method
N_rm_err = 1000 #number of iterations for r_m secant method

#Print important parameters to console
print("-----------------------------------------------------------------------------------------")
print(" ______               ___        __                      _      __    __   ___          ")
print("/_  __/__ ____  ___  / (_)__    / /  ___ ____ ___ ____  | | /| / /__ / /__/ (_)__  ___ _ ")
print(" / / / _ `/ _ \/ _ \/ / / _ \  / /__/ _ `(_-</ -_) __/  | |/ |/ / -_) / _  / / _ \/ _ `/ ")
print("/_/  \_,_/_//_/_//_/_/_/_//_/ /____/\_,_/___/\__/_/     |__/|__/\__/_/\_,_/_/_//_/\_, /  ")
print("                                                                                 /___/  ")

print("Laser Power = " +str(np.round(P_L,2))+ " W")
print("Welding Speed = " +str(np.round(v*1000,2))+ "mm/s")
print("Spot Diameter = " +str(np.round(r_ft*2000, 2))+ "mm")
print("-----------------------------------------------------------------------------------------")

def fresnel_absorption(th): #Fresnel absorption calculation (see Handbook of Laser Welding Technologies Pg 113 Eqn 5.21)
    alpha_fr = 1 - 0.5*(((1+(1 - epsilon*np.cos(th))**2)/(1+(1 + epsilon*np.cos(th))**2)) + ((epsilon**2 - 2 * epsilon*np.cos(th) + 2 * (np.cos(th))**2)/(epsilon**2 + 2 * epsilon*np.cos(th) + 2 * (np.cos(th))**2)))
    return alpha_fr

def laser_intensity(r_f, r): #Laser intensity (W/m^2)
    I = I_0*(r_f0/r_f)**2 * np.exp(-(2*r**2)/(r_f**2)) #Gaussian distribution
    #I = P_L/(np.pi*r_f**2) #Top hat distribution
    return I

def heat_flow(r, T, v, lambda_th, phi):  #conductive radial heat flow equation for moving line source (W)
    kappa = lambda_th/(rho*c_p) #thermal diffusivity (m^2/s)
    Pe = r*v/(2*kappa) #Peclet number
    q = (1/r)*(T - T_a)*lambda_th*Pe*(np.cos(phi) + sp.special.kv(1, Pe)/sp.special.kv(0, Pe))
    return q

def rk_error_fcn(r, th, r_f, T, v, lambda_th, phi, a_mr): #error function for calculating r_k
    I_amr = (1 - fresnel_absorption(th))*a_mr*laser_intensity(r_f, r)
    rk_err = heat_flow(r, T, v, lambda_th, phi) - I_amr - laser_intensity(r_f, r)*np.tan(th)*fresnel_absorption(th)
    return rk_err

def rk_init(r_f, th, T, v, lambda_th, phi, a_mr): #derives estimate for initialising rk error secant method
    R = np.linspace(0, r_f, 1000)
    rk_err = np.zeros(1000)

    for J in range(0, 999):
        rk_err[J] = rk_error_fcn(R[J], th, r_f, T, v, lambda_th, phi, a_mr)

    rk_min = R[np.where(rk_err == np.nanmin(rk_err))]

    return rk_min

# R_F = np.linspace(0.35*r_ft, r_ft, 1000)
# rk_err = np.zeros(1000)
#
# for J in range(0, 999):
#     rk_err[J] = rk_error_fcn(R_F[J], th_0, r_ft, T_v, v, lambda_th, phi, 0)
def secant_method_rk(th, r_f, T, v, lambda_th, N_rk_err, phi, a_mr): #secant method for solving rk error fcn == 0

    #See "https://personal.math.ubc.ca/~pwalls/math-python/roots-optimization/secant/" for full details

    r_0 = rk_init(r_f, th, T, v, lambda_th, phi, a_mr)
    r_1 = r_f # initial upper estimate for rk - the evaporative zone cannot be outside of the focal radius

    for j in range(1, N_rk_err + 1):
        m_j = r_0 - rk_error_fcn(r_0, th, r_f, T, v, lambda_th, phi, a_mr) * (r_1 - r_0) / (
                    rk_error_fcn(r_1, th, r_f, T, v, lambda_th, phi, a_mr) - rk_error_fcn(r_0, th, r_f, T, v, lambda_th, phi, a_mr))
        f_m_j = rk_error_fcn(m_j, th, r_f, T, v, lambda_th, phi, a_mr)

        if rk_error_fcn(r_0, th, r_f, T, v, lambda_th, phi, a_mr) * f_m_j < 0:
            r_0 = r_0
            r_1 = m_j

        elif rk_error_fcn(r_1, th, r_f, T, v, lambda_th, phi, a_mr) * f_m_j < 0:
            r_0 = m_j
            r_1 = r_1

        elif f_m_j == 0:
            # print("Found Exact Solution")
            return m_j

        else:
            print("Could Not Solve For Keyhole Radius")
            quit()
            return None

    return r_0 - rk_error_fcn(r_0, th, r_f, T, v, lambda_th, phi, a_mr) * (r_1 - r_0) / (
                rk_error_fcn(r_1, th, r_f, T, v, lambda_th, phi, a_mr) - rk_error_fcn(r_0, th, r_f, T, v, lambda_th, phi, a_mr))

def fresnel_absorption_mr(n_mr): #fresnel absorption coefficient for number of multiple reflections
    alpha_mr = 1 - (R_0)**(n_mr-1)
    return alpha_mr

def th_err_fcn(r, th, r_f, T, lambda_th, v, a_mr, phi): #error function for solving wall angle
    I_amr = (1 - fresnel_absorption(th))*a_mr*laser_intensity(r_f, r)
    th_err = th - np.arctan((heat_flow(r, T, v, lambda_th, phi) - I_amr)/(fresnel_absorption(th)*laser_intensity(r_f, r)))
    return th_err

def secant_method_th(r, r_f, T, lambda_th, N_th_err, v, a_mr, phi): #secant method for solving theta error fcn == 0

    #See "https://personal.math.ubc.ca/~pwalls/math-python/roots-optimization/secant/" for full details

    th_0 =  0 # initial lower estimate for th
    th_1 = np.pi  # initial upper estimate for th

    for j in range(1, N_th_err + 1):
        m_j = th_0 - th_err_fcn(r, th_0, r_f, T, lambda_th, v, a_mr, phi) * (th_1 - th_0) / (
                    th_err_fcn(r, th_1, r_f, T, lambda_th, v, a_mr, phi) - th_err_fcn(r, th_0, r_f, T, lambda_th, v, a_mr, phi))
        f_m_j = th_err_fcn(r, m_j, r_f, T, lambda_th, v, a_mr, phi)

        if th_err_fcn(r, th_0, r_f, T, lambda_th, v, a_mr, phi) * f_m_j < 0:
            th_0 = th_0
            th_1 = m_j

        elif th_err_fcn(r, th_1, r_f, T, lambda_th, v, a_mr, phi) * f_m_j < 0:
            th_0 = m_j
            th_1 = th_1

        elif f_m_j == 0:
            # print("Found Exact Solution")
            return m_j

        else:
            print("Could Not Solve For Wall Angle")
            # quit()
            return None

    return th_0 - th_err_fcn(r, th_0, r_f, T, lambda_th, v, a_mr, phi) * (th_1 - th_0) / (
                th_err_fcn(r, th_1, r_f, T, lambda_th, v, a_mr, phi) - th_err_fcn(r, th_0, r_f, T, lambda_th, v, a_mr, phi))

def line_source_strength(th, r_f, r_k, a_mr, v, lambda_th, phi): #determines line source strength at vaporisation front

    kappa = lambda_th/(rho*c_p) #thermal diffusivity (m^2/s)
    Pe = r_k*v/(2*kappa) #Peclet number
    I_amr = (1 - fresnel_absorption(th))*a_mr*laser_intensity(r_f, r_k) #fresnel absorption during multiple reflections
    q_v = laser_intensity(r_f, r_k)*fresnel_absorption(th)*np.tan(th) + I_amr #heat dissipated due at vaporisation temperature
    LSS = (4*q_v*kappa*np.pi/v)*np.exp(-Pe*np.cos(phi))/(sp.special.kv(0, Pe)*np.cos(phi) + sp.special.kv(1, Pe)) #line source strength

    return LSS

def rm_err_fcn(r, th, r_f, r_k, a_mr, v, lambda_th, phi, T_m):

    kappa = lambda_th/(rho*c_p) #thermal diffusivity (m^2/s)
    Pe = r*v/(2*kappa) #Peclet number
    rm_err = line_source_strength(th, r_f, r_k, a_mr, v, lambda_th, phi) - (T_m - T_a)*2*np.pi*lambda_th*np.exp(Pe*np.cos(phi))/sp.special.kv(0, Pe)

    return rm_err

def secant_method_rm(th, r_f, r_k, T, v, lambda_th, N_rm_err, phi, a_mr): #secant method for solving rm error fcn == 0

    #See "https://personal.math.ubc.ca/~pwalls/math-python/roots-optimization/secant/" for full details

    T_m = T

    r_0 = r_k
    r_1 = r_k*3# initial upper estimate for rk - the evaporative zone cannot be outside of the focal radius

    for j in range(1, N_rm_err + 1):
        m_j = r_0 - rm_err_fcn(r_0, th, r_f, r_k, a_mr, v, lambda_th, phi, T_m) * (r_1 - r_0) / (
                    rm_err_fcn(r_1, th, r_f, r_k, a_mr, v, lambda_th, phi, T_m) - rm_err_fcn(r_0, th, r_f, r_k, a_mr, v, lambda_th, phi, T_m))
        f_m_j = rm_err_fcn(m_j, th, r_f, r_k, a_mr, v, lambda_th, phi, T_m)

        if rm_err_fcn(r_0, th, r_f, r_k, a_mr, v, lambda_th, phi, T_m) * f_m_j < 0:
            r_0 = r_0
            r_1 = m_j

        elif rm_err_fcn(r_1, th, r_f, r_k, a_mr, v, lambda_th, phi, T_m) * f_m_j < 0:
            r_0 = m_j
            r_1 = r_1

        elif f_m_j == 0:
            # print("Found Exact Solution")
            return m_j

        else:
            print("Could Not Solve For Melt Radius")
            # quit()
            return None

    return r_0 - rm_err_fcn(r_0, th, r_f, r_k, a_mr, v, lambda_th, phi, T_m) * (r_1 - r_0) / (
                rm_err_fcn(r_1, th, r_f, r_k, a_mr, v, lambda_th, phi, T_m) - rm_err_fcn(r_0, th, r_f, r_k, a_mr, v, lambda_th, phi, T_m))

def keyhole_profile(th, r_ft, T_v, T_m, lambda_th, N_rk_err, N_th_err, N_rm_err, v, dz, z, a_mr, phi): #generates keyhole profile

    #Inputs:

    #th - wall angle at first z layer (assumed to be 45 deg) (rad)
    #r_ft - laser focal spot radius at top of material  (m)
    #T_v - vaporisation temperature (K)
    #lambda_th - thermal conductivity (W/mK)
    #N_rk_err - number of iterations to run secant method when solving for rk
    #N_th_err - number of iterations to run secant method when solving for th
    #t - laser on time (s)
    #dz - step size in vertical layer (m)
    #z - total length of vertical mesh (m)
    #a_mr - absorption coefficient for multiple reflections

    #Outputs:

    #rad - keyhole radius (m)
    #theta - keyhole wall angle (rad)
    #r_f - laser focal radius (m)
    #r_m - melt radius (m)

    N_z_steps = int(z/dz -1)
    z_arr = np.linspace(0, z, N_z_steps)
    rad = np.zeros(N_z_steps)
    theta = np.zeros(N_z_steps)
    melt_rad = np.zeros(N_z_steps)
    K = np.zeros(N_z_steps)
    LSS = np.zeros(N_z_steps)

    r_f = np.zeros(N_z_steps)
    r_f[0] = r_ft
    # K[0] = 2.5 #model thermocapillary flow at top
    K[0] = 1 #turn off thermocapillary flow

    z_0 = z_r*np.sqrt((r_ft/r_f0)**2 - 1) #laser height offset (m)
    rad[0] = secant_method_rk(th, r_f[0], T_v, v, lambda_th*K[0], N_rk_err, phi, a_mr) #calculate keyhole entry radius

    melt_rad[0] = secant_method_rm(th, r_f[0], rad[0], T_m, v, lambda_th*K[0], N_rm_err, phi, a_mr) #calculate melt radius at top of weld
    LSS[0] = line_source_strength(th, r_f[0], rad[0], a_mr, v, lambda_th * K[0], phi)

    if np.isnan(rad[0]) == True:
        return np.zeros(N_z_steps), np.zeros(N_z_steps), np.zeros(N_z_steps), np.zeros(N_z_steps), np.zeros(N_z_steps)

    else:
        theta[0] = th

        for i in range(1, N_z_steps-1):
            # K[i] = 1/(13.3e3*z_arr[i] + 0.667) + 1 #model thermocapillary flow at top
            K[i] = 1 #turn off thermocapillary flow
            rad[i] = rad[i-1] - dz*np.tan(theta[i-1]) #determine keyhole radius at next step
            r_f[i] = r_f0*np.sqrt(1 + ((z_arr[i] - z_0)/z_r)**2)# determine focal radius of laser at step
            theta[i] = secant_method_th(rad[i], r_f[i], T_v, lambda_th*K[i], N_th_err, v, a_mr, phi) #determine angle of keyhole wall
            melt_rad[i] = secant_method_rm(theta[i], r_f[i], rad[i], T_m, v, lambda_th*K[i], N_rm_err, phi, a_mr) #determine melt radius
            LSS[i] = line_source_strength(theta[i], r_f[i], rad[i], a_mr, v, lambda_th*K[i], phi)

            print("" + str(np.round(i * 100 / N_z_steps, 1)) + " %")  # print progress

            if rad[i]<=0:
                rad[i:] = np.nan
                theta[i:] = np.nan
                r_f[i:] = np.nan
                melt_rad[i:] = np.nan
                return rad, theta, r_f, melt_rad, LSS

        return rad, theta, r_f, melt_rad, LSS

tic = time.time() #start timer

print("First Run:")
rad, theta, r_f, melt_rad, LSS1 = keyhole_profile(th_0, r_ft, T_v, T_m, lambda_th, N_rk_err, N_th_err, N_rm_err, v, dz, z, 0, phi) #solve without multiple reflections

n_mr = np.pi/(4*np.nanmean(theta)) #mean number of reflections of incident beam
a_mr = fresnel_absorption_mr(n_mr) #absoprtion coefficient for multiple reflections

print("-----------------------------------------------------------------------------------------")

print("Second Run")
rad2, theta2, r_f2, melt_rad2, LSS2 = keyhole_profile(th_0, r_ft, T_v, T_m, lambda_th, N_rk_err, N_th_err, N_rm_err, v, dz, z, a_mr, phi) #recalculate with multiple reflections
d_p = z_arr[np.where(rad2 == np.nanmin(rad2))]*1000#calculate penetration depth

toc = time.time() - tic #end timer

plt.figure(1)
plt.plot(z_arr*1e3, rad2*1e3, label="Evap Zone")
plt.plot(z_arr*1e3, melt_rad2*1e3, label="Melt Zone")
plt.plot(z_arr*1e3, r_f2*1e3, label="Laser")
plt.xlabel("Z Axis (mm)")
plt.ylabel("R Axis (mm)")
plt.legend()
plt.grid()
plt.title("P = " +str(P_L)+ " W, v = " +str(np.round(v*1000, 0))+ " mm/s, d = " +str(np.round(d_p[0],2))+ " mm")

print("-----------------------------------------------------------------------------------------")
print("Penetration Depth = " +str(np.round(d_p[0],2))+ " mm")
print("Top Melt Width = " +str(np.round(melt_rad2[0]*2e3,3))+ " mm")
print("Top Keyhole Width = " +str(np.round(rad2[0]*2e3,3))+ " mm")
print("Bottom Melt Width = " +str(np.round(np.interp(0.8e-3, z_arr, melt_rad2)*2e3,2))+ " mm")
print("Simulation Took " +str(np.round(toc,2))+ "s")