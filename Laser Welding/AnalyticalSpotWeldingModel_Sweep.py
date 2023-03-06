import scipy as sp
import scipy.special as scsp
import scipy.optimize as scopt
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import math
import time

#This script is an analytical model of CW Nd YAG laser welding of 1 mm stainless steel
#Based off of "An Analytical Thermodynamic Model of Laser Welding" by John Powell with some minor changes

#Processing Parameters

T_a = 300 #Ambient Temperature (K)
th_0 = np.deg2rad(45) #angle of keyhole wall at top of weld (rad)

#Laser Parameters

F =150/8   #Focal ratio (ratio of focal distance to collimated beam diameter)
M = 1.07   #Output beam quality
wavelength = 1.064E-6 #Central Emission Wavelength (m)
P_L = 1000 #Nominal output power (W)

r_f0 = 2*wavelength*F*M/np.pi #focal radius (m)
z_r = 2*r_f0*F #Rayleigh Length (m)
r_ft = 6e-4 #Spot radius at top of material (m)
I_0 = 2*P_L/(np.pi*r_f0**2) #Beam laser intensity at focal radius (W/m^2)
angular_freq = 2*np.pi*299792458/wavelength #angular frequency of laser wave (rad/s)

#Material Parameters

epsilon_0 = 8.854e-12 #permittivity of free space (see "The Laser Welding of Thin Metal Sheets: An Integrated Keyhole and Weld Pool Model with Supporting Experiments" Pg 1622)
sigma_st = 1.45e6 # electrical conductance per unit depth (/Ohm/m)
epsilon = np.sqrt(2 / (1 + np.sqrt((1 + sigma_st / (angular_freq * epsilon_0)) ** 2)))  # material constant for determining Fresnel absorption
R_0 = 4*np.sqrt((angular_freq/2*np.pi)*np.pi*epsilon_0/sigma_st) #reflection at normal incidence for steel (https://eng.libretexts.org/Bookshelves/Materials_Science/Supplemental_Modules_(Materials_Science)/Optical_Properties/Metallic_Reflection)

T_v = 3143 #vaporisation temperature of iron (K)
T_m = 1673 #melting temperature of steel (K)
rho = 7750 #density of steel (kg/m^3)
c_p = 420#specific heat capacity of steel (J/kgK)
lambda_th = 31 #thermal conductivity of high chromium (10%) steel (W/mK)

K_lambda = 2.5

#Numerical Parameters

N_rk_err = 1000 #number of iterations for r_k secant method
N_th_err = 1000 #number of iterations for th secant method
N_rm_err = 1000 #number of iterations for r_m secant method
tic = time.time() #start timer

def fresnel_absorption(th): #Fresnel absorption calculation (see Handbook of Laser Welding Technologies Pg 113 Eqn 5.21)
    alpha_fr = 1 - 0.5*(((1+(1 - epsilon*np.cos(th))**2)/(1+(1 + epsilon*np.cos(th))**2)) + ((epsilon**2 - 2 * epsilon*np.cos(th) + 2 * (np.cos(th))**2)/(epsilon**2 + 2 * epsilon*np.cos(th) + 2 * (np.cos(th))**2)))
    return alpha_fr

def laser_intensity(r_f, r): #Laser intensity assuming Gaussian distribution (W/m^2)
    I = I_0*(r_f0/r_f)**2 * np.exp(-(2*r**2)/(r_f**2))
    return I

def heat_flow(r, T, lambda_th, t):  #conductive radial heat flow equation for moving line source (W)
    kappa = lambda_th/(rho*c_p) #thermal diffusivity (m^2/s)
    Fo = r**2/(4*kappa*t) #Fourier number
    q = (2*lambda_th/r)*(T - T_a)*np.exp(-Fo)*(-1/scsp.expi(-Fo))
    return q

def rk_error_fcn(r, th, r_f, T, lambda_th, t, a_mr): #error function for calculating r_k
    rk_err = heat_flow(r, T, lambda_th, t) - laser_intensity(r_f, r)*(a_mr - a_mr*fresnel_absorption(th) + fresnel_absorption(th)*np.tan(th))
    return rk_err

# def rk_error_fcn_brent(r): #error function for calculating r_k w/ scipy.optimize methods
#     th = np.pi/4
#     r_f = 2e-4
#     T = 3143
#     lambda_th = 31
#     t = 2e-4
#     rk_err = heat_flow(r, T, lambda_th, t) - laser_intensity(r_f, r)*fresnel_absorption(th)*np.tan(th)
#     return rk_err

def secant_method_rk(th, r_f, T, lambda_th, N_rk_err, t, a_mr): #secant method for solving rk error fcn == 0

    #See "https://personal.math.ubc.ca/~pwalls/math-python/roots-optimization/secant/" for full details

    r_0 =  r_f*0.1 # initial lower estimate for rk
    r_1 = r_f  # initial upper estimate for rk - the evaporative zone cannot be outside of the focal radius

    # if rk_error_fcn(r_0, th, r_f, T, lambda_th, t, a_mr) * rk_error_fcn(r_1, th, r_f, T, lambda_th, t, a_mr) >= 0:
    #     print("Could Not Solve For Keyhole Radius")
    #     # quit()
    #     return None

    for j in range(1, N_rk_err + 1):
        m_j = r_0 - rk_error_fcn(r_0, th, r_f, T, lambda_th, t, a_mr) * (r_1 - r_0) / (
                    rk_error_fcn(r_1, th, r_f, T, lambda_th, t, a_mr) - rk_error_fcn(r_0, th, r_f, T, lambda_th, t, a_mr))
        f_m_j = rk_error_fcn(m_j, th, r_f, T, lambda_th, t, a_mr)

        if rk_error_fcn(r_0, th, r_f, T, lambda_th, t, a_mr) * f_m_j < 0:
            r_0 = r_0
            r_1 = m_j

        elif rk_error_fcn(r_1, th, r_f, T, lambda_th, t, a_mr) * f_m_j < 0:
            r_0 = m_j
            r_1 = r_1

        elif f_m_j == 0:
            # print("Found Exact Solution")
            return m_j

        else:
            # print("Could Not Solve For Keyhole Radius")
            # quit()
            return None

    return r_0 - rk_error_fcn(r_0, th, r_f, T, lambda_th, t, a_mr) * (r_1 - r_0) / (
                rk_error_fcn(r_1, th, r_f, T, lambda_th, t, a_mr) - rk_error_fcn(r_0, th, r_f, T, lambda_th, t, a_mr))

def fresnel_absorption_mr(n_mr): #fresnel absorption coefficient for number of multiple reflections
    alpha_mr = 1 - (R_0)**(n_mr-1)
    return alpha_mr

def th_err_fcn(r, th, r_f, T, lambda_th, t, a_mr): #error function for solving wall angle
    I_amr = (1 - fresnel_absorption(th))*a_mr*laser_intensity(r_f, r)
    th_err = th - np.arctan((heat_flow(r, T, lambda_th, t) - I_amr)/(fresnel_absorption(th)*laser_intensity(r_f, r)))
    return th_err

def secant_method_th(r, r_f, T, lambda_th, N_th_err, t, a_mr): #secant method for solving theta error fcn == 0

    #See "https://personal.math.ubc.ca/~pwalls/math-python/roots-optimization/secant/" for full details

    th_0 =  0 # initial lower estimate for th
    th_1 = np.pi/4  # initial upper estimate for th

    for j in range(1, N_th_err + 1):
        m_j = th_0 - th_err_fcn(r, th_0, r_f, T, lambda_th, t, a_mr) * (th_1 - th_0) / (
                    th_err_fcn(r, th_1, r_f, T, lambda_th, t, a_mr) - th_err_fcn(r, th_0, r_f, T, lambda_th, t, a_mr))
        f_m_j = th_err_fcn(r, m_j, r_f, T, lambda_th, t, a_mr)

        if th_err_fcn(r, th_0, r_f, T, lambda_th, t, a_mr) * f_m_j < 0:
            th_0 = th_0
            th_1 = m_j

        elif th_err_fcn(r, th_1, r_f, T, lambda_th, t, a_mr) * f_m_j < 0:
            th_0 = m_j
            th_1 = th_1

        elif f_m_j == 0:
            # print("Found Exact Solution")
            return m_j

        else:
            # print("Could Not Solve For Wall Angle")
            # quit()
            return None

    return th_0 - th_err_fcn(r, th_0, r_f, T, lambda_th, t, a_mr) * (th_1 - th_0) / (
                th_err_fcn(r, th_1, r_f, T, lambda_th, t, a_mr) - th_err_fcn(r, th_0, r_f, T, lambda_th, t, a_mr))

def line_source_strength(T, lambda_th, r, t): #line source strength for stationary continuous line source
    kappa = lambda_th/(rho*c_p) #thermal diffusivity (m^2/s)
    LSS = (T - T_a)*4*np.pi*kappa*(-1/scsp.expi((-r**2)/(4*kappa*t))) #line source strength
    return LSS

def rm_err_fcn(LSS, T, lambda_th, r, t): #error function for determining melt radius
    kappa = lambda_th/(rho*c_p) #thermal diffusivity (m^2/s)
    rm_err = LSS - (T - T_a)*4*np.pi*kappa*(-1/scsp.expi((-r**2)/(4*kappa*t))) #error function
    return rm_err


def secant_method_rm(r_k, T_v, T_m, lambda_th, N_rm_err, t):  # secant method for solving rk error fcn == 0

    # See "https://personal.math.ubc.ca/~pwalls/math-python/roots-optimization/secant/" for full details

    LSS = line_source_strength(T_v, lambda_th, r_k, t) #determine line source strength (W/m)

    r_0 = r_k*0.1  # initial lower estimate for rk
    r_1 = r_k * 10  # initial upper estimate for rk - the evaporative zone cannot be outside of the focal radius

    for j in range(1, N_rm_err + 1):
        m_j = r_0 - rm_err_fcn(LSS, T_m, lambda_th, r_0, t) * (r_1 - r_0) / (
                rm_err_fcn(LSS, T_m, lambda_th, r_1, t) - rm_err_fcn(LSS, T_m, lambda_th, r_0, t))
        f_m_j = rm_err_fcn(LSS, T_m, lambda_th, m_j, t)

        if rm_err_fcn(LSS, T_m, lambda_th, r_0, t) * f_m_j < 0:
            r_0 = r_0
            r_1 = m_j

        elif rm_err_fcn(LSS, T_m, lambda_th, r_1, t) * f_m_j < 0:
            r_0 = m_j
            r_1 = r_1

        elif f_m_j == 0:
            # print("Found Exact Solution")
            return m_j

        else:
            # print("Could Not Solve For Melt Radius")
            # quit()
            return None

    return r_0 - rm_err_fcn(LSS, T_m, lambda_th, r_0, t) * (r_1 - r_0) / (
            rm_err_fcn(LSS, T_m, lambda_th, r_1, t) - rm_err_fcn(LSS, T_m, lambda_th, r_0, t))

def keyhole_profile(th, r_ft, T_v, T_m, lambda_th, N_rk_err, N_th_err, N_rm_err, t, dz, z, a_mr): #generates keyhole profile

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

    N_z_steps = int(z/dz -1)
    z_arr = np.linspace(0, z, N_z_steps)
    rad = np.zeros(N_z_steps)
    theta = np.zeros(N_z_steps)
    melt_rad = np.zeros(N_z_steps)

    r_f = np.zeros(N_z_steps)
    r_f[0] = r_ft

    z_0 = z_r*np.sqrt((r_ft/r_f0)**2 - 1)
    rad[0] = secant_method_rk(th, r_f[0], T_v, lambda_th, N_rk_err, t, a_mr)
    melt_rad[0] = secant_method_rm(rad[0], T_v, T_m, lambda_th, N_rm_err, t)

    if rad[0] == np.NaN:
        return Nan
    else:
        theta[0] = th

        for i in range(1, N_z_steps-1):

            rad[i] = rad[i-1] - dz*np.tan(theta[i-1])
            r_f[i] = r_f0*np.sqrt(1 + ((z_arr[i] - z_0)/z_r)**2)
            theta[i] = secant_method_th(rad[i], r_f[i], T_v, lambda_th, N_th_err, t, a_mr)
            melt_rad[i] = secant_method_rm(rad[i], T_v, T_m, lambda_th, N_rm_err, t)

            if rad[i]<=0:
                rad[i:] = np.nan
                theta[i:] = np.nan
                r_f[i:] = np.nan
                melt_rad[i:] = np.nan
                return rad, theta, r_f, melt_rad

        return rad, theta, r_f, melt_rad

t_on_i = 0.1
t_on_f = 2
N_t_on = 10

dz = 1e-6
z = 3000e-6
N_z_steps = int(z/dz - 1)
z_arr = np.linspace(0, z, N_z_steps)

t_on = np.linspace(t_on_i, t_on_f, N_t_on)
D_P = np.zeros(N_t_on)
R_M = np.zeros((N_t_on, N_z_steps))
R_K = np.zeros((N_t_on, N_z_steps))

tic = time.time() #start timer

for iter in range(0, N_t_on):

    on_time = t_on[iter]
    rad, theta, r_f, melt_rad = keyhole_profile(np.pi/4, r_ft, T_v, T_m, lambda_th, N_rk_err, N_th_err, N_rm_err, on_time, dz, z, 0) #solve without multiple reflections

    n_mr = np.pi/(4*np.nanmean(theta)) #mean number of reflections of incident beam
    a_mr = fresnel_absorption_mr(n_mr) #absoprtion coefficient for multiple reflections

    rad2, theta2, r_f2, melt_rad = keyhole_profile(np.pi/4, r_ft, T_v, T_m, lambda_th, N_rk_err, N_th_err, N_rm_err, on_time, dz, z, a_mr) #recalculate with multiple reflections

    D_P[iter] = z_arr[np.where(rad2 == np.nanmin(rad2))] #penetration depth
    R_M[iter, :] = melt_rad
    R_K[iter, :] = rad2

    print("" + str(np.round(iter * 100 / N_t_on)) + " %")

plt.figure(1)
plt.plot(t_on, D_P*1000)
plt.xlabel("Time (s)")
plt.ylabel("Penetration Depth (mm)")

print("Simulation Took " +str(np.round(toc,2))+ "s")