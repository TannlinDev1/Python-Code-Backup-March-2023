import numpy as np
import scipy.optimize as scopt
import scipy.special as scsp
import matplotlib.pyplot as plt

#The following is a basic heat transfer model for pulsed Nd:YAG laser welding of stainless steel
# Based off of "Effect of Laser Spot Weld Energy and Duration on Melting and Absorption" by PW Fuerschbach and GR Eisler

N_GLOB = 100

P_MIN = 100
P_MAX = 400
P_ARR = np.linspace(P_MIN, P_MAX, N_GLOB)

T_P_MIN = 1e-3
T_P_MAX  = 10e-3
T_P_ARR = np.linspace(T_P_MIN, T_P_MAX, N_GLOB)

D_P = np.zeros((N_GLOB, N_GLOB))

wavelength = 1.064E-6 #Central Emission Wavelength (m)
angular_freq = 2*np.pi*299792458/wavelength #angular frequency of laser wave (rad/s)

epsilon_0 = 8.854e-12 #permittivity of free space (see "The Laser Welding of Thin Metal Sheets: An Integrated Keyhole and Weld Pool Model with Supporting Experiments" Pg 1622)
sigma_st = 1.45e6 # electrical conductance per unit depth (/Ohm/m)
epsilon = np.sqrt(2 / (1 + np.sqrt((1 + sigma_st / (angular_freq * epsilon_0)) ** 2)))  # material constant for determining Fresnel absorption
R_0 = 4*np.sqrt((angular_freq/2*np.pi)*np.pi*epsilon_0/sigma_st) #reflection at normal incidence for steel (https://eng.libretexts.org/Bookshelves/Materials_Science/Supplemental_Modules_(Materials_Science)/Optical_Properties/Metallic_Reflection)

T_v = 3143 #vaporisation temperature of iron (K)
T_m = 1727 #melting temperature of steel (K)
rho = 8000 #density of steel (kg/m^3)
c_p = 500#specific heat capacity of steel (J/kgK)
k = 16.2 #thermal conductivity of high chromium (10%) steel (W/mK)

a = k/(rho*c_p) #thermal diffusivity (m^2/s)

T_0 = 300 #initial temperature (K)

print("-----------------------------------------------------------------------------------------")
print(" ______               ___        __                      _      __    __   ___          ")
print("/_  __/__ ____  ___  / (_)__    / /  ___ ____ ___ ____  | | /| / /__ / /__/ (_)__  ___ _ ")
print(" / / / _ `/ _ \/ _ \/ / / _ \  / /__/ _ `(_-</ -_) __/  | |/ |/ / -_) / _  / / _ \/ _ `/ ")
print("/_/  \_,_/_//_/_//_/_/_/_//_/ /____/\_,_/___/\__/_/     |__/|__/\__/_/\_,_/_/_//_/\_, /  ")
print("                                                                                 /___/  ")
print("-----------------------------------------------------------------------------------------")
print("Conduction Welding Model - Parameter Sweep")

for glob_I in range(0, N_GLOB):
    P = P_ARR[glob_I]

    print("-----------------------------------------------------------------------------------------")
    print("Step " +str(glob_I+1)+ " Out of " +str(N_GLOB))
    print("-----------------------------------------------------------------------------------------")

    for glob_J in range(0, N_GLOB):

        print("" +str(np.round(glob_J*100/N_GLOB, 2))+ " %")

        t_p = T_P_ARR[glob_J]

        q = P*R_0 #absorbed power (W)

        def temp_function(r, t):
            temp_fcn_r = T_m + (q/(2*np.pi*k*r))*(scsp.erf(r/np.sqrt(4*a*t)) - scsp.erf(r/np.sqrt(4*a*(t - t_p))))
            return temp_fcn_r

        def sec_method_T(t):

            r_1 = 1e-6
            r_2 = 1e-2

            N = 1000

            '''Approximate solution of f(x)=0 on interval [a,b] by the secant method.
        
            Parameters
            ----------
            f : function
                The function for which we are trying to approximate a solution f(x)=0.
                a,b : numbers
                    The interval in which to search for a solution. The function returns
                    None if f(a)*f(b) >= 0 since a solution is not guaranteed.
                N : (positive) integer
                    The number of iterations to implement.
        
                Returns
                -------
                m_N : number
                    The x intercept of the secant line on the the Nth interval
                        m_n = a_n - f(a_n)*(b_n - a_n)/(f(b_n) - f(a_n))
                    The initial interval [a_0,b_0] is given by [a,b]. If f(m_n) == 0
                    for some intercept m_n then the function returns this solution.
                    If all signs of values f(a_n), f(b_n) and f(m_n) are the same at any
                    iterations, the secant method fails and return None.
        
                Examples
                --------
                >>> f = lambda x: x**2 - x - 1
                >>> secant(f,1,2,5)
                1.6180257510729614
                '''
            # if temp_function(r_1, t) * temp_function(r_2, t) >= 0:
            #     print("Secant method fails.")
            #     return None

            for n in range(1, N + 1):
                m_n = r_1 - temp_function(r_1, t) * (r_2 - r_1) / (temp_function(r_2, t) - temp_function(r_1, t))
                f_m_n = temp_function(m_n, t)
                if temp_function(r_1, t) * f_m_n < 0:
                    r_1 = r_1
                    r_2 = m_n
                elif temp_function(r_2, t) * f_m_n < 0:
                    r_1 = m_n
                    r_2 = r_2
                elif f_m_n == 0:
                    return m_n
                else:
                    print("Secant method fails.")
                    return np.NaN
            return -1*(r_1 - temp_function(r_1, t) * (r_2 - r_1) / (temp_function(r_2, t) - temp_function(r_1, t)))

        # t_star = scopt.fminbound(radius_solver, t_min, t_max)

        t_min = t_p
        t_max = t_p*1.5

        r_tmax = sec_method_T(t_max)
        r_tmin = sec_method_T(t_min)

        if np.isnan(r_tmax) == True or np.isnan(r_tmin) == True:
            D_P[glob_I, glob_J] = None

        else:

            t_opt = scopt.fminbound(sec_method_T, t_min, t_max)
            r_max = -1*sec_method_T(t_opt)
            D_P[glob_I, glob_J] = r_max

tp_arr, p_arr = np.meshgrid(T_P_ARR, P_ARR)

fig1, ax1 = plt.subplots(constrained_layout=True)
CS1 = ax1.contourf(tp_arr*1000, p_arr, D_P*1000)
ax1.set_xlabel("Pulse Duration (ms)")
ax1.set_ylabel("Laser Power (W)")
cbar1 = fig1.colorbar(CS1, location="bottom")
cbar1.ax.set_xlabel("Penetration Depth (mm)")