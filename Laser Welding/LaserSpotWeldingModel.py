import numpy as np
import scipy.optimize as scopt
import scipy.special as scsp
import matplotlib.pyplot as plt

#The following is a basic heat transfer model for pulsed Nd:YAG laser welding of stainless steel
# Based off of "Effect of Laser Spot Weld Energy and Duration on Melting and Absorption" by PW Fuerschbach and GR Eisler

P = 200 #laser power (W)
t_p = 20e-3#pulse duration (s)

wavelength = 1.07E-6 #Central Emission Wavelength (m)
angular_freq = 2*np.pi*299792458/wavelength #angular frequency of laser wave (rad/s)

epsilon_0 = 8.854e-12 #permittivity of free space (see "The Laser Welding of Thin Metal Sheets: An Integrated Keyhole and Weld Pool Model with Supporting Experiments" Pg 1622)
sigma_st = 1.45e6 # electrical conductance per unit depth (/Ohm/m)
epsilon = np.sqrt(2 / (1 + np.sqrt((1 + sigma_st / (angular_freq * epsilon_0)) ** 2)))  # material constant for determining Fresnel absorption
R_0 = 4*np.sqrt((angular_freq/2*np.pi)*np.pi*epsilon_0/sigma_st) #reflection at normal incidence for steel (https://eng.libretexts.org/Bookshelves/Materials_Science/Supplemental_Modules_(Materials_Science)/Optical_Properties/Metallic_Reflection)

q = P*R_0 #absorbed power (W)

T_v = 3143 #vaporisation temperature of iron (K)
T_m = 1727 #melting temperature of steel (K)
rho = 8000 #density of steel (kg/m^3)
c_p = 500#specific heat capacity of steel (J/kgK)
k = 16.2 #thermal conductivity of high chromium (10%) steel (W/mK)

a = k/(rho*c_p) #thermal diffusivity (m^2/s)

T_0 = 300 #initial temperature (K)

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
    if temp_function(r_1, t) * temp_function(r_2, t) >= 0:
        print("Secant method fails.")
        return None

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
            return None
    return -1*(r_1 - temp_function(r_1, t) * (r_2 - r_1) / (temp_function(r_2, t) - temp_function(r_1, t)))

# t_star = scopt.fminbound(radius_solver, t_min, t_max)

t_min = t_p
t_max = t_p*1.5

t_opt = scopt.fminbound(sec_method_T, t_min, t_max)
r_max = -1*sec_method_T(t_opt)

theta = np.linspace(np.pi, np.pi*2, 1000)
x_m = np.zeros(1000)
y_m = np.zeros(1000)

for J in range(0, 1000):
    x_m[J] = r_max*np.cos(theta[J])
    y_m[J] = r_max*np.sin(theta[J])

plt.figure(1)
plt.plot(x_m*1000, y_m*1000)
plt.xlabel("X Axis (mm)")
plt.ylabel("Y Axis (mm)")
plt.title("Laser Conduction Weld Results (P = " +str(np.round(P,2))+ " W, t = " +str(np.round(t_p*1000,2))+ " ms)")