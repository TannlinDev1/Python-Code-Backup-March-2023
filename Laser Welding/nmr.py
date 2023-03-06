import numpy as np
import matplotlib.pyplot as plt

R_0 = 0.88
theta_min = np.deg2rad(2)
theta_max = np.deg2rad(30)
N = 100

theta_arr = np.linspace(theta_min, theta_max, N)

def fresnel_absorption_mr(n_mr): #fresnel absorption coefficient for number of multiple reflections
    alpha_mr = 1 - (R_0)**(n_mr-1)
    return alpha_mr

n_mr = np.zeros(N)
a_mr = np.zeros(N)

for i in range(0, N):
    n_mr[i] = np.pi/(4*np.nanmean(theta_arr[i])) #mean number of reflections of incident beam (need to double to give better weld results, don't know why yet)
    a_mr[i] = fresnel_absorption_mr(n_mr[i])

plt.figure(1)
plt.plot(np.rad2deg(theta_arr), n_mr)
plt.xlabel("Mean Wall Angle (deg)")
plt.ylabel("Number of Reflections")

plt.figure(2)
plt.plot(np.rad2deg(theta_arr), a_mr)
plt.xlabel("Mean Wall Angle (deg)")
plt.ylabel("Absorption Coefficient")