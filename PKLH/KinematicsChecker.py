import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import os
from scipy import interpolate

# Optimisation script for determining kinematic properties

def kinematic(pm_x, t, L, dtheta, face_width):
    bend_rad = L/dtheta
    x_rest = pm_x/(np.cos(np.pi/4 - dtheta)*np.sqrt(2)-1)
    link_length = np.sqrt(2)*x_rest
    total_length = L + x_rest + face_width
    return [bend_rad, x_rest, link_length, total_length]

def bending_mechanics(t, bend_rad, yield_stress, E, E_t):
    total_strain = t/(2*bend_rad)
    elastic_strain = yield_stress/E

    if total_strain > elastic_strain:
        plastic_strain = total_strain - elastic_strain
        bending_stress = yield_stress + E_t*plastic_strain
        plastic_zone = yield_stress*bend_rad/E
    else:
        bending_stress = total_strain*E
        plastic_zone = 0

    return [bending_stress, plastic_zone, total_strain]

def stress_concentration(external_rad, d, H):
    H_d_ratio = np.round(H/d,2)
    r_d_ratio = external_rad/d
    os.chdir(r"E:\Users\Tannlin User\Documents\Parallel Kinematic Laser Head Profile\Literature\Stress Concentration")

    if H_d_ratio == 1.01 or H_d_ratio == 1:
        if r_d_ratio > 0.1:
            print("Ratio limits are out of bounds")
            return
        else:
            matrix_101 = np.loadtxt(fname="1.01.txt", delimiter = ",")
            f_101 = interpolate.interp1d(matrix_101[:,0],matrix_101[:,1])
            K_t = f_101(r_d_ratio)

    if 1.01 < H_d_ratio <= 1.03:
        if r_d_ratio > 0.1:
            print("Ratio limits are out of bounds")
            return
        else:
            matrix_102 = np.loadtxt(fname="1.02.txt", delimiter = ",")
            f_102 = interpolate.interp1d(matrix_102[:,0],matrix_102[:,1])
            K_t = f_102(r_d_ratio)

    if 1.04 <= H_d_ratio <= 1.07:
        matrix_103 = np.loadtxt(fname="1.03.txt", delimiter = ",")
        f_103 = interpolate.interp1d(matrix_103[:,0],matrix_103[:,1])
        K_t = f_103(r_d_ratio)

    if 1.08 <= H_d_ratio < 1.35:
        matrix_11 = np.loadtxt(fname="1.1.txt", delimiter = ",")
        f_11 = interpolate.interp1d(matrix_11[:,0],matrix_11[:,1])
        K_t = f_11(r_d_ratio)

    if 1.35 <= H_d_ratio < 1.75:
        matrix_15 = np.loadtxt(fname="1.5.txt", delimiter = ",")
        f_15 = interpolate.interp1d(matrix_15[:,0],matrix_15[:,1])
        K_t = f_15(r_d_ratio)

    if 1.75 <= H_d_ratio < 2.5:
        if r_d_ratio < 0.025:
            print("Ratio limits are out of bounds")
            return
        else:
            matrix_2 = np.loadtxt(fname="2.txt", delimiter = ",")
            f_2 = interpolate.interp1d(matrix_2[:,0],matrix_2[:,1])
            K_t = f_2(r_d_ratio)

    if 3 >= H_d_ratio >= 2.5:
        if r_d_ratio < 0.025:
            print("Ratio limits are out of bounds")
        else:
            matrix_3 = np.loadtxt(fname="3.txt", delimiter = ",")
            f_3 = interpolate.interp1d(matrix_3[:,0],matrix_3[:,1])
            K_t = f_3(r_d_ratio)

    if H_d_ratio < 1 or H_d_ratio > 3:
        print("H/d Ratio is out of bounds")

    return K_t

def shear_mechanics(external_rad, d, H, m, total_length, t):
    nom_stress = 6*m*9.81*total_length/(t*d**2)
    K_t = stress_concentration(external_rad, d, H)
    K_t = float(K_t)
    shear_stress = nom_stress*K_t
    return shear_stress

def multiaxial_stress(shear_stress, bending_stress):
    principal_stress_1 = bending_stress/2 + np.sqrt((bending_stress/2)**2 + shear_stress**2)
    principal_stress_2 = bending_stress/2 - np.sqrt((bending_stress/2)**2 + shear_stress**2)

    von_mises_stress = (1/np.sqrt(2))*np.sqrt((principal_stress_1-principal_stress_2)**2)
    tresca_stress = abs(principal_stress_1-principal_stress_2)

    return von_mises_stress, tresca_stress

def fatigue_life_FS(gamma_max, s_n_max, yield_stress, v_e, v_p, UTS, E, b, c ):

    #Uses Uniform Material Law

    s_f = 1.5 * UTS

    if UTS / E <= 0.003:
        psi = 1
    if UTS / E >= 0.003:
        psi = 1.375 - 125 * UTS / E

    e_f = 0.59 * psi

    n = 0.6

    N_min = 1
    N_max = 15
    N_N = 100*((N_max-N_min)+1)
    N = np.logspace(N_min,N_max, num = N_N)

    for i in range(0, N_N):
        RHS_FS = gamma_max * (1 + n * s_n_max / yield_stress)
        LHS_FS = (1 + v_e) * (s_f / E) * (2 * N[i])**b + (n / 2) * (1 + v_e) * ((s_f**2) / (E * yield_stress)) * (
                    2 * N[i]) ** (2 * b) + (1 + v_p) * e_f * (2 * N[i])**c + (n / 2) * (1 + v_p) * (
                             e_f * s_f / yield_stress) * (2 * N[i])**(b + c)
        if RHS_FS - LHS_FS >= 0:
            return N[i]

# def fatigue_life_SWT(von_mises_stress, strain_amplitude, E, UTS, b, c):
#
#     s_f = 1.5 * UTS
#
#     if UTS / E <= 0.003:
#         psi = 1
#     if UTS / E >= 0.003:
#         psi = 1.375 - 125 * UTS / E
#
#     e_f = 0.59 * psi
#
#     N_min = 1
#     N_max = 10
#     N_N = 100 * ((N_max - N_min) + 1)
#     N = np.logspace(N_min, N_max, num=N_N)
#
#     for i in range(0, N_N):
#         LHS_SWT = von_mises_stress*strain_amplitude*E
#         RHS_SWT = (s_f)**2 * (2*N[i])**(2*b) + s_f * e_f * E * (2*N[i])**(b+c)
#         if RHS_SWT <= LHS_SWT:
#             return N[i]

def simulation(pm_x, t, L, dtheta, facewidth, yield_stress, E, E_t, external_rad, d, H, m, G, v_e, v_p, UTS, b, c):

    bend_rad, x_rest, link_length, total_length = kinematic(pm_x, t, L, dtheta, face_width)
    bending_stress, plastic_zone, total_strain = bending_mechanics(t, bend_rad, yield_stress, E, E_t)
    shear_stress = shear_mechanics(external_rad, d, H, m, total_length, t)
    von_mises_stress, tresca_stress = multiaxial_stress(shear_stress, bending_stress)

    shear_strain = shear_stress/G

    N_FS = fatigue_life_FS(shear_strain, bending_stress, yield_stress, v_e, v_p, UTS, E, b, c)
    # N_SWT = fatigue_life_SWT(von_mises_stress, total_strain, E, UTS, b, c)

    return von_mises_stress, tresca_stress, N_FS, x_rest, link_length, #N_SWT

#Define Kinematic Variables

pm_x = 22.5 #permitted travel from rest (mm)
t = 0.15 #flexure thickness (mm)
face_width = 10 #Face width (mm)
external_rad = 5*0.5 #External radius (mm)
d = 12.45 #Bend Section Depth (mm)
H = 25.4 #Flexure Height (mm)
m = 0.25 #Laser/Collimator weight acting on top flexures (kg)

#Define Material Variables

yield_stress = 950 #Yield Stress (MPa)
E = 183e3  #Young's modulus (MPa)
E_t = E/10 #Tangent modulus (MPa)
v_e = 0.29 # Elastic poisson's ratio
K = E/(3*(1 - 2*v_e)) # Bulk modulus (GPa)
v_p = 0.5-E_t/(6*K) # Plastic poisson's ratio (Lames equation)
G = E/(2*(1+v_e))# Shear modulus (MPa)
UTS = 1200 #Ultimate tensile strength (MPa)

#Define Fatigue Variables

b = -0.126 # Fatigue strength exponent
c = -0.255 # Fatigue ductility exponent

L = 5.88*2
dtheta = np.deg2rad(24.93)

von_mises_stress, tresca_stress, N_FS, x_rest, link_length = simulation(pm_x, t, L, dtheta, face_width, yield_stress, E, E_t, external_rad, d, H, m, G, v_e, v_p, UTS, b, c)


