import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import os
from scipy import interpolate

# This script optimises the TKT system with respect to fatigue life
# Assumes that weight is split equally by top and lower mounts, so that flexure most likely to fail is top rear

def kinematic(pm_x, t, L, dtheta, face_width): #kinematics
    bend_rad = L/dtheta #flexure bend radius
    x_rest = pm_x/(np.cos(np.pi/4 - dtheta)*np.sqrt(2)-1) #length from linkage end-point to mounting plate
    link_length = np.sqrt(2)*x_rest #hypotenuse length from linkage end-point to mounting plate
    total_length = x_rest + pm_x #total length of torque arm due to assembly weight (assumed)
    return [bend_rad, x_rest, link_length, total_length]

def bending_mechanics(t, bend_rad, yield_stress, E, E_t): #simple bending mechanics
    total_strain = t/(2*bend_rad) #total bending strain
    elastic_strain = yield_stress/E #elastic strain

    if total_strain > elastic_strain: #simple strain hardening model for determining total stress
        plastic_strain = total_strain - elastic_strain
        bending_stress = yield_stress + E_t*plastic_strain
        plastic_zone = yield_stress*bend_rad/E
    else:
        bending_stress = total_strain*E #elastic stress
        plastic_zone = 0

    return [bending_stress, plastic_zone, total_strain]

def stress_concentration(external_rad, d, H): #stress concentration as function of flexure geometry
    H_d_ratio = np.round(H/d,2)
    r_d_ratio = external_rad/d
    os.chdir(r"E:\Users\Tannlin User\Documents\Parallel Kinematic Laser Head Profile\Literature\Stress Concentration")

    #   interpolates stress concentration from lookup tables

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

def shear_mechanics(external_rad, d, H, m, total_length, t): #calculates shear stress
    nom_stress = 6*m*9.81*total_length/(t*d**2) #nominal stress for determining shear stress
    K_t = stress_concentration(external_rad, d, H)
    K_t = float(K_t)
    shear_stress = nom_stress*K_t
    return shear_stress

def multiaxial_stress(shear_stress, bending_stress): #determines resultant von mises stress due to multiaxial stress state
    principal_stress_1 = bending_stress/2 + np.sqrt((bending_stress/2)**2 + shear_stress**2)
    principal_stress_2 = bending_stress/2 - np.sqrt((bending_stress/2)**2 + shear_stress**2)

    von_mises_stress = (1/np.sqrt(2))*np.sqrt((principal_stress_1-principal_stress_2)**2)
    tresca_stress = abs(principal_stress_1-principal_stress_2)

    return von_mises_stress, tresca_stress

def fatigue_life_FS(gamma_max, s_n_max, yield_stress, v_e, v_p, UTS, E, b, c ): #uses multiaxial Fatemi-Socie fatigue model to predict service life

    #Uses Uniform Material Law

    s_f = 1.5 * UTS

    if UTS / E <= 0.003:
        psi = 1
    if UTS / E >= 0.003:
        psi = 1.375 - 125 * UTS / E

    e_f = 0.59 * psi

    n = 0.6

    #Simple brute-force method of calculating fatigue life

    N_min = 1
    N_max = 10
    N_N = 100*((N_max-N_min)+1)
    N = np.logspace(N_min,N_max, num = N_N)

    for i in range(0, N_N):
        RHS_FS = gamma_max * (1 + n * s_n_max / yield_stress)
        LHS_FS = (1 + v_e) * (s_f / E) * (2 * N[i])**b + (n / 2) * (1 + v_e) * ((s_f**2) / (E * yield_stress)) * (
                    2 * N[i]) ** (2 * b) + (1 + v_p) * e_f * (2 * N[i])**c + (n / 2) * (1 + v_p) * (
                             e_f * s_f / yield_stress) * (2 * N[i])**(b + c)
        if RHS_FS - LHS_FS >= 0:
            return N[i]

def simulation(pm_x, t, L, dtheta, facewidth, yield_stress, E, E_t, external_rad, d, H, m, G, v_e, v_p, UTS, b, c): #main script

    bend_rad, x_rest, link_length, total_length = kinematic(pm_x, t, L, dtheta, face_width)
    bending_stress, plastic_zone, total_strain = bending_mechanics(t, bend_rad, yield_stress, E, E_t)
    shear_stress = shear_mechanics(external_rad, d, H, m, total_length, t)
    von_mises_stress, tresca_stress = multiaxial_stress(shear_stress, bending_stress)

    shear_strain = shear_stress/G

    N_FS = fatigue_life_FS(shear_strain, bending_stress, yield_stress, v_e, v_p, UTS, E, b, c)

    return von_mises_stress, tresca_stress, N_FS, x_rest, link_length

#Define Kinematic Variables

pm_x = 20 #permitted travel from rest (mm)
t = 0.15 #flexure thickness (mm)
face_width = 10 #Face width (mm)
external_rad = 5*0.5 #External radius (mm)
d = 12.45 #Bend Section Depth (mm)
H = 25.4 #Flexure Height (mm)
m = 1.2 #Total mass acting on top flexures (kg)

#Define Material Variables

yield_stress = 330 #Yield Stress (MPa)
E = 193e3  #Young's modulus (MPa)
E_t = 0.085*E #Tangent modulus (MPa)
G = 74e3# Shear modulus (MPa)
K = 134e3 # Bulk modulus (GPa)
v_e = 0.29 # Elastic poisson's ratio
v_p = 0.5-E_t/(6*K) # Plastic poisson's ratio (Lames equation)
UTS = 590 #Ultimate tensile strength (MPa)

#Define Fatigue Variables

b = -0.087 # Fatigue strength exponent
c = -0.58 # Fatigue ductility exponent

#Define search routine limits

N_steps = 100

L_min = 5
L_max = 12

L_array = np.linspace(L_min, L_max, num = N_steps)

dtheta_min = np.deg2rad(5)
dtheta_max = np.deg2rad(40)
dtheta_array = np.linspace(dtheta_min, dtheta_max, num=N_steps)

#Memory preallocation

vm_stress = np.zeros((N_steps, N_steps))
tr_stress = np.zeros((N_steps, N_steps))
N_cycles_FS = np.zeros((N_steps, N_steps))
x_rest_arr = np.zeros((N_steps,N_steps))
link_length_arr = np.zeros((N_steps,N_steps))

for i in range(0,N_steps):
    L = L_array[i]
    dtheta_max_L = np.arctan((0.5*face_width + 0.5*L - 3.5)/(face_width + 0.5*L)) #maximum angle of travel to avoid collision
    for j in range(0,N_steps):
        dtheta = dtheta_array[j]

        von_mises_stress, tresca_stress, N_FS, x_rest, link_length = simulation(pm_x, t, L, dtheta, face_width, yield_stress, E, E_t, external_rad, d, H, m, G, v_e, v_p, UTS, b, c)

        dtheta_y_clearance = np.pi / 4 - np.arcsin(2*(0.5 * L + face_width) / link_length) #minimum angle of travel to allow for clearance in y direction

        if dtheta >= dtheta_y_clearance or dtheta >= dtheta_max_L:
            vm_stress[i, j] = 0
            N_cycles_FS[i, j] = 0
            x_rest_arr[i, j] = 0
            link_length_arr[i, j] = 0

        else:

            vm_stress[i, j] = von_mises_stress
            N_cycles_FS[i, j] = N_FS
            x_rest_arr[i,j] = x_rest
            link_length_arr[i, j] = link_length

    print(i*100/N_steps)

#index results to find minimum von mises stress:

vm_stress_nonzero = np.ma.masked_equal(vm_stress, 0)
min_stress = np.amin(vm_stress_nonzero)
index_min_stress = np.where(vm_stress == min_stress)

L_min_stress = L_array[index_min_stress[0]]
dtheta_min_stress = 57.3*dtheta_array[index_min_stress[1]]

x_rest_min_stress = x_rest_arr[index_min_stress]
link_length_min_stress = link_length_arr[index_min_stress]

linkarm_length_min_stress = link_length_min_stress - 2 * (face_width + 0.5*L_min_stress)/np.cos(np.pi/4)

#index results to find maximum fatigue life:

max_N_FS = np.amax(N_cycles_FS)
index_max_cycles_FS = np.where(N_cycles_FS == max_N_FS)

L_max_N_FS = L_array[index_max_cycles_FS[0]]
dtheta_max_N_FS = dtheta_array[index_max_cycles_FS[1]]

index_max_N_min_L = np.where(L_array == min(L_max_N_FS))
index_max_N_min_dtheta = np.where(dtheta_array == min(dtheta_max_N_FS))

L_max_N_min_L = L_array[index_max_N_min_L]

dtheta_max_N_min_dtheta = dtheta_array[index_max_N_min_dtheta]
dtheta_max_N_min_dtheta = 57.3*dtheta_max_N_min_dtheta

link_length_max_N_min_L = link_length_arr[index_max_N_min_L, index_max_N_min_dtheta]
link_length_max_N_min_L = link_length_max_N_min_L - np.sqrt(2)*(face_width + 0.5*L_max_N_min_L) - np.sqrt(2)*(0.5*face_width + 0.5 * L_max_N_min_L)
x_rest_pos = x_rest_arr[index_max_N_min_L, index_max_N_min_dtheta]

print("----------------------------")
print("Results - Minimum Stress")
print("Bending Length: %.2f mm" % L_min_stress)
print("Swept Angle: %.2f degrees" % dtheta_min_stress)
print("Link Arm Length: %.2f mm" % linkarm_length_min_stress)
print("Resting X Position: %.2f mm" % x_rest_min_stress)

print("-----------------------------")
print("Results - Maximum Fatigue Life")
print("Bending Length: %.2f mm" % L_max_N_min_L)
print("Swept Angle: %.2f degrees" % dtheta_max_N_min_dtheta)
print("Link Arm Length: %.2f mm" % link_length_max_N_min_L)
print("Resting X Position: %.2f mm" % x_rest_pos)

dtheta_array, L_array = np.meshgrid(dtheta_array, L_array)

fig = plt.figure(1)
ax = fig.gca(projection='3d')
surf = ax.plot_surface(dtheta_array*57.3, L_array, vm_stress, cmap = cm.coolwarm, linewidth=0, antialiased=False)
plt.xlabel("Change in Angle (deg)")
plt.ylabel("Bending Length (mm)")
plt.title("Stress Optimisation")

fig = plt.figure(2)
ax = fig.gca(projection='3d')
surf = ax.plot_surface(dtheta_array*57.3, L_array, N_cycles_FS, cmap = cm.coolwarm, linewidth=0, antialiased=False)
plt.xlabel("Change in Angle (deg)")
plt.ylabel("Bending Length (mm)")
plt.title("Number of Cycles Optimisation (Fatemi-Socie")


