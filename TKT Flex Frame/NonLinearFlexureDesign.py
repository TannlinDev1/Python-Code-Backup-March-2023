import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

# Searching algorithm for determining smallest "L" and "t" to give target force at given deflection (nonlinear mechanics)

s_y = 1997# yield stress (MPa)
s_u = 2106 # ultimate tensile stress (MPa)

E = 212.4# young's modulus (MPa)
e_y = s_y/E# yield strain

# The following uses a continuous strength method as detailed in the Design Manual for Structural Stainless Steel 4th Edition

C_2 = 0.16 #CSM material coefficient 2 (0.16 for austenitic and duplex, 0.45 for ferritic)
C_3 = 1 #CSM material coefficient 3 (1 for austentic and duplex, 0.6 for ferritic)

e_u = C_3*(1 - s_y/s_u) # ultimate tensile strain
E_t = (s_u - s_y)/(C_2*e_u - e_y) # strain hardening modulus

# e = E_t/E #ratio of tangent to elastic modulus

e = 0.085

c = 0.4# percentage of half thickness which is still elastic
N = 1# number of flexures

P_lim = 8 #target tension per mm depth (N)
d_lim = 3 #target tip displacement (mm)

# flexure thickness search window

t_min = 0.5
t_max = 3
t_num = 10
t = np.linspace(t_min,t_max, t_num)

# flexure length search window

L_min = 5
L_max = 50
L_num = 10
L = np.linspace(L_min, L_max, L_num)

P = np.zeros((t_num, L_num))
d = np.zeros((t_num, L_num))

def GeoSolver(t_num, L_num, c, e, s_y, t, L, E, P_lim, d_lim):
    for i in range(0,t_num):
        for j in range(0, L_num):
            M = (3 - c**2 + (2 - c + c**3)*e/c)*(s_y*t[i]**2)/12 #moment as function of plastic zone (Nmm)
            P[j,i] = M*N/L[j] # tensioning force (N)
            P_y = (s_y*t[i]**2)/(6*L[j]) # yield force (N)
            d_y = (4*P_y*L[j]**3)/(E*t[i]**3) #yield deflection (mm)
            d[j,i] = d_y * (P_y/(P[j, i]/N))**2 * (5 - (3 + (P[j, i]/N)/P_y)*np.sqrt(3 - 2 * (P[j, i]/N)/ P_y)) # deflection (mm)

            if P[j,i] > P_lim and d[j,i] > d_lim: #terminating statement
                t_sol = t[i]
                L_sol = L[j]
                P_sol= P[j,i]
                d_sol= d[j, i]

                return [t_sol, L_sol, P_sol, d_sol]

sol_geo = GeoSolver(t_num, L_num, c, e, s_y, t, L, E, P_lim, d_lim)

t_str = str(round(sol_geo[0],2))
L_str = str(round(sol_geo[1],3))
P_str = str(round(sol_geo[2],2))
d_str = str(round(sol_geo[3],2))

print("For " +str(N)+ " flexures, at " +str(c*100)+ "% plasticity")
print("----------------------------------------------------")
print("Flexure Thickness = " +t_str+ "mm")
print("Flexure Length = " +L_str+ "mm")
print("Force = " +P_str+ "N")
print("Displacement = " +d_str+ "mm")

# t_arr, L_arr = np.meshgrid(t, L)

# fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
# surf = ax.plot_surface(t_arr, L_arr, d, cmap = cm.coolwarm, linewidth=0, antialiased=False)
# ax.set_xlabel("Thickness (mm)")
# ax.set_ylabel("Length (mm)")
# ax.set_zlabel("Displacement (mm)")
# plt.title("Geometric Relations for Plastic Cantilever (c = 0.6, s_y = 965 MPa)")
#
# fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
# surf = ax.plot_surface(t_arr, L_arr, P, cmap = cm.coolwarm, linewidth=0.1, antialiased=False)
# ax.set_xlabel("Thickness (mm)")
# ax.set_ylabel("Length (mm)")
# ax.set_zlabel("Force (N)")
# plt.title("Kinetic Relations for Plastic Cantilever (c = 0.6, s_y = 965 MPa)")