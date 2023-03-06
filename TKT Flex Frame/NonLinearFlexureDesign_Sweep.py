import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

# Searching algorithm for determining smallest "L" and "t" to give target force at given deflection (nonlinear mechanics)

s_y = 735# yield stress (MPa)
E = 200e3# young's modulus (MPa)
e = 0.1
c = 0.75# percentage of half thickness which is still elastic
N = 2# number of flexures

P_lim = 6 #target tension per mm depth (N)
d_lim = 2 #target tip displacement (mm)

# flexure thickness search window

t_min = 0.3
t_max = 1.2
t_num = 100
t = np.linspace(t_min,t_max, t_num)

# flexure length search window

L_min = 2
L_max = 25
L_num = 100
L = np.linspace(L_min, L_max, L_num)

P = np.zeros((t_num, L_num))
d = np.zeros((t_num, L_num))
t_sol = np.zeros((t_num, L_num))
L_sol = np.zeros((t_num, L_num))
P_sol = np.zeros((t_num, L_num))
d_sol = np.zeros((t_num, L_num))

for i in range(0,t_num):
    for j in range(0, L_num):
        M = (3 - c**2 + (2 - c + c**3)*e/c)*(s_y*t[i]**2)/12 #moment as function of plastic zone (Nmm)
        P[j,i] = M*N/L[j] # tensioning force (N)
        P_y = (s_y*t[i]**2)/(6*L[j]) # yield force (N)
        d_y = (4*P_y*L[j]**3)/(E*t[i]**3) #yield deflection (mm)
        d[j,i] = d_y * (P_y/(P[j, i]/N))**2 * (5 - (3 + (P[j, i]/N)/P_y)*np.sqrt(3 - 2 * (P[j, i]/N)/ P_y)) # deflection (mm)

        if P[j,i] > P_lim and d[j,i] > d_lim: #terminating statement
            t_sol[j,i] = t[i]
            L_sol[j,i] = L[j]
            P_sol[j,i]= P[j,i]
            d_sol[j,i]= d[j, i]
