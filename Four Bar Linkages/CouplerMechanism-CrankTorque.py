import numpy as np
import matplotlib.pyplot as plt

# This script generates a four bar mechanism with a coupler link to follow three points. It also calculates the crank torque required for a 5 N/mm frame
# Follows from Pages 148 to 152 of "Analytical Kinematics, Analysis and Synthesis of Planar Mechanisms by Roger F Gans

#Points to be followed relative to centre of crank

z_1_min = 50
z_1_max = 250
N = 100
z_1_arr = np.linspace(z_1_min, z_1_max, N)
M_A_arr = np.zeros(N)

def vector_dot(z1, z2):
    th_1 = np.angle(z1)
    r_1 = np.sqrt(np.real(z1)**2 + np.imag(z1)**2)
    th_2 = np.angle(z2)
    r_2 = np.sqrt(np.real(z2)**2 + np.imag(z2)**2)

    return complex(r_1*r_2*np.cos(th_2-th_1), r_1*r_2*np.sin(th_2-th_1))

for i in range(0, N):

    z_1 = -z_1_arr[i]

    d_P0P1 = 0.6 - 0.64j #distance between P_0 and P_1
    d_P1P2 = 0 - 0.6j #distance between P_1 and P_2

    P_0 = 150 + 100j #P_0 relevant to point A (input variable)

    P_1 = P_0 + d_P0P1

    P_2  = P_1 + d_P1P2

    # P_0 = 3.5543 + 5j
    # P_1 = 3.7492 + 5.9084j
    # P_2  = 2.8085 + 5.8478j
    #
    # z_1 = -6 # Length of ground pivot arm

    phi_21 = np.deg2rad(5) # Change in angle of crank when moving from P_0 to P_1 (rad)
    phi_22 = phi_21 + np.deg2rad(1.5) # Change in angle of crank when moving from P_0 to P_2 (rad)

    f_1 = complex(np.cos(-phi_21), np.sin(-phi_21))*P_1 - P_0
    f_2 = complex(np.cos(-phi_22), np.sin(-phi_22))*P_2 - P_0

    z_20 = (f_1*P_2*np.conj(P_2) - f_2*P_1*np.conj(P_1) - (f_1 - f_2)*P_0*np.conj(P_0))/(f_1*np.conj(f_2) - f_2*np.conj(f_1))

    w_20 = P_0 - z_20

    Z_phi_31 = (P_1 - complex(np.cos(phi_21), np.sin(phi_21))*z_20)/w_20
    Z_phi_32 = (P_2 - complex(np.cos(phi_22), np.sin(phi_22))*z_20)/w_20

    phi_31 = np.angle(Z_phi_31)
    phi_32 = np.angle(Z_phi_32)

    q_0 = P_0 + z_1
    q_1 = P_1 + z_1
    q_2 = P_2 + z_1

    g_1 = (complex(np.cos(-phi_31), np.sin(-phi_31))*q_1 - q_0)
    g_2 = (complex(np.cos(-phi_32), np.sin(-phi_32))*q_2 - q_0)

    w_40 = -(g_1*q_2*np.conj(q_2) - g_2*q_1*np.conj(q_1) - (g_1 - g_2)*q_0*np.conj(q_0))/(g_1*np.conj(g_2) - g_2*np.conj(g_1))

    z_40 = -(q_0 + w_40)
    z_3 = w_20 + w_40

    r_2 = abs(z_20)
    r_3 = abs(z_3)
    r_4 = abs(z_40)
    w_4 = abs(w_40)
    w_2 = abs(w_20)

    z_32 = complex(r_3*np.cos(np.angle(z_3) + phi_32), r_3*np.sin(np.angle(z_3) + phi_32))
    z_22 = complex(r_2*np.cos(np.angle(z_20) + phi_22), r_2*np.sin(np.angle(z_20) + phi_22))
    z_PB = P_2 - z_22
    z_42 = -(z_1 + z_22 + z_32)

    # Crank torque at crimping stage:

    f = 0 + 8250j #assuming crimping force acts only in the vertical (15 N/mm for a 550 mm long flexure)

    M_A = np.imag(vector_dot(z_22, f)) - (np.imag(vector_dot(z_PB, f)) / np.imag(vector_dot(z_32, z_42))) * np.imag(
        vector_dot(z_22, z_42))

    M_A_arr[i] = abs(M_A)

plt.figure(1)
plt.plot(z_1_arr, M_A_arr/1000)
plt.xlabel("Base Length (mm)")
plt.ylabel("Crank Torque (Nm)")


