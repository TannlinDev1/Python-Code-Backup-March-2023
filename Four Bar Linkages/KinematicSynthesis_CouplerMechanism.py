import numpy as np
import matplotlib.pyplot as plt

# This script generates a four bar mechanism with a coupler link to follow three points. It also calculates the crank torque required for a 5 N/mm frame
# Follows from Pages 148 to 152 of "Analytical Kinematics, Analysis and Synthesis of Planar Mechanisms by Roger F Gans

#Points to be followed relative to centre of crank

d_P0P1 = 0.21 - 0.5j #distance between P_0 and P_1
d_P1P2 = 0 - 0.2j #distance between P_1 and P_2

#Control Variables:

z_1 = -100 #distance between base pivot points
phi_21 = np.deg2rad(3) # Change in angle of crank when moving from P_0 to P_1 (rad)
phi_22 = phi_21 + np.deg2rad(0.75) # Change in angle of crank when moving from P_0 to P_2 (rad)
P_0 = 75 + 50j #P_0 relevant to point A (input variable)

P_1 = P_0 + d_P0P1

P_2  = P_1 + d_P1P2

def vector_dot(z1, z2):
    th_1 = np.angle(z1)
    r_1 = np.sqrt(np.real(z1)**2 + np.imag(z1)**2)
    th_2 = np.angle(z2)
    r_2 = np.sqrt(np.real(z2)**2 + np.imag(z2)**2)

    return complex(r_1*r_2*np.cos(th_2-th_1), r_1*r_2*np.sin(th_2-th_1))

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
z_42 = -(z_1 + z_22 + z_32)

#Defining XY points for plotting full linkage:

XY = np.zeros((6,2))

XY[1,0] = np.real(z_20)# Point B Start (X)
XY[1,1] = np.imag(z_20)# Point B Start (Y)
XY[2,0] = XY[1,0] + np.real(z_3)# Point C Start (X)
XY[2,1] = XY[1,1] + np.imag(z_3)# Point C Start (Y)
XY[3,0] = abs(z_1)# Point D Start (X)
XY[4,0] = np.real(P_0)# Point P Start (X)
XY[4,1] = np.imag(P_0)# Point P Start (Y)
XY[5,0] = XY[2,0]
XY[5,1] = XY[2,1]

XY_1 = np.zeros((6,2))

XY_1[1,0] = r_2*np.cos((np.angle(z_20) + phi_21))# Point B Middle (X)
XY_1[1,1] = r_2*np.sin((np.angle(z_20) + phi_21))# Point B Middle (Y)
XY_1[2,0] = XY_1[1,0] + r_3*np.cos(np.angle(z_3) + phi_31)# Point C Middle (X)
XY_1[2,1] = XY_1[1,1] + r_3*np.sin(np.angle(z_3) + phi_31)# Point C Middle (Y)
XY_1[3,0] = abs(z_1)# Point D Start (X)
XY_1[4,0] = np.real(P_1)# Point P Middle (X)
XY_1[4,1] = np.imag(P_1)# Point P Middle (Y)
XY_1[5,0] = XY_1[2,0]
XY_1[5,1] = XY_1[2,1]

XY_2 = np.zeros((6,2))

XY_2[1,0] = r_2*np.cos((np.angle(z_20) + phi_22))# Point B Middle (X)
XY_2[1,1] = r_2*np.sin((np.angle(z_20) + phi_22))# Point B Middle (Y)
XY_2[2,0] = XY_2[1,0] + r_3*np.cos(np.angle(z_3) + phi_32)# Point C Middle (X)
XY_2[2,1] = XY_2[1,1] + r_3*np.sin(np.angle(z_3) + phi_32)# Point C Middle (Y)
XY_2[3,0] = abs(z_1)# Point D Start (X)
XY_2[4,0] = np.real(P_2)# Point P Middle (X)
XY_2[4,1] = np.imag(P_2)# Point P Middle (Y)
XY_2[5,0] = XY_2[2,0]
XY_2[5,1] = XY_2[2,1]

r2_str = str(round(r_2,2))
r3_str = str(round(r_3,3))
r4_str = str(round(r_4,2))
w2_str = str(round(w_2,2))
w4_str = str(round(w_4,2))

# Crank torque at crimping stage:

z_PB = P_2 - z_22
f_2 = 0 - 8250j #assuming crimping force acts only in the vertical (15 N/mm for a 550 mm long flexure)
M_A2 = np.imag(vector_dot(z_22, f_2)) - (np.imag(vector_dot(z_PB, f_2))/np.imag(vector_dot(z_32, z_42)))*np.imag(vector_dot(z_22, z_42))

f_0 = -2750 + 0j
z_P0B = P_0 - z_20
M_A0 = np.imag(vector_dot(z_20, f_0)) - (np.imag(vector_dot(z_P0B, f_0))/np.imag(vector_dot(z_3, z_40)))*np.imag(vector_dot(z_20, z_40))

print("--------------------------------")
print("Link Lengths")
print("L2 = " +r2_str+ " mm")
print("L3 = " +r3_str+ " mm")
print("L4 = " +r4_str+ " mm")
print("W2 = " +w2_str+ " mm")
print("W4 = " +w4_str+ " mm")
print("--------------------------------")
print("Crank Torque for Crimping")
print("M_A = " +str(np.round(abs(M_A2)/1000, 2))+ " Nm")
print("--------------------------------")
print("Crank Torque for Opening")
print("M_A = " +str(np.round(abs(M_A0)/1000, 2))+ " Nm")

# plot path tool point P

N = 100

th_2i = np.angle(z_20) #link arm 2 initial angle
th_2m = th_2i + phi_21 #link arm 2 mid angle
th_2f = np.angle(z_22) #link arm 2 final angle

th_2im = np.linspace(th_2i, th_2m, N)
th_2mf = np.linspace(th_2f, th_2m, N)

th_4i = np.angle(z_40) #link arm 4 initial angle

z_31 = Z_phi_31*z_3
z_21 = complex(np.cos(phi_21), np.sin(phi_21))*z_20
z_41 = -(z_21 + z_31 + z_1)

th_4m = np.angle(z_41) #link arm 4 mid angle
th_4f = np.angle(z_42) #link arm 4 final angle

th_4im = np.linspace(th_4i, th_4m, N)
th_4mf = np.linspace(th_4f, th_4m, N)

th_LHD = np.arccos((r_3**2 + w_2**2 - w_4**2)/(2*r_3*w_2)) #Left Hand Dyad angle (cosine rule)

def path(th_2, th_4, N):

    P = np.zeros((N, 2))

    for i in range(0, N):

        z4 = complex(np.cos(th_4[i])*r_4, np.sin(th_4[i])*r_4)
        z2 = complex(np.cos(th_2[i])*r_2, np.sin(th_2[i])*r_2)
        z3 = -(z_1 + z2 + z4)
        th_w2 = th_LHD + np.angle(z3)

        w2 = complex(np.cos(th_w2)*w_2, np.sin(th_w2)*w_2)
        P_v = z2 + w2
        P[i,0] = np.real(P_v)
        P[i,1] = np.imag(P_v)

    return P

P1 = path(th_2im, th_4im, N)#calculate initial to mid path
P2 = path(th_2mf, th_4mf, N)#calculate mid to final path

for i in range(0, 3, 1):
    plt.plot(XY[i:i+2,0], XY[i:i+2,1], 'ro-')

plt.plot(np.array([XY[1,0], XY[4,0]]), np.array([XY[1,1], XY[4,1]]), 'ro-')
plt.plot(np.array([XY[4,0], XY[2,0]]), np.array([XY[4,1], XY[2,1]]), 'ro-')

for i in range(0, 3, 1):
    plt.plot(XY_1[i:i+2,0], XY_1[i:i+2,1], 'go-')

plt.plot(np.array([XY_1[1,0], XY_1[4,0]]), np.array([XY_1[1,1], XY_1[4,1]]), 'go-')
plt.plot(np.array([XY_1[4,0], XY_1[2,0]]), np.array([XY_1[4,1], XY_1[2,1]]), 'go-')

for i in range(0, 3, 1):
    plt.plot(XY_2[i:i+2,0], XY_2[i:i+2,1], 'bo-')

plt.plot(np.array([XY_2[1,0], XY_2[4,0]]), np.array([XY_2[1,1], XY_2[4,1]]), 'bo-')
plt.plot(np.array([XY_2[4,0], XY_2[2,0]]), np.array([XY_2[4,1], XY_2[2,1]]), 'bo-')

plt.plot(P1[:,0], P1[:,1],color="tab:orange", linestyle="dashed")
plt.plot(P2[:,0], P2[:,1],color="tab:orange", linestyle="dashed")

plt.grid()
plt.title("4 Bar Linkage Path Generation")
plt.xlabel("X Coordinate (mm)")
plt.ylabel("Y Coordinate (mm)")

