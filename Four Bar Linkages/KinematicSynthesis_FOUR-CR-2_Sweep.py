import numpy as np
import matplotlib.pyplot as plt

# The following script determines kinematic properties of a four bar mechanism for a given swing angle and rocker angle

psi_0 = np.deg2rad(26.91)
psi = np.deg2rad(14.88)

th_min = np.deg2rad(7.5)
th_max = np.deg2rad(15)

c = 23.27
N = 1000

f = 2500 #force required for tip deflection (N)

th_arr = np.linspace(th_min, th_max, N)

d_TA_min = np.zeros(N)
d_TA_max = np.zeros(N)
A = np.zeros(N)
B = np.zeros(N)
D = np.zeros(N)
TH_0 = np.zeros(N)
T_A = np.zeros(N) #Crank Torque (Nm)

for I in range(0,N):

    th = th_arr[I]

    th_f = th + np.pi

    y = psi_0 + psi - th_f

    m_2 = -np.sin(psi_0)*np.sin(th_f)
    m_1 = np.sin(y) + np.cos(psi_0)*np.sin(th_f)  - np.sin(psi_0)*np.cos(th_f)
    m_0 = np.cos(psi_0)*np.cos(th_f) - np.cos(y)

    coefs_x = [m_0, m_1, m_2]
    x = np.roots(coefs_x)

    th_0 = np.arctan(x)

    p_1 = np.sin(psi_0)/np.sin(th_0)
    p_2 = np.sin(psi + psi_0)/np.sin(th + th_0)
    p_3 = np.sin(psi_0 - th_0)

    d = c * p_3 / np.sin(th_0)
    b = c * d * (p_1 + p_2) / 2
    a = c * d * (p_1 - p_2) / 2

    for i in range(0,len(a)):
        if a[i]>0 and b[i]>0 and d[i] >0:
            a_sol = a[i]
            b_sol = b[i]
            d_sol = d[i]
            th_0_sol = th_0[i]

    d_TA_min[I] = abs(np.pi/2 - np.arccos((b_sol**2 + c**2 - (1 - a_sol)**2)/(2*b_sol*c)))
    d_TA_max[I] = abs(np.pi/2 - np.arccos((b_sol**2 + c**2 - (1 + a_sol)**2)/(2*b_sol*c)))

    A[I] = a_sol
    B[I] = b_sol
    D[I] = d_sol
    TH_0[I] = th_0_sol

    L_D = d_sol/c
    L_C = d_sol

    alpha = np.deg2rad(15)

    X_C = L_D - np.sin(psi_0 - np.pi/2)*L_C
    Y_C = L_C*np.cos(psi_0 - np.pi/2)

    T_A[I] = f*(np.cos(alpha)*Y_C - np.sin(alpha)*X_C)

globmin_max_TA = np.nanmin(d_TA_max)
index_minTA = np.where(d_TA_max == globmin_max_TA)

th_min = th_arr[index_minTA]

A_min = A[index_minTA]
B_min = B[index_minTA]
D_min = D[index_minTA]

Th_0_min = TH_0[index_minTA]

TA_min = np.rad2deg(np.arccos((B_min**2 + c**2 - (1 - A_min)**2)/(2*B_min*c)))
TA_max = np.rad2deg(np.arccos((B_min**2 + c**2 - (1 + A_min)**2)/(2*B_min*c)))
T_A_min = T_A[index_minTA]

plt.figure(1)
plt.plot(th_arr*57.3, d_TA_min*57.3,label="Minimum")
plt.plot(th_arr*57.3, d_TA_max*57.3,label="Maximum")
plt.xlabel("Crank Angle Offset (deg)")
plt.ylabel("Transmission Angle Deviation (deg)")
plt.grid()
plt.legend()

plt.figure(2)
plt.plot(th_arr*57.3, T_A/1000)
plt.xlabel("Crank Angle Offset (deg)")
plt.ylabel("Crank Torque (Nm)")
plt.grid()

plt.figure(3)
plt.plot(th_arr*57.3, A, label="A")
plt.plot(th_arr*57.3, B, label="B")
plt.plot(th_arr*57.3, D, label="D")
plt.xlabel("Starting Crank Angle (deg)")
plt.ylabel("Linkage Length (mm)")
plt.grid()
plt.legend()

A_str = str(round(A_min[0],2))
B_str = str(round(B_min[0],2))
C_str = str(c)
D_str = str(round(D_min[0],2))

TA_min_str = str(round(TA_min[0],2))
TA_max_str = str(round(TA_max[0],2))

print("Initial Rocker Angle = " +str(round(np.rad2deg(psi_0),2))+ "deg")
print("Rocker Swept Angle = " +str(round(np.rad2deg(psi),2))+ "deg")

print("--------------------------------")
print("Link Lengths")
print("A = " +A_str+ "mm")
print("B = " +B_str+ "mm")
print("C = " +C_str+ "mm")
print("D = " +D_str+ "mm")

print("Min Transmission Angle = " +TA_min_str+ " deg")
print("Max Transmission Angle = " +TA_max_str+ " deg")
print("Crank Torque = " +str(round(T_A_min[0]/1000,2))+ "Nm")