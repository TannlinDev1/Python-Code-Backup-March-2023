import numpy as np
import matplotlib.pyplot as plt

#This determines a double rocker mechanism's link lengths for optimum transmission characteristics
#Taken from "Synthesis of Double Rocker Mechanism with Optimum Transmission Characteristics" by AK Khare and RK Dave

# d_th = np.deg2rad(60)
# d_psi = np.deg2rad(30)
#
# mu_1 = np.pi/2 - (d_th - d_psi)/2
# mu_2 = np.pi/2 + (d_th - d_psi)/2
#
# th_1 = np.arctan((np.sin(mu_2)*(np.sin(d_th) + np.sin(d_th + mu_1) - np.sin(mu_1)))/(np.sin(mu_1)*(1 + np.cos(mu_2)) - np.sin(mu_2)*(np.cos(d_th) + np.cos(d_th + mu_1))))
#
# a = np.sin(th_1 + d_th + mu_1)
# b = (np.sin(th_1 + mu_2)/np.sin(mu_2)) - (np.sin(th_1 + d_th + mu_1)/np.sin(mu_1))
# c = np.sin(th_1)/np.sin(mu_2)

d_th_min = np.deg2rad(5)
d_th_max = np.deg2rad(40)
N_steps = 1000

f = 2500
alpha = np.deg2rad(2)

d_th_arr = np.linspace(d_th_min, d_th_max, N_steps)
R1 = np.zeros(N_steps)
R2 = np.zeros(N_steps)
R3 = np.zeros(N_steps)
mu_min = np.zeros(N_steps)
mu_max = np.zeros(N_steps)
T_A = np.zeros(N_steps)
T_D = np.zeros(N_steps)
TH_1 = np.zeros(N_steps)

d_psi = np.deg2rad(16.56)
psi_1 = np.deg2rad(108.98)

psi_2 = psi_1 + d_psi

R4 = 80

for I in range(0,N_steps):

    d_th = d_th_arr[I]
    gamma = psi_2 - d_th

    L =  np.cos(psi_1)*np.cos(d_th) - np.cos(gamma)
    M = np.sin(psi_1)*np.sin(d_th)
    N = np.sin(gamma) - np.sin(psi_1)*np.cos(d_th) + np.cos(psi_1)*np.sin(d_th)
    P = np.cos(psi_1)*np.sin(psi_2) - np.cos(gamma)*np.sin(psi_1)
    Q = np.sin(gamma)*np.sin(psi_1) - (np.sin(psi_2))**2

    e = 1/(L**2 - P**2)

    a = e*Q**2
    b = 2*e*(M*N - P*Q)
    c = e*(2*L*M + N**2 - P**2 - Q**2)
    d = 2*e*(N*L - P*Q)

    coefs = np.array([1, d, c, b, a])

    y = np.roots(coefs)

    for i in range(0, len(y)):
        if y[i].imag  == 0 and y[i].real > 0:
            y_sol = y[i].real

    th_1 = np.arctan(y_sol)
    TH_1[I] = th_1
    th_2 = th_1 + d_th

    K4 = np.sin(th_1)/np.sin(psi_1 - th_1)

    r2 = np.sin(psi_2)/np.sin(psi_2 - th_2)
    r3 = (np.sin(psi_1)/np.sin(psi_1 - th_1)) - np.sin(psi_2)/np.sin(psi_2 - th_2)
    r4 = np.sin(th_1)/np.sin(psi_1 - th_1)

    R1[I] = R4/K4
    R2[I] = R1[I]*(np.sin(psi_2)/np.sin(psi_2 - th_2))
    R3[I] = R1[I]*((np.sin(psi_1)/np.sin(psi_1 - th_1)) - np.sin(psi_2)/np.sin(psi_2 - th_2))

    C = 0.5*(1 + r2 + r4 + r3)
    mu_max[I] = abs(90 - np.rad2deg(2*np.arctan(np.sqrt(((C - r2 - r3)*(C - r4))/(C*(C - 1))))))
    mu_min[I] = abs(90 - np.rad2deg(2*np.arctan(np.sqrt(((C - r2)*(C - r4 -r3))/(C*(C - 1))))))
    X_C = R1[I] - R4*np.cos(np.pi - psi_1)
    Y_C = np.sin(np.pi - psi_1)*R4
    T_A[I] = f*(np.cos(alpha)*Y_C - np.sin(alpha)*X_C)
    T_D[I] = f*(np.cos(alpha)*Y_C + np.sin(alpha)*(R4*np.cos(np.pi - psi_1)))


globmin_mu_max = np.nanmin(mu_max)
index_mu_max = np.where(mu_max == globmin_mu_max)

d_th_minmu = d_th_arr[index_mu_max]
th_1_minmu = TH_1[index_mu_max]

T_A_minmu = T_A[index_mu_max]
T_D_minmu = T_D[index_mu_max]

R1_min = R1[index_mu_max]
R2_min = R2[index_mu_max]
R3_min = R3[index_mu_max]

mu_min_globmin = mu_min[index_mu_max]

plt.figure(1)
plt.title("Link Lengths (R4 = " +str(R4)+ " mm)")
plt.plot(np.rad2deg(d_th_arr), R1,label="R1")
plt.plot(np.rad2deg(d_th_arr), R2,label="R2")
plt.plot(np.rad2deg(d_th_arr), R3,label="R3")
plt.xlabel("Crank Rocker Travel (deg)")
plt.ylabel("Length (mm)")
plt.legend()
plt.grid()

plt.figure(2)
plt.title("Transmission Angle Deviation")
plt.plot(np.rad2deg(d_th_arr), mu_min, label="Min")
plt.plot(np.rad2deg(d_th_arr), mu_max,label="Max")
plt.xlabel("Crank Rocker Travel (deg)")
plt.ylabel("Angle (deg)")
plt.legend()
plt.grid()

plt.figure(3)
plt.title("Torque Requirements")
plt.plot(np.rad2deg(d_th_arr), T_A/1000, label="A")
plt.plot(np.rad2deg(d_th_arr), T_D/1000, label="D")
plt.xlabel("Crank Rocker Travel (deg)")
plt.ylabel("Torque (Nm)")
plt.legend()
plt.grid()

print("--------------------------------")
print("Change in Crank Angle = " +str(np.round(np.rad2deg(d_th_minmu[0]),2))+ " deg")
print("Torque about A = " +str(np.round(T_A_minmu[0]/1000,2))+ " Nm")
print("Torque about D = " +str(np.round(T_D_minmu[0]/1000,2))+ " Nm")
print("Link Lengths")
print("R1 = " +str(np.round(R1_min[0],2))+ " mm")
print("R2 = " +str(np.round(R2_min[0],2))+ " mm")
print("R3 = " +str(np.round(R3_min[0],2))+ " mm")
print("R4 = " +str(R4)+ " mm")