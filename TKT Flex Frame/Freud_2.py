import numpy as np
import matplotlib.pyplot as plt

phi_i = 30
phi_f = phi_i + 90

psi_i = 240
psi_f = 300

psi = np.zeros(3)
phi = np.zeros(3)

for i in range(0, len(phi)):
    phi[i] = 0.5*(phi_i + phi_f) - 0.5*(phi_f - phi_i)*np.cos((2*(i+1) - 1)*np.pi/6)
    psi[i] = 0.5*(psi_i + psi_f) - 0.5*(psi_f - psi_i)*np.cos((2*(i+1) - 1)*np.pi/6)


A = -1*np.ones((3,3))

A[0,0] = np.cos(np.deg2rad(phi[0]))
A[1,0] = np.cos(np.deg2rad(phi[1]))
A[2,0] = np.cos(np.deg2rad(phi[2]))
A[0,1] = np.cos(np.deg2rad(psi[0]))
A[1,1] = np.cos(np.deg2rad(psi[1]))
A[2,1] = np.cos(np.deg2rad(psi[2]))

b = np.zeros((3,1))

b[0,0] = np.cos(np.deg2rad(phi[0]) - np.deg2rad(psi[0]))
b[1,0] = np.cos(np.deg2rad(phi[1]) - np.deg2rad(psi[1]))
b[2,0] = np.cos(np.deg2rad(phi[2]) - np.deg2rad(psi[2]))

K = np.matmul(np.linalg.inv(A),b)

R1 = 150

R4 = R1/K[0]
R2 = R1/K[1]

R3 = np.sqrt(R1**2 + R2**2 + R4**2 - 2*R2*R4*K[2])

# R4[I] = R1/K[0]
# R2[I] = R1/K[1]
#
# R3[I] = np.sqrt(2*K[2]*R2[I]*R4[I] + R1**2 + R2[I]**2 + R4[I]**2)
#
#
# plt.figure(1)
# plt.plot(d_th4, R2, label="R2")
# plt.plot(d_th4, R3, label="R3")
# plt.plot(d_th4, R4, label="R4")
# plt.xlabel("Change in Theta 4 (deg)")
# plt.ylabel("Link Length (mm)")
# plt.grid()
# plt.legend()