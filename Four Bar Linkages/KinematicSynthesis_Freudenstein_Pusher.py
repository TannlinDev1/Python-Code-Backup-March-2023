import numpy as np
import matplotlib.pyplot as plt

th_2i = 10 # >180 deg
th_2f = th_2i + 85

th_4i = 285
th_4f = th_4i + 15

R4 = 80

th4 = np.zeros(3)
th2 = np.zeros(3)

for i in range(0, len(th2)):
    th2[i] = 0.5*(th_2i + th_2f) - 0.5*(th_2f - th_2i)*np.cos((2*(i+1) - 1)*np.pi/6)
    th4[i] = 0.5*(th_4i + th_4f) - 0.5*(th_4f - th_4i)*np.cos((2*(i+1) - 1)*np.pi/6)

A = np.ones((3,3))

A[0,0] = np.cos(np.deg2rad(th2[0]))
A[1,0] = np.cos(np.deg2rad(th2[1]))
A[2,0] = np.cos(np.deg2rad(th2[2]))
A[0,1] = np.cos(np.deg2rad(th4[0]))
A[1,1] = np.cos(np.deg2rad(th4[1]))
A[2,1] = np.cos(np.deg2rad(th4[2]))

b = np.zeros((3,1))

b[0,0] = np.cos(np.deg2rad(th2[0]) - np.deg2rad(th4[0]))
b[1,0] = np.cos(np.deg2rad(th2[1]) - np.deg2rad(th4[1]))
b[2,0] = np.cos(np.deg2rad(th2[2]) - np.deg2rad(th4[2]))

K = np.matmul(np.linalg.inv(A),b)

R1 = R4*K[0]
R2 = R1/K[1]

R3 = np.sqrt(2*K[2]*R2*R4 + R1**2 + R2**2 + R4**2)

mu = np.arccos((R3**2 + R4**2 - (R2**2 + R1**2) + 2 * R2 * R1 * np.cos(np.deg2rad(th_2i)))/(2*R3*R4))

R1_str = str(R1)
R2_str = str(round(R2[0],2))
R3_str = str(round(R3[0],2))
R4_str = str(round(R4,2))

mu_str = str(round(np.rad2deg(mu[0]),2))

print("--------------------------------")
print("Link Lengths")
print("R1 = " +R1_str+ " mm")
print("R2 = " +R2_str+ " mm")
print("R3 = " +R3_str+ " mm")
print("R4 = " +R4_str+ " mm")

print("Transmission Angle = " +mu_str+ "deg")