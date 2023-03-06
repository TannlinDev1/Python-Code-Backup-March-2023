import numpy as np
import matplotlib.pyplot as plt

E = 193e3 #Young's modulus (MPa)

r_0 = np.linspace(25,100,100) #Initial curvature
l = 25
I = (0.6**3)*1/12

# theta_L0 = l/r_0
#
# psi = np.zeros(100)
# K = np.zeros(100)
#
# for i in range(0,100):
#     psi[i] = theta_L0[i] + (theta_L0[i]/2)*np.cos(2*theta_L0[i]) - 0.75*np.sin(2*theta_L0[i])
#     K[i] = E*I/(2*psi[i]*r_0[i]**3)
#
# plt.figure(1)
# plt.plot(r_0, K)

k_0 = l/r_0

a_i = np.zeros(100)
b_i = np.zeros(100)
gamma = np.zeros(100)
rho = np.zeros(100)
K_th = np.zeros(100)
K = np.zeros(100)

for i in range(0,100):
    a_i[i] = l*(2/k_0[i])*np.sin(k_0[i]/2)
    b_i[i] = (l/k_0[i])*(1 - np.cos(k_0[i]/2))

    if k_0[i] > 0.5 and k_0[i] <0.595:
        gamma[i] = 0.8005 - 0.0173*k_0[i]
    else:
        gamma[i] = 0.8063 - 0.0265*k_0[i]

    rho[i] = np.sqrt((a_i[i]/l - (1 - gamma[i]))**2 + (2 * b_i[i]/l)**2)

    K_th[i] = 2.568 - 0.028*k_0[i] + 0.137* k_0[i]**2

    K[i] = 2 * rho[i] * K_th[i] * E * I / l

plt.figure(1)
plt.plot(r_0,K)
plt.xlabel("Curvarture")
plt.ylabel("Spring Constant (N/mm)")
plt.grid()


