import numpy as np

psi = np.deg2rad(15)
mu_min = np.deg2rad(80)

K1 = np.sqrt((1 - np.cos(psi))/(2*np.cos(mu_min)**2))
K2 = np.sqrt((1 - K1**2)/(1 - K1**2 * np.cos(mu_min)**2))
K3 = np.sqrt(K1**2 + K2**2 - 1)

A = 20
D = A/K3
B = D*K1
C = D*K2