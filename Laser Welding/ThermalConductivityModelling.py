import numpy as np
import matplotlib.pyplot as plt

lambda_th_nom = 26

z = np.linspace(0, 0.8e-3, 1000)
K = np.zeros(1000)

for i in range(0, 1000):
    K[i] = 1/(13.3e3*z[i] + 0.667) + 1

plt.plot(z*1e3, K)