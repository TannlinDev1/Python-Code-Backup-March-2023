import numpy as np
import matplotlib.pyplot as plt

# The following script determines kinematic properties of a four bar mechanism for a given swing angle and rocker angle

th_0 = np.deg2rad(142.22)
th = np.deg2rad(0)

psi = np.deg2rad(90)

th_f = th + np.pi

y = psi - th - th_0

n_1 = np.sin(th_0)
n_2 = np.cos(th_0)
n_3 = np.sin(th + th_0)
n_4 = np.sin(y)
n_5 = np.cos(y)

psi_0 = np.arctan((n_1*(n_3 + n_4))/(n_2*n_3 - n_1*n_5))

p_1 = np.sin(psi_0)/np.sin(th_0)
p_2 = np.sin(psi + psi_0)/np.sin(th + th_0)
p_3 = np.sin(psi_0 - th_0)

c = np.sin(th_0)/p_3
a = c*(p_1 - p_2)/2
b = c*(p_1 + p_2)/2

a *= 150
b *= 150
c *= 150

d_TA_min = np.arccos((b**2 + c**2 - (1 - a)**2)/(2*b*c))
d_TA_max= np.arccos((b**2 + c**2 - (1 + a)**2)/(2*b*c))

