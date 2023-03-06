import numpy as np
import matplotlib.pyplot as plt

# The following script determines kinematic properties of a four bar mechanism for a given swing angle and rocker angle

th = np.deg2rad(-1.16)

psi_0 = np.deg2rad(138.85)

th_f = th + np.pi
psi = np.deg2rad(1.15)

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

c = 30

d = c*p_3/np.sin(th_0)
b = c*d*(p_1+p_2)/2
a = c*d*(p_1-p_2)/2

# c = np.sin(th_0)/p_3
# a = c*(p_1 - p_2)/2
# b = c*(p_1 + p_2)/2

for i in range(0,len(a)):
    if a[i]>0 and b[i]>0 and d[i] >0:
        a_sol = a[i]
        b_sol = b[i]
        d_sol = d[i]
        th_0_sol = th_0[i]

# d_TA_min = np.arccos((b_sol**2 + c_sol**2 - (1 - a_sol)**2)/(2*b_sol*c_sol))
# d_TA_max= np.arccos((b_sol**2 + c_sol**2 - (1 + a_sol)**2)/(2*b_sol*c_sol))

print("--------------------------------")
print("Link Lengths")
print("a = " +str(np.round(a_sol,2))+ " mm")
print("b = " +str(np.round(b_sol,2))+ " mm")
print("c = " +str(np.round(c,2))+ " mm")
print("d = " +str(np.round(d_sol,2))+ " mm")