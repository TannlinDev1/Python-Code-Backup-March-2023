import numpy as np
import os
import matplotlib.pyplot as plt
from scipy import interpolate

d = 12.45
#
# def stress_concentration(external_rad, d, H):
#     H_d_ratio = np.round(H/d,2)
#     r_d_ratio = external_rad/d
#     deg = 2
#
#
#     os.chdir(r"C:\Users\Tannlin User\Documents\Parallel Kinematic Laser Head Profile\Literature\Stress Concentration")
#
#     if H_d_ratio == 1.01 or H_d_ratio == 1:
#         matrix_101 = np.loadtxt(fname="1.01.txt", delimiter = ",")
#         f_101 = interpolate.interp1d(matrix_101[:,0],matrix_101[:,1])
#         K_t = f_101(r_d_ratio)
#
#     elif H_d_ratio > 1.01 and H_d_ratio <= 1.03:
#         matrix_102 = np.loadtxt(fname="1.02.txt", delimiter = ",")
#         poly_102 = np.polyfit(matrix_102[:,0],matrix_102[:,1],deg)
#         K_t = np.polyval(poly_102, r_d_ratio)
#
#     elif H_d_ratio >= 1.04 and H_d_ratio <=1.07:
#         matrix_103 = np.loadtxt(fname="1.03.txt", delimiter = ",")
#         poly_103 = np.polyfit(matrix_103[:,0],matrix_103[:,1],deg)
#         K_t = np.polyval(poly_103, r_d_ratio)
#
#     elif H_d_ratio >= 1.08 and H_d_ratio <1.35:
#         matrix_11 = np.loadtxt(fname="1.1.txt", delimiter = ",")
#         poly_11 = np.polyfit(matrix_11[:,0],matrix_11[:,1],deg)
#         K_t = np.polyval(poly_11, r_d_ratio)
#
#     elif H_d_ratio >= 1.35 and H_d_ratio <1.75:
#         matrix_15 = np.loadtxt(fname="1.5.txt", delimiter = ",")
#         poly_15 = np.polyfit(matrix_15[:,0],matrix_15[:,1],deg)
#         K_t = np.polyval(poly_15, r_d_ratio)
#
#     elif H_d_ratio >= 1.75 and H_d_ratio <2.5:
#         matrix_2 = np.loadtxt(fname="2.txt", delimiter = ",")
#         poly_2 = np.polyfit(matrix_2[:,0],matrix_2[:,1],deg)
#         K_t = np.polyval(poly_2, r_d_ratio)
#
#     elif H_d_ratio <= 3 and H_d_ratio >= 2.5:
#         matrix_3 = np.loadtxt(fname="3.txt", delimiter = ",")
#         poly_3 = np.polyfit(matrix_3[:,0],matrix_3[:,1],deg)
#         K_t = np.polyval(poly_3, r_d_ratio)
#
#     # else:
#     #     print("H/d Ratio is out of bounds")
#
#     return K_t, matrix_101
os.chdir(r"C:\Users\Tannlin User\Documents\Parallel Kinematic Laser Head Profile\Literature\Stress Concentration")

external_radius = 0.5
r_d_ratio = external_radius/d
matrix_101 = np.loadtxt(fname="1.01.txt", delimiter=",")
f_101 = interpolate.interp1d(matrix_101[:, 0], matrix_101[:, 1])
K_t = f_101(r_d_ratio)

# external_rad_array = np.linspace(0.01*d, 0.3*d, num = 100)
# H = 1.01*d
# Kt = np.zeros(100)
#
# for i in range(0,100):
#     external_rad = external_rad_array[i]
#     K_t, matrix_101 = stress_concentration(external_rad, d, H)
#     Kt[i] = K_t
#
# plt.plot(external_rad_array,Kt)
# plt.scatter(matrix_101[:,0]*d,matrix_101[:,1])
# plt.show()