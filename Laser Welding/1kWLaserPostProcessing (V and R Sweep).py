import matplotlib.pyplot as plt
import numpy as np
import os

w_T_S = np.zeros((2,3))

w_T_S[0,0] = 1.175
w_T_S[1,0] = 0.641

w_T_S[0,1] = 1.235
w_T_S[1,1] = 0.638

w_T_S[0,2] = 1.263
w_T_S[1,2] = 0.682

w_T_E = np.zeros((2,3))

w_T_E[0,0] = 0.964
w_T_E[1,0] = 0.661

w_T_E[0,1] = 0.836
w_T_E[1,1] = 0.77

w_T_E[0,2] = 0.991
w_T_E[1,2] = 0.761

d_p_S = np.zeros((2,3))

d_p_S[0,0] = 0.8
d_p_S[1,0] = 0.8

d_p_S[0,1] = 0.8
d_p_S[1,1] = 0.42

d_p_S[0,2] = 0.71
d_p_S[1,2] = 0.26

d_p_E = np.zeros((2,3))

d_p_E[0,0] = 0.8
d_p_E[1,0] = 0.8

d_p_E[0,1] = 0.8
d_p_E[1,1] = 0.692

d_p_E[0,2] = 0.681
d_p_E[1,2] = 0.527

w_B_E = np.zeros((2,3))

w_B_E[0,0] = 0.809
w_B_E[1,0] = 0.455

w_B_E[0,1] = 0.613

w_B_S = np.zeros((3,2))

w_B_S[0,0] = 0.89
w_B_S[1,0] = 0.35

w_B_S[0,1] = 0.73

rf_arr = np.array([0.3, 0.37, 0.43])

def polyLoBF(x, y, x_0, x_1, N, deg):
    Coefs = np.polyfit(x, y, deg)
    x_arr = np.linspace(x_0, x_1, N)
    y_arr = np.polyval(Coefs, x_arr)
    return y_arr

def error(x_E, x_S):
    error = abs(x_S - x_E)*100/x_E
    return error

N = 1000

LoBF_w_T_E = np.zeros((N, 2))
LoBF_w_T_S = np.zeros((N, 2))

# LoBF_w_B_E = np.zeros((N,2))
# LoBF_w_B_S = np.zeros((N,2))

LoBF_d_p_S = np.zeros((N, 2))
LoBF_d_p_E = np.zeros((N, 2))

for i in range(0, 2):

    LoBF_w_T_E[:,i]  = polyLoBF(rf_arr, w_T_E[i,:], 0.3, 0.432, N, 2)
    LoBF_w_T_S[:,i]  = polyLoBF(rf_arr, w_T_S[i,:], 0.3, 0.432, N, 2)

    LoBF_d_p_S[:,i]  = polyLoBF(rf_arr, d_p_S[i,:], 0.3, 0.432, N, 2)
    LoBF_d_p_E[:,i]  = polyLoBF(rf_arr, d_p_E[i,:], 0.3, 0.432, N, 2)

i =0

# for i in range(0, 2):
#     LoBF_w_B_E[:,i]  = polyLoBF(v_arr, w_B_E[:,i], 15, 45, N, 2)
#     LoBF_w_B_S[:,i]  = polyLoBF(v_arr, w_B_S[:,i], 15, 45, N, 2)

rf_linspace = np.linspace(0.3, 0.43, N)

plt.figure(1)
plt.plot(rf_linspace, LoBF_w_T_E[:,0], label="30 mm/s Test", color="tab:blue")
plt.plot(rf_linspace, LoBF_w_T_E[:,1], label="60 mm/s Test", color="tab:orange")

plt.plot(rf_linspace, LoBF_w_T_S[:,0], label="30 mm/s Sim", color="tab:blue", linestyle="dashed")
plt.plot(rf_linspace, LoBF_w_T_S[:,1], label="60 mm/s Sim", color="tab:orange", linestyle="dashed")

plt.legend()

plt.scatter(rf_arr, w_T_E[0,:], color="tab:blue")
plt.scatter(rf_arr, w_T_E[1,:], color="tab:orange")

plt.scatter(rf_arr, w_T_S[0,:], color="tab:blue", marker="X")
plt.scatter(rf_arr, w_T_S[1,:], color="tab:orange", marker="X")

plt.xlabel("Spot Radius (mm)")
plt.ylabel("Top Melt Width (mm)")

plt.figure(3)
plt.plot(rf_linspace, LoBF_d_p_E[:,0], color="tab:blue", label="30 mm/s Test")
plt.plot(rf_linspace, LoBF_d_p_E[:,1], color="tab:orange", label="60 mm/s Test")

plt.plot(rf_linspace, LoBF_d_p_S[:,0], color="tab:blue", linestyle="dashed", label="30 mm/s Sim")
plt.plot(rf_linspace, LoBF_d_p_S[:,1], color="tab:orange", linestyle="dashed", label="60 mm/s Sim")

plt.legend()

plt.scatter(rf_arr, d_p_E[0,:], color="tab:blue")
plt.scatter(rf_arr, d_p_E[1,:], color="tab:orange")

plt.scatter(rf_arr, d_p_S[0,:], color="tab:blue", marker="X")
plt.scatter(rf_arr, d_p_S[1,:], color="tab:orange", marker="X")

plt.xlabel("spot Radius (mm)")
plt.ylabel("Penetration Depth (mm)")
