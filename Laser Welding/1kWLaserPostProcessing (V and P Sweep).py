import matplotlib.pyplot as plt
import numpy as np
import os

w_T_S = np.zeros((3,3))

w_T_S[0,0] = 1.619
w_T_S[1,0] = 1.079
w_T_S[2,0] = 0.808

w_T_S[0,1] = 1.767
w_T_S[1,1] = 1.229
w_T_S[2,1] = 0.938

w_T_S[0,2] = 1.859
w_T_S[1,2] = 1.319
w_T_S[2,2] = 0.87

w_T_E = np.zeros((3,3))

w_T_E[0,0] = 0.819
w_T_E[1,0] = 0.777
w_T_E[2,0] = 0.724

w_T_E[0,1] = 1.24
w_T_E[1,1] = 0.821
w_T_E[2,1] = 0.763

w_T_E[0,2] = 1.954
w_T_E[1,2] = 1.276
w_T_E[2,2] = 0.9065

d_p_S = np.zeros((3,3))

d_p_S[0,0] = 0.78
d_p_S[1,0] = 0.38
d_p_S[2,0] = 0.24

d_p_S[0,1] = 0.8
d_p_S[1,1] = 0.8
d_p_S[2,1] = 0.53

d_p_S[0,2] = 0.8
d_p_S[1,2] = 0.8
d_p_S[2,2] = 0.8

d_p_E = np.zeros((3,3))

d_p_E[0,0] = 0.8
d_p_E[1,0] = 0.549
d_p_E[2,0] = 0.469

d_p_E[0,1] = 0.8
d_p_E[1,1] = 0.8
d_p_E[2,1] = 0.686

d_p_E[0,2] = 0.8
d_p_E[1,2] = 0.8
d_p_E[2,2] = 0.8

w_B_E = np.zeros((3,2))

w_B_E[0,0] = 1.265
w_B_E[1,0] = 0.8
w_B_E[2,0] = 0

w_B_E[0,1] = 2.025
w_B_E[1,1] = 1.067
w_B_E[2,1] = 0.81

w_B_S = np.zeros((3,2))

w_B_S[0,0] = 1.38
w_B_S[1,0] = 0.54
w_B_S[2,0] = 0

w_B_S[0,1] = 1.62
w_B_S[1,1] = 0.96
w_B_S[2,1] = 0.57

v_arr = np.array([15, 30, 45])
P_arr = np.array([500, 750, 1000])

def polyLoBF(x, y, x_0, x_1, N, deg):
    Coefs = np.polyfit(x, y, deg)
    x_arr = np.linspace(x_0, x_1, N)
    y_arr = np.polyval(Coefs, x_arr)
    return y_arr

def error(x_E, x_S):
    error = abs(x_S - x_E)*100/x_E
    return error

N = 1000

LoBF_w_T_E = np.zeros((N, 3))
LoBF_w_T_S = np.zeros((N, 3))

LoBF_w_B_E = np.zeros((N,2))
LoBF_w_B_S = np.zeros((N,2))

LoBF_d_p_S = np.zeros((N, 3))
LoBF_d_p_E = np.zeros((N, 3))

err_w_T = np.zeros((N,3))
err_d_p = np.zeros((N,3))
err_w_B = np.zeros((N,2))

for i in range(0, 3):

    LoBF_w_T_E[:,i]  = polyLoBF(v_arr, w_T_E[:,i], 15, 45, N, 2)
    LoBF_w_T_S[:,i]  = polyLoBF(v_arr, w_T_S[:,i], 15, 45, N, 2)
    err_w_T[:,i] = error(LoBF_w_T_E[:,i], LoBF_w_T_S[:,i])

    LoBF_d_p_S[:,i]  = polyLoBF(v_arr, d_p_S[:,i], 15, 45, N, 2)
    LoBF_d_p_E[:,i]  = polyLoBF(v_arr, d_p_E[:,i], 15, 45, N, 2)
    err_d_p[:,i] = error(LoBF_d_p_E[:,i], LoBF_d_p_S[:,i])

i =0
for i in range(0, 2):
    LoBF_w_B_E[:,i]  = polyLoBF(v_arr, w_B_E[:,i], 15, 45, N, 2)
    LoBF_w_B_S[:,i]  = polyLoBF(v_arr, w_B_S[:,i], 15, 45, N, 2)
    err_w_B[:,i] = error(LoBF_w_B_E[:,i], LoBF_w_B_S[:,i])

v_linspace = np.linspace(15, 45, N)

plt.figure(1)
plt.plot(v_linspace, LoBF_w_T_E[:,0], label="500 W Test", color="tab:blue")
plt.plot(v_linspace, LoBF_w_T_E[:,1], label="750 W Test", color="tab:orange")
plt.plot(v_linspace, LoBF_w_T_E[:,2], label="1000 W Test", color="tab:red")

plt.plot(v_linspace, LoBF_w_T_S[:,0], label="500 W Sim", color="tab:blue", linestyle="dashed")
plt.plot(v_linspace, LoBF_w_T_S[:,1], label="750 W Sim", color="tab:orange", linestyle="dashed")
plt.plot(v_linspace, LoBF_w_T_S[:,2], label="1000 W Sim", color="tab:red", linestyle="dashed")

plt.legend()

plt.scatter(v_arr, w_T_E[:,0], color="tab:blue")
plt.scatter(v_arr, w_T_E[:,1], color="tab:orange")
plt.scatter(v_arr, w_T_E[:,2], color="tab:red")

plt.scatter(v_arr, w_T_S[:,0], color="tab:blue", marker="X")
plt.scatter(v_arr, w_T_S[:,1], color="tab:orange", marker="X")
plt.scatter(v_arr, w_T_S[:,2], color="tab:red", marker="X")

plt.xlabel("Welding Speed (mm/s)")
plt.ylabel("Top Melt Width (mm)")

plt.figure(2)
plt.plot(v_linspace, err_w_T[:,0], color="tab:blue", label="500W")
plt.plot(v_linspace, err_w_T[:,1], color="tab:orange", label="750W")
plt.plot(v_linspace, err_w_T[:,2], color="tab:red", label="1000W")
plt.xlabel("Welding Speed (mm/s)")
plt.ylabel("Top Melt Width Error (%)")
plt.legend()

plt.figure(3)
plt.plot(v_linspace, LoBF_d_p_E[:,0], color="tab:blue", label="Test")
plt.plot(v_linspace, LoBF_d_p_S[:,0], color="tab:blue", linestyle="dashed", label="Sim")
plt.legend()
plt.scatter(v_arr, d_p_E[:,0], color="tab:blue")
plt.scatter(v_arr, d_p_S[:,0], color="tab:blue", marker="X")
plt.xlabel("Welding Speed (mm/s)")
plt.ylabel("Penetration Depth (mm)")
plt.title("500 W")

plt.figure(4)
plt.plot(v_linspace, err_d_p[:,0])
plt.xlabel("Welding Speed (mm/s)")
plt.ylabel("Penetration Depth Error (%)")
plt.title("500 W")

plt.figure(5)
plt.plot(v_linspace, LoBF_w_B_E[:,0], label="750 W Test", color="tab:orange")
plt.plot(v_linspace, LoBF_w_B_E[:,1], label="1000 W Test", color="tab:red")

plt.plot(v_linspace, LoBF_w_B_S[:,0], label="750 W Sim", color="tab:orange", linestyle="dashed")
plt.plot(v_linspace, LoBF_w_B_S[:,1], label="1000 W Sim", color="tab:red", linestyle="dashed")

plt.legend()

plt.scatter(v_arr, w_B_E[:,0], color="tab:orange")
plt.scatter(v_arr, w_B_E[:,1], color="tab:red")
plt.scatter(v_arr, w_B_S[:,0], color="tab:orange", marker="X")
plt.scatter(v_arr, w_B_S[:,1], color="tab:red", marker="X")
plt.xlabel("Welding Speed (mm/s)")
plt.ylabel("Bottom Melt Width (mm)")

plt.figure(6)
plt.plot(v_linspace, err_w_B[:,0], color="tab:orange", label="750 W")
plt.plot(v_linspace, err_w_B[:,1], color="tab:red", label="1000 W")
plt.xlabel("Welding Speed (mm/s)")
plt.ylabel("Bottom Melt Width Error (%)")
