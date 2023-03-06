import os
import matplotlib.pyplot as plt
import numpy as np

# This script imports raw sensor data, filters it to find 4 segments and performs some basic analysis

l_u = 77.25  # upper limit to search within
l_l = 73  # lower limit to search within

f = 0.02 #sampling frequency

path_10 = r"R:\TFF\Assembly Line\ABB\1200 Setup\R1\Results\090822 (v = 10 mmps)"
path_30 = r"R:\TFF\Assembly Line\ABB\1200 Setup\R1\Results\090822 (v = 30 mmps)"
path_60 = r"R:\TFF\Assembly Line\ABB\1200 Setup\R1\Results\090822 (v = 60 mmps)"
path_100 = r"R:\TFF\Assembly Line\ABB\1200 Setup\R1\Results\090822 (v = 100 mmps)"

def Z_stats(path_str, l_u, l_l):

    os.chdir(path_str)

    Z = np.loadtxt(fname="Z.txt")
    Z *= 4.967
    Z += 50.022

    Z_mean = np.mean(Z)
    Z_std = np.std(Z)

    for i in range(0, len(Z)):
        if Z[i] > l_u or Z[i] < l_l:
            Z[i] = np.nan #flag numerical values outside of limits with nan

    I = 0
    j_l = 0
    i_l = np.zeros(50)
    i_u = np.zeros(50)

    for I in range(0, 50): #finds start and end points of segments

        for j in range(j_l, len(Z)):

            if j < len(Z)-1:

                if np.isnan(Z[j]) == True and np.isnan(Z[j+1]) == False:
                    i_l[I] = j+1 #lower index

                if np.isnan(Z[j]) == False and np.isnan(Z[j+1]) == True:
                    i_u[I] = j#upper index
                    j_l = j+1
                    break

            else:
                break

    for J in range(0, len(i_l)): #flags out noisy "segments" which weren't caught by upper lower limits
        if i_u[J] - i_l[J] <= 5 or i_u[J] == 0:
            i_u[J] = None
            i_l[J] = None

    i_u = i_u[~np.isnan(i_u)] #deletes segments flagged by Nan
    i_l = i_l[~np.isnan(i_l)]

    i_u = i_u.astype(int) #change index array to integer type
    i_l = i_l.astype(int)
    N = np.zeros(4)

    for k in range(0, 4):

        N[k] = np.min([(i_u[k]-i_l[k]), (i_u[k+4]-i_l[k+4]), (i_u[k+8]-i_l[k+8])])

        if k == 0:
            t1 = np.linspace(0, (N[k]-2) * f, int(N[k]-2))
            Z1 = np.zeros((3, int(N[k]-2)))
            Z1[0, :] = Z[i_l[k]+2:int(i_l[k]+N[k])]
            Z1[1, :] = Z[i_l[k+4]+2:int(i_l[k+4]+N[k])]
            Z1[2, :] = Z[i_l[k+8]+2:int(i_l[k+8]+N[k])]

        if k == 1:
            t2 = np.linspace(0, N[k] * f, int(N[k]))
            Z2 = np.zeros((3, int(N[k])))
            Z2[0, :] = Z[i_l[k]:int(i_l[k]+N[k])]
            Z2[1, :] = Z[i_l[k+4]:int(i_l[k+4]+N[k])]
            Z2[2, :] = Z[i_l[k+8]:int(i_l[k+8]+N[k])]

        if k == 2:
            t3 = np.linspace(0, N[k] * f, int(N[k]))
            Z3 = np.zeros((3, int(N[k])))
            Z3[0, :] = Z[i_l[k]:int(i_l[k]+N[k])]
            Z3[1, :] = Z[i_l[k+4]:int(i_l[k+4]+N[k])]
            Z3[2, :] = Z[i_l[k+8]:int(i_l[k+8]+N[k])]

        if k == 3:
            t4 = np.linspace(0, N[k] * f, int(N[k]))
            Z4 = np.zeros((3, int(N[k])))
            Z4[0, :] = Z[i_l[k]:int(i_l[k]+N[k])]
            Z4[1, :] = Z[i_l[k+4]:int(i_l[k+4]+N[k])]
            Z4[2, :] = Z[i_l[k+8]:int(i_l[k+8]+N[k])]

    Zm = np.zeros((4, 3))
    Zstd = np.zeros((4, 3))
    Zr = np.zeros((4,3))

    for J in range(0, 3):

        Zm[0, J] = np.mean(Z1[J,:])
        Zm[1, J] = np.mean(Z2[J,:])
        Zm[2, J] = np.mean(Z3[J,:])
        Zm[3, J] = np.mean(Z4[J,:])
        # Zm[4, J] = np.mean(Z5[J,:])

        Zstd[0, J] = np.std(Z1[J,:])
        Zstd[1, J] = np.std(Z2[J,:])
        Zstd[2, J] = np.std(Z3[J,:])
        Zstd[3, J] = np.std(Z4[J,:])
        # Zstd[4, J] = np.std(Z5[J,:])

        Zr[0,J] = np.max(Z1[J,:]) - np.min(Z1[J,:])
        Zr[1,J] = np.max(Z2[J,:]) - np.min(Z2[J,:])
        Zr[2,J] = np.max(Z3[J,:]) - np.min(Z3[J,:])
        Zr[3,J] = np.max(Z4[J,:]) - np.min(Z4[J,:])

    Zr = np.mean(Zr, axis = 1)
    Zstd = np.mean(Zstd, axis = 1)

    return Zr, Zstd

def LoBF(x, y, deg, N):
    p = np.polyfit(x, y, deg)
    x_vector = np.linspace(np.min(x), np.max(x), N)
    y_LoBF = np.polyval(p, x_vector)

    # r-squared
    p = np.poly1d(p)
    # fit values, and mean
    yhat = p(x)                         # or [p(z) for z in x]
    ybar = np.sum(y)/len(y)          # or sum(y)/len(y)
    ssreg = np.sum((yhat-ybar)**2)   # or sum([ (yihat - ybar)**2 for yihat in yhat])
    sstot = np.sum((y - ybar)**2)    # or sum([ (yi - ybar)**2 for yi in y])
    r_sq = ssreg/sstot

    return y_LoBF, r_sq

Zr_10, Zstd_10 = Z_stats(path_10, l_u, l_l)
Zr_30, Zstd_30 = Z_stats(path_30, l_u, l_l)
Zr_60, Zstd_60 = Z_stats(path_60, l_u, l_l)
Zr_100, Zstd_100 = Z_stats(path_100, l_u, l_l)

v = np.array([10, 30, 60, 100])

Zr = np.zeros((4,4))
Zstd = np.zeros((4,4))

N = 100
v_linspace = np.linspace(10, 100, N)

Zr[0,:] = Zr_10
Zr[1,:] = Zr_30
Zr[2,:] = Zr_60
Zr[3,:] = Zr_100

Zstd[0,:] = Zstd_10
Zstd[1,:] = Zstd_30
Zstd[2,:] = Zstd_60
Zstd[3,:] = Zstd_100

Zr_LoBF = np.zeros((4, N))
r_sq_Zr = np.zeros(4)

Zstd_LoBF = np.zeros((4, N))
r_sq_Zstd = np.zeros(4)

for j in range(0, 4):
    Zr_LoBF[j,:], r_sq_Zr[j]  = LoBF(v, Zr[:,j], 1, N)
    Zstd_LoBF[j,:], r_sq_Zstd[j] = LoBF(v, Zstd[:,j], 1, N)


plt.figure(1)
plt.plot(v_linspace, Zr_LoBF[0,:], label="Segment 1, $r^2$ = "+str(np.round(r_sq_Zr[0],2))+"", color="tab:blue", linestyle="dashed")
plt.plot(v_linspace, Zr_LoBF[1,:], label="Segment 2 $r^2$ = "+str(np.round(r_sq_Zr[1],2))+"", color="tab:orange", linestyle="dashed")
plt.plot(v_linspace, Zr_LoBF[2,:], label="Segment 3, $r^2$ = "+str(np.round(r_sq_Zr[2],2))+"", color="tab:green", linestyle="dashed")
plt.plot(v_linspace, Zr_LoBF[3,:], label="Segment 4, $r^2$ = "+str(np.round(r_sq_Zr[3],2))+"", color="tab:red", linestyle="dashed")
plt.legend()
plt.scatter(v, Zr[:,0], color="tab:blue")
plt.scatter(v, Zr[:,1], color="tab:orange")
plt.scatter(v, Zr[:,2], color="tab:green")
plt.scatter(v, Zr[:,3], color="tab:red")
plt.legend()
plt.xlabel("Velocity (mm/s)")
plt.ylabel("Range (mm)")

plt.figure(2)
plt.plot(v_linspace, Zstd_LoBF[0,:], label="Segment 1, $r^2$ = "+str(np.round(r_sq_Zstd[0],2))+"", color="tab:blue", linestyle="dashed")
plt.plot(v_linspace, Zstd_LoBF[1,:], label="Segment 2 $r^2$ = "+str(np.round(r_sq_Zstd[1],2))+"", color="tab:orange", linestyle="dashed")
plt.plot(v_linspace, Zstd_LoBF[2,:], label="Segment 3, $r^2$ = "+str(np.round(r_sq_Zstd[2],2))+"", color="tab:green", linestyle="dashed")
plt.plot(v_linspace, Zstd_LoBF[3,:], label="Segment 4, $r^2$ = "+str(np.round(r_sq_Zstd[3],2))+"", color="tab:red", linestyle="dashed")
plt.legend()
plt.scatter(v, Zstd[:,0], color="tab:blue")
plt.scatter(v, Zstd[:,1], color="tab:orange")
plt.scatter(v, Zstd[:,2], color="tab:green")
plt.scatter(v, Zstd[:,3], color="tab:red")
plt.legend()
plt.xlabel("Velocity (mm/s)")
plt.ylabel("Standard Deviation (mm)")
