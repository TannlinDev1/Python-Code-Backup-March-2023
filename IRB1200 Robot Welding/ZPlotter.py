import os
import matplotlib.pyplot as plt
import numpy as np

# This script imports raw sensor data, filters it to find 4 segments and performs some basic analysis

os.chdir(r"R:\TFF\Assembly Line\ABB\1200 Setup\R1\Results\090822 (v = 100 mmps)")

f = 0.02 #sampling frequency

Z = np.loadtxt(fname="Z.txt")
Z *= 4.967
Z += 50.022

Z_mean = np.mean(Z)
Z_std = np.std(Z)

t = np.linspace(0, len(Z)*f, len(Z))

l_u = 76.5 #upper limit to search within
l_l = 74.5 #lower limit to search within

plt.figure(8)

plt.plot(t, Z)
plt.hlines(l_l, t[0], t[-1], linestyles="dashed", label="Lower Limit", color = "tab:orange")
plt.hlines(l_u, t[0], t[-1], linestyles="dashed", label="Upper Limit", color="tab:green")
plt.xlabel("Time (s)")
plt.ylabel("Normal Distance (mm)")
plt.title("Raw Data")
plt.legend()

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
    #
    # if k == 4:
    #     t5 = np.linspace(0, N[k] * f, int(N[k]))
    #     Z5 = np.zeros((3, int(N[k])))
    #     Z5[0, :] = Z[i_l[k]:int(i_l[k]+N[k])]
    #     Z5[1, :] = Z[i_l[k+5]:int(i_l[k+5]+N[k])]
    #     Z5[2, :] = Z[i_l[k+10]:int(i_l[k+10]+N[k])]

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

segments = np.linspace(1, 4, 4)

plt.figure(1)

for i in range(0, 3):
    plt.subplot(2,2,1)
    plt.plot(t1, Z1[i,:])
    plt.title("Segment 1")

    plt.subplot(2,2,2)
    plt.plot(t2, Z2[i,:])
    plt.title("Segment 2")

    plt.subplot(2,2,3)
    plt.plot(t3, Z3[i,:])
    plt.title("Segment 3")

    plt.subplot(2,2,4)
    plt.plot(t4, Z4[i, :])
    plt.title("Segment 4")


# plt.xlabel("Time (s)")
# plt.ylabel("Normal Distance (mm)")
# plt.title("Segment 1")

#
# plt.figure(2)
#
# for i in range(0, 3):
#
#
# plt.xlabel("Time (s)")
# plt.ylabel("Normal Distance (mm)")
# plt.title("Segment 2")
#
# # plt.figure(3)
#
# for i in range(0, 3):
#
#
# plt.xlabel("Time (s)")
# plt.ylabel("Normal Distance (mm)")
# plt.title("Segment 3")
#
# # plt.figure(4)
#
# for i in range(0, 3):
#
# plt.xlabel("Time (s)")
# plt.ylabel("Normal Distance (mm)")
# plt.title("Segment 4")

# plt.figure(5)
#
# for i in range(0, 3):
#     plt.plot(t5, Z5[i,:])
#
# plt.xlabel("Time (s)")
# plt.ylabel("Normal Distance (mm)")
# plt.title("Segment 5")

plt.figure(6)

plt.plot(segments, Zm[:,0], label="Test 1")
plt.plot(segments, Zm[:,1], label="Test 2")
plt.plot(segments, Zm[:,2], label="Test 3")
plt.xlabel("Segment No")
plt.ylabel("Mean Normal Distance (mm)")

plt.figure(7)

plt.plot(segments, Zstd[:,0], label="Test 1")
plt.plot(segments, Zstd[:,1], label="Test 2")
plt.plot(segments, Zstd[:,2], label="Test 3")
plt.xlabel("Segment No")
plt.ylabel("Standard Deviation")

plt.figure(9)

plt.plot(segments, Zr[:,0], label="Test 1")
plt.plot(segments, Zr[:,1], label="Test 2")
plt.plot(segments, Zr[:,2], label="Test 3")
plt.xlabel("Segment No")
plt.ylabel("Range (mm)")