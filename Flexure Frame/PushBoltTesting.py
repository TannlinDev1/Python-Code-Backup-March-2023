import numpy as np
import matplotlib.pyplot as plt
import os

os.chdir(r"C:\Users\Tannlin User\Documents\TKT Flexure Frame\Testing\Pushbolt Testing\2x 350um")

t350_1 = np.loadtxt(fname="2x350 - 1.txt")
t350_2 = np.loadtxt(fname="2x350 - 2.txt")
t350_3 = np.loadtxt(fname="2x350 - 3.txt")

t350_1[:,1] = t350_1[:,1]/25
t350_2[:,1] = t350_2[:,1]/25
t350_3[:,1] = t350_3[:,1]/25

poly_deg = 3
Coefs_t350_1 = np.polyfit(t350_1[:,1], t350_1[:,0], deg = poly_deg)
Coefs_t350_2 = np.polyfit(t350_2[:,1], t350_2[:,0], deg = poly_deg)
Coefs_t350_3 = np.polyfit(t350_3[:,1], t350_3[:,0], deg = poly_deg)

x_t350_1 = np.linspace(0, np.max(t350_1[:,1]), 100)
x_t350_2 = np.linspace(0, np.max(t350_2[:,1]), 100)
x_t350_3 = np.linspace(0, np.max(t350_3[:,1]), 100)

LoBF_t350_1 = np.polyval(Coefs_t350_1, x_t350_1)
LoBF_t350_2 = np.polyval(Coefs_t350_2, x_t350_2)
LoBF_t350_3 = np.polyval(Coefs_t350_3, x_t350_3)


os.chdir(r"C:\Users\Tannlin User\Documents\TKT Flexure Frame\Testing\Pushbolt Testing\2x 400um")

t400_1 = np.loadtxt(fname="2x 400 - 1.txt")
t400_2 = np.loadtxt(fname="2x 400 - 2.txt")
t400_3 = np.loadtxt(fname="2x 400 - 3.txt")

t400_1[:,1] = t400_1[:,1]/25
t400_2[:,1] = t400_2[:,1]/25
t400_3[:,1] = t400_3[:,1]/25

poly_deg = 3
Coefs_t400_1 = np.polyfit(t400_1[:,1], t400_1[:,0], deg = poly_deg)
Coefs_t400_2 = np.polyfit(t400_2[:,1], t400_2[:,0], deg = poly_deg)
Coefs_t400_3 = np.polyfit(t400_3[:,1], t400_3[:,0], deg = poly_deg)

x_t400_1 = np.linspace(0, np.max(t400_1[:,1]), 100)
x_t400_2 = np.linspace(0, np.max(t400_2[:,1]), 100)
x_t400_3 = np.linspace(0, np.max(t400_3[:,1]), 100)

LoBF_t400_1 = np.polyval(Coefs_t400_1, x_t400_1)
LoBF_t400_2 = np.polyval(Coefs_t400_2, x_t400_2)
LoBF_t400_3 = np.polyval(Coefs_t400_3, x_t400_3)

os.chdir(r"C:\Users\Tannlin User\Documents\TKT Flexure Frame\Testing\Pushbolt Testing\2x 450 um")

t450_1 = np.loadtxt(fname="2x450 - 1.txt")
t450_2 = np.loadtxt(fname="2x450 - 2.txt")
t450_3 = np.loadtxt(fname="2x450 - 3.txt")

t450_1[:,1] = t450_1[:,1]/25
t450_2[:,1] = t450_2[:,1]/25
t450_3[:,1] = t450_3[:,1]/25

poly_deg = 3
Coefs_t450_1 = np.polyfit(t450_1[:,1], t450_1[:,0], deg = poly_deg)
Coefs_t450_2 = np.polyfit(t450_2[:,1], t450_2[:,0], deg = poly_deg)
Coefs_t450_3 = np.polyfit(t450_3[:,1], t450_3[:,0], deg = poly_deg)

x_t450_1 = np.linspace(0, np.max(t450_1[:,1]), 100)
x_t450_2 = np.linspace(0, np.max(t450_2[:,1]), 100)
x_t450_3 = np.linspace(0, np.max(t450_3[:,1]), 100)

LoBF_t450_1 = np.polyval(Coefs_t450_1, x_t450_1)
LoBF_t450_2 = np.polyval(Coefs_t450_2, x_t450_2)
LoBF_t450_3 = np.polyval(Coefs_t450_3, x_t450_3)

os.chdir(r"C:\Users\Tannlin User\Documents\TKT Flexure Frame\Sim Results\Double 0.45mm Flexure")
t450_S = np.loadtxt(fname="MaxDisp5N.txt")

plt.figure(1)
plt.title("2 x 350 um SS304 Flexures")
# plt.scatter(t350_1[:,1], t350_1[:,0], color="r")
# plt.plot(x_t350_1, LoBF_t350_1, color="r")
plt.scatter(t350_2[:,1], t350_2[:,0], color="g")
plt.plot(x_t350_2, LoBF_t350_2, color="g")
plt.scatter(t350_3[:,1], t350_3[:,0], color="b")
plt.plot(x_t350_3, LoBF_t350_3, color="b")
plt.ylabel("Displacement (mm)")
plt.xlabel("Force (N)")
plt.grid()

plt.figure(2)
plt.title("2 x 400 um SS304 Flexures")
plt.scatter(t400_1[:,1], t400_1[:,0], color="r")
plt.plot(x_t400_1, LoBF_t400_1, color="r")
plt.scatter(t400_2[:,1], t400_2[:,0], color="g")
plt.plot(x_t400_2, LoBF_t400_2, color="g")
plt.scatter(t400_3[:,1], t400_3[:,0], color="b")
plt.plot(x_t400_3, LoBF_t400_3, color="b")
plt.ylabel("Displacement (mm)")
plt.xlabel("Force (N)")
plt.grid()

plt.figure(3)
plt.title("2 x 450 um SS304 Flexures")
plt.plot(x_t450_1, LoBF_t450_1, color="r",label="Sample 1")
plt.plot(x_t450_2, LoBF_t450_2, color="g",label="Sample 2")
plt.plot(x_t450_3, LoBF_t450_3, color="b",label="Sample 3")
plt.plot(t450_S[:,0], t450_S[:,1],color="m",label="Simulated")
plt.legend()
plt.scatter(t450_1[:,1], t450_1[:,0], color="r")
plt.scatter(t450_2[:,1], t450_2[:,0], color="g")
plt.scatter(t450_3[:,1], t450_3[:,0], color="b")
plt.ylabel("Displacement (mm)")
plt.xlabel("Force (N)")
plt.grid()

plt.show()