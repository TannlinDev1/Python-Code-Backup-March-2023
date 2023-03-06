import os
import numpy as np
import matplotlib.pyplot as plt

os.chdir(r"E:\Users\Tannlin User\Documents\TKT Flexure Frame\Bending Stiffness Sims\Fixed Frame\ForceVsDisplacement")

DoubleFlex_06_Ideal = np.loadtxt(fname="TXvsF_Ideal.txt")
DoubleFlex_06_Ideal[:,0] = DoubleFlex_06_Ideal[:,0]*6

DoubleFlex_06_Fixed = np.loadtxt(fname="TXvsF_Fixed.txt")
DoubleFlex_06_Fixed[:,0] = DoubleFlex_06_Fixed[:,0]*6

DoubleFlex_06_Sep = np.loadtxt(fname="TXvsF_Sep.txt")
DoubleFlex_06_Sep[:,0] = DoubleFlex_06_Sep[:,0]*6

plt.figure(1)
plt.plot(DoubleFlex_06_Fixed[:,0], DoubleFlex_06_Fixed[:,1], color="r",label="Fixed")
plt.plot(DoubleFlex_06_Sep[:,0],DoubleFlex_06_Sep[:,1],color="b",label="Separation")
plt.plot(DoubleFlex_06_Ideal[:,0], DoubleFlex_06_Ideal[:,1], color="g", label="Idealised")
plt.grid()
plt.legend()
plt.ylabel("Displacement (mm)")
plt.xlabel("Force (N)")
plt.title("2x 0.6 mm SS304 Flexures")