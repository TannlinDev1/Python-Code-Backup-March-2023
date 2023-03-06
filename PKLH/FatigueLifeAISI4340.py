import matplotlib.pyplot as plt
import numpy as np
import os

os.chdir(r"C:\Users\Tannlin User\Documents\Parallel Kinematic Laser Head Profile\Simulations\SN Curve Example")

elastic = np.loadtxt(fname="Elastic.txt",delimiter=",")
plastic = np.loadtxt(fname="Plastic.txt",delimiter=",")
combined = np.loadtxt(fname="Combined.txt",delimiter=",")

plt.figure(1)
plt.loglog(elastic[:,0],elastic[:,1],color="r",label="Elastic")
plt.loglog(plastic[:,0],plastic[:,1],color="g",label="Plastic")
plt.loglog(combined[:,0],combined[:,1],color="b",label="Combined")
plt.grid()
plt.legend()
plt.xlabel("Number of Reversals")
plt.ylabel("Total Strain Amplitude")