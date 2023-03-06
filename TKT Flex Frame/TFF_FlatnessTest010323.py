import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import scipy as sp
from scipy.linalg import lstsq
from itertools import chain

# Takes in measured dz data points from TFF frame, corrects for jig realignment and gives interactive graph
# see https://math.stackexchange.com/questions/99299/best-fitting-plane-given-a-set-of-points for more info
os.chdir(r"R:\TFF\Testing Results\TFF Frame Flatness Tests")
dz = np.loadtxt(fname="Results.txt")#import data

N = np.size(dz, 0)
dx = 50
dy = 50

xs = np.linspace(0, dx*(N-1), N)
ys = np.linspace(0, dy*(N-1), N)
dz = list(chain.from_iterable(dz))#turns 6x6 array into 36x1 list

x = np.zeros(N**2)
y = np.zeros(N**2)

for I in range(0, N):
    x[I*N:(I+1)*N] = xs[I]
    y[I*N:(I+1)*N] = ys

x = list(x)
y = list(y) #generates x and y data in list format

tmp_A = []
tmp_b = []
for i in range(len(x)):
    tmp_A.append([x[i], y[i], 1])
    tmp_b.append(dz[i])

b = np.matrix(tmp_b).T
A = np.matrix(tmp_A)

# fit = (A.T * A).I * A.T * b
# errors = b - A * fit
# residual = np.linalg.norm(errors) #uses least squares solution to fit plane of best fit (z = ax + by + c)

fit, residual, rnk, s = lstsq(A, b)
errors = b - A * fit

print("solution: %f x + %f y + %f = z" % (fit[0], fit[1], fit[2]))

errors = np.array(errors)
z_err = np.zeros((N,N))

for J in range(0, N):
    z_err[J,:] = errors[J*N:(J+1)*N,0]

# plot plane

X,Y = np.meshgrid(xs, ys)

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

surf = ax.plot_surface(X, Y, z_err, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
# Customize the z axis.
# ax.set_zlim(-, 1.01)
# ax.zaxis.set_major_locator(LinearLocator(10))
# # A StrMethodFormatter is used automatically
# ax.zaxis.set_major_formatter('{x:.02f}')

ax.set_xlabel('X (mm)')
ax.set_ylabel('Y (mm)')
ax.set_zlabel('Z ($\mu$m)')

plt.show()