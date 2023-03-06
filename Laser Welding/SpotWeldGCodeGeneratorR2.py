import os
import numpy as np
import matplotlib.pyplot as plt

# This script generates Aerotech code for a "W" stitch pattern

x_0 = -276.5 # starting x position
x_f = 276.5 #final x position

y_0 = -3.9# centreline of stitch pattern

width = 15 #width of filtered sections

x_step = 1.5 # x distance between spots
dy = 0.125
#y distance of spot from centreline

N = int(1 + abs(x_f - x_0)/x_step)

points = np.zeros((N,2)) #memory preallocation
filter_points = np.zeros((N,2))

x_lim = np.array([-244.15, -125, -13.4, 125, 244.15]) #x limits for filter (centrelines)

for i in range(0, N):

    points[i,0] = x_0 + x_step * i #x iteration
    points[i,1] = y_0 + np.real(1j ** (2*i)) * dy #y iteration (uses imaginary unit)

    for J in range(0, len(x_lim)): #apply filter

        if np.sign(x_lim[J]) == -1.0: #negative filter
            if points[i,0] >= x_lim[J] - width/2 and points[i,0] <= x_lim[J] + width/2:
                filter_points[i, 0] = points[i, 0]
                filter_points[i, 1] = points[i, 1]
                points[i, 0] = np.NaN
                points[i, 1] = np.NaN

        else: #positive filter
            if points[i,0] <= x_lim[J] + width/2 and points[i,0] >= x_lim[J] - width/2:
                filter_points[i, 0] = points[i, 0]
                filter_points[i, 1] = points[i, 1]
                points[i, 0] = np.NaN
                points[i, 1] = np.NaN


points = points[~np.isnan(points).any(axis=1)] #delete rows flagged by NaN
filter_points = filter_points[~np.all(filter_points == 0, axis = 1)] #delete zero rows

def GCode_Gen(points, N): #define G code generator function

    code = np.zeros((N, 1))
    code = code.astype("str")  # change array type to string

    J = 0
    I = 0

    while I<N: #write G Code

        code[I, 0] = "G01 " + "X " + str(np.round(points[J,0],2)) + " Y " + str(np.round(points[J,1],2))
        code[I+1,0] = "CALL WELD"

        I += 2
        J += 1

    return code

N = np.size(points[:,0])*2 #size of matrix for generating G-code
N_F = np.size(filter_points[:,0])*2

code = GCode_Gen(points, N) #generate non-filter code
filter_code = GCode_Gen(filter_points, N_F) #generate filter code

plt.figure(1) #plot points
plt.scatter(points[:,0], points[:,1], color="g", label="First Pass")
plt.scatter(filter_points[:,0], filter_points[:,1], color="r" , label="Second Pass")
plt.legend()
plt.title("Weld Pattern")
plt.xlabel("X Position (mm)")
plt.ylabel("Y Position (mm)")

os.chdir(r"C:\Users\angus.mcallister\Documents\welds") #write G code to directory
np.savetxt("G_CODE.txt",code, delimiter=" ", fmt='%s')
np.savetxt("G_CODE (Filtered).txt", filter_code, delimiter=" ", fmt="%s")