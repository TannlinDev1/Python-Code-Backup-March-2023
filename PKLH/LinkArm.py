# Script for Link arm Dimensions

import numpy as np

dtheta = 20
theta_2 = (45-dtheta)/57.2958
d = 20
x_rest = d/(np.cos(theta_2)*np.sqrt(2)-1)
x_rest = round(x_rest,2)
l = np.sqrt(2)*x_rest
l = round(l,2)
print(l)
