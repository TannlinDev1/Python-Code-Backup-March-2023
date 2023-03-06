import numpy as np
import matplotlib.pyplot as plt

E = 193000
endurance_limit = 215

# strain_limit = endurance_limit/E

strain_limit = 1.435e-3
pm_x = 20
facewidth = 15

# thickness_min = 0.05
# thickness_max = 0.3
# thickness_step = 0.05
# thickness_N =round(1+((thickness_max-thickness_min)/thickness_step))
# thickness = np.linspace(thickness_min,thickness_max,thickness_N)

thickness = 0.1

dtheta_min = 5
dtheta_max = 40
dtheta_step = 5
dtheta_N = 10*int((dtheta_max-dtheta_min)/dtheta_step + 1)
dtheta = np.linspace(dtheta_min,dtheta_max,dtheta_N)

bend_rad = np.zeros(dtheta_N)
bend_length = np.zeros(dtheta_N)
x_rest_pos = np.zeros(dtheta_N)
link_length = np.zeros(dtheta_N)
length_total = np.zeros(dtheta_N)

for i in range(0,dtheta_N):
    bend_rad[i] = thickness/(2*strain_limit)
    bend_length[i] = bend_rad[i]*(dtheta[i]/57.3)
    x_rest_pos[i] = pm_x/(np.cos((45-dtheta[i])/57.3)*np.sqrt(2)-1)
    link_length[i] = np.sqrt(2)*x_rest_pos[i]
    length_total[i] = 2*bend_length[i]+x_rest_pos[i]+facewidth*2.5

min_length_index = np.where(length_total == length_total.min())
min_length = min(length_total)
bend_length_opt = bend_length[min_length_index]
link_length_opt = link_length[min_length_index]
dtheta_opt = dtheta[min_length_index]
length_total_opt = length_total[min_length_index]

plt.figure(1)

plt.plot(dtheta,length_total)
plt.scatter(dtheta_opt,length_total_opt,color="r")
plt.grid()
plt.xlabel("Change in Angle (deg)")
plt.ylabel("Total Length (mm)")