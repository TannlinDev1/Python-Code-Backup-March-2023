#Supersonic Nozzle Generator
import numpy as np
import matplotlib.pyplot as plt
import os

P_0 = 1.8E6 #Inlet Pressure (Pa)
P_e = 101325 #Ambient Pressure (Pa)
D_C = 1.2 #Throat Diameter (mm)
M_C = 1 #Choked Mach No
gamma = 1.4 #Gas constant (c_p/c_v)
L_N = 7 #Length of Nozzle (mm)
R_0 = 4 #Inlet Radius (mm)

M_E = np.sqrt((2/(gamma-1))*((P_0/P_e)**((gamma-1)/gamma)-1))

Ratio_EtoC = np.sqrt((1/M_E)*((1+M_E**2*(gamma-1)/2)/(1+(gamma-1)/2))**((gamma+1)/(2*(gamma-1))))

D_E = Ratio_EtoC*D_C

M = np.linspace(M_C,M_E,num=100)
psi_A = 9.925*(np.pi/180)
tau_E = 1.643

psi_E = psi_A*2
K_1 = np.sqrt((gamma+1)/(gamma-1))
K_2 = np.sqrt((gamma-1)/(gamma+1))

def find_A_Te():
    Eqn_1 = np.zeros(len(M))
    Eqn_2 = np.zeros(len(M))
    A = np.zeros(len(M))
    
    for i in range(0,len(M)):

        A[i] = np.sqrt(M[i]**2-1)
        Eqn_1[i] = 0.5*(K_1*np.arctan(K_2*A[i])-np.arctan(A[i]))-psi_A
        Eqn_2[i] = np.sqrt((1/M[i])*((2/(gamma+1))+(K_2**2)*M[i]**2)**((gamma+1)/(2*(gamma-1))))

        if Eqn_1[i]>0:
            sol_A = A[i]
            sol_Te = Eqn_2[i]
            return[sol_A, sol_Te]
        
sol = find_A_Te()

A_start = sol[0]
A_end = np.sqrt(M_E**2-1)
tau_A = sol[1]

A = np.linspace(A_start,A_end,num=100)
x = np.zeros(len(A))
y = np.zeros(len(A))
M = np.zeros(len(A))
psi_P = np.zeros(len(A))

for i in range(0,len(A)):
    M[i] = np.sqrt(A[i]**2+1)
    psi_P[i] = 0.5*(K_1*np.arctan(K_2*A[i])-np.arctan(A[i]))
    theta_P = psi_E-psi_P[i]
    F = np.sqrt((np.sin(theta_P)**2)+2*(np.cos(theta_P)-np.cos(psi_A))*(A[i]*np.sin(theta_P)+np.cos(theta_P)))
    tau_P = np.sqrt((1/M[i])*((2/(gamma+1))+((gamma-1)/(gamma+1))*M[i]**2)**((gamma+1)/(2*(gamma-1))))
    y[i] = (D_E/(4*np.sin(psi_A/2)))*(tau_P/tau_E)*F
    x_1 = (D_E/(4*np.sin(psi_A/2)))*(tau_P/tau_E)*((1+F*(np.cos(theta_P)*A[i]-np.sin(theta_P)))/(np.sin(theta_P)*A[i]+np.cos(theta_P)))
    x_0 = (D_E/(2*tau_E))*((1/np.tan(psi_A))-(tau_A*np.cos(psi_A/2)-1)/(2*np.cos(psi_A/2)*(np.sin(psi_A/2)+np.cos(psi_A/2))))
    x[i] = x_1-x_0
    
R = (D_E/(4*tau_E*np.sin(psi_A/2)))*((tau_A*np.cos(psi_A/2))-1)/(np.cos(psi_A/2)+np.sin(psi_A/2))
x_D = R*np.sin(psi_A)
y_D = (D_E/2)+R*(1-np.cos(psi_A))
L = (D_E/(4*np.sin(psi_A/2)))+(D_E/2)*(np.sqrt(M_E**2-1))-x_0
x += (L_N-x[-1])
L_0 = L_N-L
x_c = np.linspace(0, L_0, num=100)
alpha = np.zeros(len(x_c))
radius_convergent = np.zeros(len(x_c))
R_C = D_C/2

for i in range(len(x_c)):
    alpha[i] = x_c[i]/L_0
    radius_convergent[i] = R_C/np.sqrt(1-(1-(R_C/R_0)**2)*((1-alpha[i]**2)**2/(1+alpha[i]**2/3)))
    

plot1 = plt.figure(1)
plt.plot(x,y,'b',label="Diverging")
plt.plot(x_c,radius_convergent,'r',label="Converging")
plt.xlabel("Axial (mm)")
plt.ylabel("Radial (mm)")
plt.grid()
plt.legend()
plt.show()

XY_div = np.zeros((len(x),2))
##XY_div[:,0] = np.linspace(1,len(x),num=len(x))
XY_conv = np.zeros((len(x_c),2))


XY_div[:,0] = np.round(x,4)
XY_div[:,1] = np.round(y,4)
#XY_div[:,0] = np.ones(len(x))
#XY_div[:,1] = np.linspace(1,100,num=len(x))

XY_conv[:,0] = np.round(x_c,4)
XY_conv[:,1] = np.round(radius_convergent,4)

#XY_conv[:,0] = np.ones(len(x))
#XY_conv[:,1] = np.linspace(1,100,num=len(x))

os.chdir(r"R:\Nozzle Analysis\Supersonic Nozzle\10 bar SS Nozzle")
np.savetxt("Divergent Section XY Points.txt", XY_div, delimiter=" ",fmt='%f')
np.savetxt("Convergent Section XY Points.txt", XY_conv, delimiter=" ",fmt='%f')

