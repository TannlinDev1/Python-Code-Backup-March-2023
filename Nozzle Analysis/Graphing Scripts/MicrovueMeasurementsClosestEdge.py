import matplotlib.pyplot as plt
import os
import numpy as np

os.chdir("C:/Users/Tannlin User/Documents/Tannlin Internship 2020/Side Profiles/MicroVue Measurements")
os.getcwd()

step = 0.1
step1 = 0.05

XY_P18_H50 = np.zeros((3,61))
XY_P18_H100 = np.zeros((3,61))
XY_P18_H150 = np.zeros((3,61))
XY_P18_H200 = np.zeros((3,61))

XY_P18_H50[0,:] = np.loadtxt(fname = "XY_P18_H50_1.txt",delimiter=',')
XY_P18_H50[1,:] = np.loadtxt(fname = "XY_P18_H50_2.txt",delimiter=',')
XY_P18_H50[2,:] = np.loadtxt(fname = "XY_P18_H50_3.txt",delimiter=',')

XY_P18_H100[0,:] = np.loadtxt(fname = "XY_P18_H100_1.txt",delimiter=',')
XY_P18_H100[1,:] = np.loadtxt(fname = "XY_P18_H100_2.txt",delimiter=',')
XY_P18_H100[2,:] = np.loadtxt(fname = "XY_P18_H100_3.txt",delimiter=',')

XY_P18_H150[0,:] = np.loadtxt(fname = "XY_P18_H150_1.txt",delimiter=',')
XY_P18_H150[1,:] = np.loadtxt(fname = "XY_P18_H150_2.txt",delimiter=',')
XY_P18_H150[2,:] = np.loadtxt(fname = "XY_P18_H150_3.txt",delimiter=',')

XY_P18_H200[0,:] = np.loadtxt(fname = "XY_P18_H200_1.txt",delimiter=',')
XY_P18_H200[1,:] = np.loadtxt(fname = "XY_P18_H200_2.txt",delimiter=',')
XY_P18_H200[2,:] = np.loadtxt(fname = "XY_P18_H200_3.txt",delimiter=',')
XY_P18_H200_new = np.loadtxt(fname = "XY_P18_H200_4.txt",delimiter=',')

#tol = 0.1

#for i in range(0,3):
    #XY_P18_H50[i,:] = XY_P18_H50[i,:][ (XY_P18_H50[i,:]<(0.2+tol)) & (XY_P18_H50[i,:]>(0.2-tol))]

Y = np.linspace(0, step*len(XY_P18_H50[0,:]), num = len(XY_P18_H50[0,:]))
Y1 = np.linspace(0, step1*len(XY_P18_H200_new), num = len(XY_P18_H200_new))

P_50 = np.zeros((3,4))
P_100 = np.zeros((3,4))
P_150 = np.zeros((3,4))
P_200 = np.zeros((3,4))

BF_P18_H50 = np.zeros((3,len(XY_P18_H50[0,:])))
XYN_P18_H50 = np.zeros((3,len(XY_P18_H50[0,:])))

BF_P18_H100 = np.zeros((3,len(XY_P18_H100[0,:])))
XYN_P18_H100 = np.zeros((3,len(XY_P18_H100[0,:])))

BF_P18_H150 = np.zeros((3,len(XY_P18_H150[0,:])))
XYN_P18_H150 = np.zeros((3,len(XY_P18_H150[0,:])))

BF_P18_H200 = np.zeros((3,len(XY_P18_H200[0,:])))
XYN_P18_H200 = np.zeros((3,len(XY_P18_H200[0,:])))

for i in range(0,3):

    P_50[i,:] = np.polyfit(Y,XY_P18_H50[i,:], 3)
    BF_P18_H50[i,:] = np.polyval(P_50[i,:], Y)
    XYN_P18_H50[i,:] = 1000*(XY_P18_H50[i,:] - 0.2)
    
    P_100[i,:] = np.polyfit(Y,XY_P18_H100[i,:], 3)
    BF_P18_H100[i,:] = np.polyval(P_100[i,:], Y)
    XYN_P18_H100[i,:] = 1000*(XY_P18_H100[i,:] - 0.2)
    
    P_150[i,:] = np.polyfit(Y,XY_P18_H150[i,:], 3)
    BF_P18_H150[i,:] = np.polyval(P_150[i,:], Y)
    XYN_P18_H150[i,:] = 1000*(XY_P18_H150[i,:] - 0.2)
    
    P_200[i,:] = np.polyfit(Y,XY_P18_H200[i,:], 3)
    BF_P18_H200[i,:] = np.polyval(P_200[i,:], Y)
    XYN_P18_H200[i,:] = 1000*(XY_P18_H200[i,:] - 0.2)

P_200_new = np.polyfit(Y1,XY_P18_H200_new, 3)
BF_P18_H200_new = np.polyval(P_200_new, Y1)
XYN_P18_H200 = 1000*(XY_P18_H200_new - 0.2)

XY_P18 = np.zeros((4,len(XY_P18_H50[0,:])))

for i in range(0,len(BF_P18_H50[i,:])):
    XY_P18[0,i] = (BF_P18_H50[0,i]+BF_P18_H50[1,i]+BF_P18_H50[2,i])/3
    XY_P18[1,i] = (BF_P18_H100[0,i]+BF_P18_H100[1,i]+BF_P18_H100[2,i])/3
    XY_P18[2,i] = (BF_P18_H150[0,i]+BF_P18_H150[1,i]+BF_P18_H150[2,i])/3
    XY_P18[3,i] = (BF_P18_H200[0,i]+BF_P18_H200[1,i]+BF_P18_H200[2,i])/3

Ra_P18 = np.zeros((4,3))

for i in range(0,3):
    Ra_P18[0,i] = np.mean(XYN_P18_H50[i,:])
    Ra_P18[1,i] = np.mean(XYN_P18_H100[i,:])
    Ra_P18[2,i] = np.mean(XYN_P18_H150[i,:])
   # Ra_P18[3,i] = np.mean(XYN_P18_H200[i,:])

Ra_ave_P18 = np.zeros(4)

Ra_ave_P18[0] = np.mean(Ra_P18[0,:])
Ra_ave_P18[1] = np.mean(Ra_P18[1,:])
Ra_ave_P18[2] = np.mean(Ra_P18[2,:])
Ra_ave_P18[3] = np.mean(Ra_P18[3,:])

Z = [50, 100, 150, 200]

plot1 = plt.figure(1)
plt.scatter(Y,XY_P18_H50[0,:], color='r',label='1')
plt.scatter(Y,XY_P18_H50[1,:], color='g',label='2')
plt.scatter(Y,XY_P18_H50[2,:], color='b',label='3')
plt.plot(Y,BF_P18_H50[0,:],'r')
plt.plot(Y,BF_P18_H50[1,:],'g')
plt.plot(Y,BF_P18_H50[2,:],'b')
plt.plot(Y,XY_P18[0,:])
plt.legend()
plt.xlabel("Cut Length (mm)")
plt.ylabel("Surface Roughness (mm)")
plt.title("Underside Surface Roughness P = 18 bar, H = 50 um")
plt.grid()

plot2 = plt.figure(2)
plt.scatter(Y,XY_P18_H100[0,:], color='r',label='1')
plt.scatter(Y,XY_P18_H100[1,:], color='g',label='2')
plt.scatter(Y,XY_P18_H100[2,:], color='b',label='3')
plt.plot(Y,BF_P18_H100[0,:],'r')
plt.plot(Y,BF_P18_H100[1,:],'g')
plt.plot(Y,BF_P18_H100[2,:],'b')
plt.plot(Y,XY_P18[1,:])
plt.legend()
plt.xlabel("Cut Length (mm)")
plt.ylabel("Surface Roughness (mm)")
plt.title("Underside Surface Roughness P = 18 bar, H = 100 um")
plt.grid()

plot3 = plt.figure(3)
plt.scatter(Y,XY_P18_H150[0,:], color='r',label='1')
plt.scatter(Y,XY_P18_H150[1,:], color='g',label='2')
plt.scatter(Y,XY_P18_H150[2,:], color='b',label='3')
plt.plot(Y,BF_P18_H150[0,:],'r')
plt.plot(Y,BF_P18_H150[1,:],'g')
plt.plot(Y,BF_P18_H150[2,:],'b')
plt.plot(Y,XY_P18[2,:])
plt.legend()
plt.xlabel("Cut Length (mm)")
plt.ylabel("Surface Roughness (mm)")
plt.title("Underside Surface Roughness P = 18 bar, H = 150 um")
plt.grid()

plot4 = plt.figure(4)
plt.scatter(Y,XY_P18_H200[0,:], color='r',label='1')
plt.scatter(Y,XY_P18_H200[1,:], color='g',label='2')
plt.scatter(Y,XY_P18_H200[2,:], color='b',label='3')
plt.scatter(Y1,XY_P18_H200_new, color = 'c', label= '4')
plt.plot(Y,BF_P18_H200[0,:],'r')
plt.plot(Y,BF_P18_H200[1,:],'g')
plt.plot(Y,BF_P18_H200[2,:],'b')
plt.plot(Y1,BF_P18_H200_new,'c')
plt.legend()
plt.xlabel("Cut Length (mm)")
plt.ylabel("Surface Roughness (mm)")
plt.title("Underside Surface Roughness P = 18 bar, H = 200 um")
plt.grid()

plot5 = plt.figure(5)
plt.plot(Y,XY_P18[0,:], label="50 um")
plt.plot(Y,XY_P18[1,:], label="100 um")
plt.plot(Y,XY_P18[2,:], label="150 um")
plt.plot(Y,XY_P18[3,:], label="200 um")
plt.legend()
plt.xlabel("Cut Length (mm)")
plt.ylabel("Surface Roughness (mm)")
plt.title("Averaged Underside Roughness Variation with Run Height at P =18 bar")
plt.grid()

plot6 = plt.figure(6)
plt.scatter(Z,Ra_ave_P18)
plt.xlabel("Run Height (um)")
plt.ylabel("Surface Roughness (um)")
plt.title("Averaged Surface Roughness Variation with Run Height")
plt.grid()

#plot2 = plt.figure(2)
#plt.plot(Y_1, XYN_P18_H50_1,color='r')
#plt.plot(Y_2, XYN_P18_H50_2)
#plt.plot(Y_3, XYN_P18_H50_3)
#plt.xlabel("Cut Length (mm)")
#plt.ylabel("Surface Roughness (um)")
#plt.grid()

plt.show()
