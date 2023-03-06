import matplotlib.pyplot as plt
import os
import numpy as np

os.chdir("C:/Users/Tannlin User/Documents/Tannlin Internship 2020/Side Profiles/MicroVue Measurements/Closest and Farthest Edge")
os.getcwd()

step = 0.1
N_step = 60

XY_H200 = np.zeros((3,122))
XY_H150 = np.zeros((3,122))
XY_H100 = np.zeros((3,122))
XY_H50 = np.zeros((3,122))

XY_H200[0,:] = np.loadtxt(fname = "XY_P18_H200_1.txt",delimiter=',')
XY_H200[1,:] = np.loadtxt(fname = "XY_P18_H200_2.txt",delimiter=',')
XY_H200[2,:] = np.loadtxt(fname = "XY_P18_H200_3.txt",delimiter=',')

XY_H150[0,:] = np.loadtxt(fname = "XY_P18_H150_1.txt",delimiter=',')
XY_H150[1,:] = np.loadtxt(fname = "XY_P18_H150_2.txt",delimiter=',')
XY_H150[2,:] = np.loadtxt(fname = "XY_P18_H150_3.txt",delimiter=',')

XY_H100[0,:] = np.loadtxt(fname = "XY_P18_H100_1.txt",delimiter=',')
XY_H100[1,:] = np.loadtxt(fname = "XY_P18_H100_2.txt",delimiter=',')
XY_H100[2,:] = np.loadtxt(fname = "XY_P18_H100_3.txt",delimiter=',')

XY_H50[0,:] = np.loadtxt(fname = "XY_P18_H50_1.txt",delimiter=',')
XY_H50[1,:] = np.loadtxt(fname = "XY_P18_H50_2.txt",delimiter=',')
XY_H50[2,:] = np.loadtxt(fname = "XY_P18_H50_3.txt",delimiter=',')

XYC_H200 = np.zeros((3,61))
XYF_H200 = np.zeros((3,61))

XYC_H150 = np.zeros((3,61))
XYF_H150 = np.zeros((3,61))

XYC_H100 = np.zeros((3,61))
XYF_H100 = np.zeros((3,61))

XYC_H50 = np.zeros((3,61))
XYF_H50 = np.zeros((3,61))

for i in range(0,3):

    XYC_H200[i] = XY_H200[i,0:61]
    XYF_H200[i] = XY_H200[i,61:122]

    XYC_H150[i] = XY_H150[i,0:61]
    XYF_H150[i] = XY_H150[i,61:122]

    XYC_H100[i] = XY_H100[i,0:61]
    XYF_H100[i] = XY_H100[i,61:122]

    XYC_H50[i] = XY_H50[i,0:61]
    XYF_H50[i] = XY_H50[i,61:122]

XYM_H200 = np.zeros((3,len(XYC_H200[0,:])))
XYM_H150 = np.zeros((3,len(XYC_H150[0,:])))
XYM_H100 = np.zeros((3,len(XYC_H100[0,:])))
XYM_H50 = np.zeros((3,len(XYC_H50[0,:])))

for j in range(0,3):
    for i in range(0,len(XYC_H200[j,:])):

        XYM_H200[j,i] = (XYC_H200[j,i]+XYF_H200[j,i])/2
        XYM_H150[j,i] = (XYC_H150[j,i]+XYF_H150[j,i])/2
        XYM_H100[j,i] = (XYC_H100[j,i]+XYF_H100[j,i])/2
        XYM_H50[j,i] = (XYC_H50[j,i]+XYF_H50[j,i])/2

XY_Av = np.zeros((4,len(XYM_H50[0,:])))
                 
for i in range(0,len(XYM_H50[0,:])):
    XY_Av[0,i] = (XYM_H50[0,i] + XYM_H50[1,i] + XYM_H50[2,i])/3
    XY_Av[1,i] = (XYM_H100[0,i] + XYM_H100[1,i] + XYM_H100[2,i])/3
    XY_Av[2,i] = (XYM_H150[0,i] + XYM_H150[1,i] + XYM_H150[2,i])/3
    XY_Av[3,i] = (XYM_H200[0,i] + XYM_H200[1,i] + XYM_H200[2,i])/3
            
Y = np.linspace(0, len(XYC_H200[0,:]), num = len(XYC_H200[0,:]))

deg = 3

PC_H200 = np.zeros((deg,4))
BFC_H200 = np.zeros((3,len(XYC_H200[0,:])))
PC_H150 = np.zeros((deg,4))
BFC_H150 = np.zeros((3,len(XYC_H150[0,:])))
PC_H100 = np.zeros((deg,4))
BFC_H100 = np.zeros((3,len(XYC_H100[0,:])))
PC_H50 = np.zeros((deg,4))
BFC_H50 = np.zeros((3,len(XYC_H50[0,:])))

PF_H200 = np.zeros((deg,4))
BFF_H200 = np.zeros((3,len(XYF_H200[0,:])))
PF_H150 = np.zeros((deg,4))
BFF_H150 = np.zeros((3,len(XYF_H150[0,:])))
PF_H100 = np.zeros((deg,4))
BFF_H100 = np.zeros((3,len(XYF_H100[0,:])))
PF_H50 = np.zeros((deg,4))
BFF_H50 = np.zeros((3,len(XYF_H50[0,:])))

PM_H200 = np.zeros((deg,4))
BFM_H200 = np.zeros((3,len(XYM_H200[0,:])))
PM_H150 = np.zeros((deg,4))
BFM_H150 = np.zeros((3,len(XYM_H150[0,:])))
PM_H100 = np.zeros((deg,4))
BFM_H100 = np.zeros((3,len(XYM_H100[0,:])))
PM_H50 = np.zeros((deg,4))
BFM_H50 = np.zeros((3,len(XYM_H50[0,:])))

for i in range(0,3):   
    PC_H200[i,:] = np.polyfit(Y,XYC_H200[i,:], deg)
    BFC_H200[i,:] = np.polyval(PC_H200[i,:], Y)
    PC_H150[i,:] = np.polyfit(Y,XYC_H150[i,:], deg)
    BFC_H150[i,:] = np.polyval(PC_H150[i,:], Y)
    PC_H100[i,:] = np.polyfit(Y,XYC_H100[i,:], deg)
    BFC_H100[i,:] = np.polyval(PC_H100[i,:], Y)
    PC_H50[i,:] = np.polyfit(Y,XYC_H50[i,:], deg)
    BFC_H50[i,:] = np.polyval(PC_H50[i,:], Y)

    PF_H200[i,:] = np.polyfit(Y,XYF_H200[i,:], deg)
    BFF_H200[i,:] = np.polyval(PF_H200[i,:], Y)
    PF_H150[i,:] = np.polyfit(Y,XYF_H150[i,:], deg)
    BFF_H150[i,:] = np.polyval(PF_H150[i,:], Y)
    PF_H100[i,:] = np.polyfit(Y,XYF_H100[i,:], deg)
    BFF_H100[i,:] = np.polyval(PF_H100[i,:], Y)
    PF_H50[i,:] = np.polyfit(Y,XYF_H50[i,:], deg)
    BFF_H50[i,:] = np.polyval(PF_H50[i,:], Y)

    PM_H200[i,:] = np.polyfit(Y,XYM_H200[i,:], deg)
    BFM_H200[i,:] = np.polyval(PM_H200[i,:], Y)
    PM_H150[i,:] = np.polyfit(Y,XYM_H150[i,:], deg)
    BFM_H150[i,:] = np.polyval(PM_H150[i,:], Y)
    PM_H100[i,:] = np.polyfit(Y,XYM_H100[i,:], deg)
    BFM_H100[i,:] = np.polyval(PM_H100[i,:], Y)
    PM_H50[i,:] = np.polyfit(Y,XYM_H50[i,:], deg)
    BFM_H50[i,:] = np.polyval(PM_H50[i,:], Y)

XY_BF = np.zeros((len(BFC_H200[0,:]),4))

for i in range(0,len(BFC_H200[0,:])):
               XY_BF[i,0] = (BFM_H50[0,i]+BFM_H50[1,i]+BFM_H50[2,i])/3
               XY_BF[i,1] = (BFM_H100[0,i]+BFM_H100[1,i]+BFM_H100[2,i])/3
               XY_BF[i,2] = (BFM_H150[0,i]+BFM_H150[1,i]+BFM_H150[2,i])/3
               XY_BF[i,3] = (BFM_H200[0,i]+BFM_H200[1,i]+BFM_H200[2,i])/3

Ra = np.zeros(4)
VarXY = np.zeros(4)

for i in range(0,4):

    Ra[i] = np.mean(XY_Av[i,:])
    VarXY[i] = np.var(XY_Av[i,:])

plt.figure(1)

for i in range(0,3):
    #plt.scatter(Y,XYC_H200[i,:],color='r',label='Farthest')
    #plt.scatter(Y,XYF_H200[i,:],color='g',label='Closest')
    #plt.scatter(Y,XYAv_H200[i,:],color='b',label='Averaged')
    plt.plot(Y,BFC_H200[i,:],color='r')
    plt.plot(Y,BFF_H200[i,:],color='g')
    plt.plot(Y,BFM_H200[i,:],color='b')
    plt.grid()

plt.xlabel("Cut Distance (mm)")
plt.ylabel("Surface Roughness (um)")
plt.title("Surface Roughness Across Cut Distance for H = 200 um")

plt.figure(2)

for i in range(0,3):
    #plt.scatter(Y,XYC_H200[i,:],color='r',label='Farthest')
    #plt.scatter(Y,XYF_H200[i,:],color='g',label='Closest')
    #plt.scatter(Y,XYAv_H200[i,:],color='b',label='Averaged')
    plt.plot(Y,BFC_H150[i,:],color='r')
    plt.plot(Y,BFF_H150[i,:],color='g')
    plt.plot(Y,BFM_H150[i,:],color='b')
    plt.grid()

plt.xlabel("Cut Distance (mm)")
plt.ylabel("Surface Roughness (um)")
plt.title("Surface Roughness Across Cut Distance for H = 150 um")

plt.figure(3)

for i in range(0,3):
    #plt.scatter(Y,XYC_H200[i,:],color='r',label='Farthest')
    #plt.scatter(Y,XYF_H200[i,:],color='g',label='Closest')
    #plt.scatter(Y,XYAv_H200[i,:],color='b',label='Averaged')
    plt.plot(Y,BFC_H100[i,:],color='r')
    plt.plot(Y,BFF_H100[i,:],color='g')
    plt.plot(Y,BFM_H100[i,:],color='b')
    plt.grid()

plt.xlabel("Cut Distance (mm)")
plt.ylabel("Surface Roughness (um)")
plt.title("Surface Roughness Across Cut Distance for H = 100 um")

plt.figure(4)

for i in range(0,3):
    #plt.scatter(Y,XYC_H200[i,:],color='r',label='Farthest')
    #plt.scatter(Y,XYF_H200[i,:],color='g',label='Closest')
    #plt.scatter(Y,XYAv_H200[i,:],color='b',label='Averaged')
    plt.plot(Y,BFC_H50[i,:],color='r')
    plt.plot(Y,BFF_H50[i,:],color='g')
    plt.plot(Y,BFM_H50[i,:],color='b')
    plt.grid()

plt.xlabel("Cut Distance (mm)")
plt.ylabel("Surface Roughness (um)")
plt.title("Surface Roughness Across Cut Distance for H = 50 um")

plt.figure(5)

plt.plot(Y,XY_Av[0,:],label='50 um')
plt.plot(Y,XY_Av[1,:],label='100 um')
plt.plot(Y,XY_Av[2,:],label='150 um')
plt.plot(Y,XY_Av[3,:],label='200 um')

plt.xlabel('Cut Distance (mm)')
plt.ylabel('Surface Roughness (um)')
plt.legend()
plt.grid()
plt.title('Averaged Surface Roughness Variation with Run Height')

plt.show()
