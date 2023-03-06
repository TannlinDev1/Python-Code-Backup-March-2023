import matplotlib.pyplot as plt
import os
import numpy as np

os.chdir("C:/Users/Tannlin User/Documents/Tannlin Internship 2020/Side Profiles/MicroVue Measurements/Nozzle Diameters")
os.getcwd()

step = 0.1
N_step = 60

XY_H50_N1200 = np.zeros((4,122))
XY_H50_N1000 = np.zeros((4,122))
XY_H50_N800 = np.zeros((4,122))

XY_H100_N1200 = np.zeros((4,122))
XY_H100_N1000 = np.zeros((4,122))
XY_H100_N800 = np.zeros((4,122))

XY_H50_N800[0,:] = np.loadtxt(fname = "XY_H50_N800_1.txt", delimiter=',')
XY_H50_N800[1,:] = np.loadtxt(fname = "XY_H50_N800_2.txt", delimiter=',')
XY_H50_N800[2,:] = np.loadtxt(fname = "XY_H50_N800_3.txt", delimiter=',')
XY_H50_N800[3,:] = np.loadtxt(fname = "XY_H50_N800_4.txt", delimiter=',')

XY_H50_N1000[0,:] =np.loadtxt(fname = "XY_H50_N1000_1.txt", delimiter=',')
XY_H50_N1000[1,:] =np.loadtxt(fname = "XY_H50_N1000_2.txt", delimiter=',')
XY_H50_N1000[2,:] =np.loadtxt(fname = "XY_H50_N1000_3.txt", delimiter=',')
XY_H50_N1000[3,:] =np.loadtxt(fname = "XY_H50_N1000_4.txt", delimiter=',')

XY_H50_N1200[0,:] = np.loadtxt(fname = "XY_H50_N1200_1.txt", delimiter=',')
XY_H50_N1200[1,:] = np.loadtxt(fname = "XY_H50_N1200_2.txt", delimiter=',')
XY_H50_N1200[2,:] = np.loadtxt(fname = "XY_H50_N1200_3.txt", delimiter=',')
XY_H50_N1200[3,:] = np.loadtxt(fname = "XY_H50_N1200_4.txt", delimiter=',')

XY_H100_N800[0,:] = np.loadtxt(fname = "XY_H100_N800_1.txt", delimiter=',')
XY_H100_N800[1,:] = np.loadtxt(fname = "XY_H100_N800_2.txt", delimiter=',')
XY_H100_N800[2,:] = np.loadtxt(fname = "XY_H100_N800_3.txt", delimiter=',')
XY_H100_N800[3,:] = np.loadtxt(fname = "XY_H100_N800_4.txt", delimiter=',')

XY_H100_N1000[0,:] =np.loadtxt(fname = "XY_H100_N1000_1.txt", delimiter=',')
XY_H100_N1000[1,:] =np.loadtxt(fname = "XY_H100_N1000_2.txt", delimiter=',')
XY_H100_N1000[2,:] =np.loadtxt(fname = "XY_H100_N1000_3.txt", delimiter=',')
XY_H100_N1000[3,:] =np.loadtxt(fname = "XY_H100_N1000_4.txt", delimiter=',')

XY_H100_N1200[0,:] = np.loadtxt(fname = "XY_H100_N1200_1.txt", delimiter=',')
XY_H100_N1200[1,:] = np.loadtxt(fname = "XY_H100_N1200_2.txt", delimiter=',')
XY_H100_N1200[2,:] = np.loadtxt(fname = "XY_H100_N1200_3.txt", delimiter=',')
XY_H100_N1200[3,:] = np.loadtxt(fname = "XY_H100_N1200_4.txt", delimiter=',')

XYC_H50_N800 = np.zeros((4,61))
XYF_H50_N800 = np.zeros((4,61))

XYC_H50_N1000 = np.zeros((4,61))
XYF_H50_N1000 = np.zeros((4,61))

XYC_H50_N1200 = np.zeros((4,61))
XYF_H50_N1200 = np.zeros((4,61))

XYC_H100_N800 = np.zeros((4,61))
XYF_H100_N800 = np.zeros((4,61))

XYC_H100_N1000 = np.zeros((4,61))
XYF_H100_N1000 = np.zeros((4,61))

XYC_H100_N1200 = np.zeros((4,61))
XYF_H100_N1200 = np.zeros((4,61))

for i in range(0,4):

    XYC_H50_N800[i] = XY_H50_N800[i,0:61]
    XYF_H50_N800[i] = XY_H50_N800[i,61:122]

    XYC_H50_N1000[i] = XY_H50_N1000[i,0:61]
    XYF_H50_N1000[i] = XY_H50_N1000[i,61:122]

    XYC_H50_N1200[i] = XY_H50_N1200[i,0:61]
    XYF_H50_N1200[i] = XY_H50_N1200[i,61:122]

    XYC_H100_N800[i] = XY_H100_N800[i,0:61]
    XYF_H100_N800[i] = XY_H100_N800[i,61:122]

    XYC_H100_N1000[i] = XY_H100_N1000[i,0:61]
    XYF_H100_N1000[i] = XY_H100_N1000[i,61:122]

    XYC_H100_N1200[i] = XY_H100_N1200[i,0:61]
    XYF_H100_N1200[i] = XY_H100_N1200[i,61:122]
    
Range_H50_N800 = np.zeros((4,len(XYC_H50_N800[0,:])))
Range_H50_N1000 = np.zeros((4,len(XYC_H50_N1000[0,:])))
Range_H50_N1200 = np.zeros((4,len(XYC_H50_N1200[0,:])))

Range_H100_N800 = np.zeros((4,len(XYC_H50_N800[0,:])))
Range_H100_N1000 = np.zeros((4,len(XYC_H50_N1000[0,:])))
Range_H100_N1200 = np.zeros((4,len(XYC_H50_N1200[0,:])))

for j in range(0,4):
    for i in range(0,len(XYC_H50_N800[j,:])):

        Range_H50_N800[j,i] = XYC_H50_N800[j,i]-XYF_H50_N800[j,i]
        Range_H50_N1000[j,i] = XYC_H50_N1000[j,i]-XYF_H50_N1000[j,i]
        Range_H50_N1200[j,i] = XYC_H50_N1200[j,i]-XYF_H50_N1200[j,i]

        Range_H100_N800[j,i] = XYC_H100_N800[j,i]-XYF_H100_N800[j,i]
        Range_H100_N1000[j,i] = XYC_H100_N1000[j,i]-XYF_H100_N1000[j,i]
        Range_H100_N1200[j,i] = XYC_H100_N1200[j,i]-XYF_H100_N1200[j,i]

Ave_Range_H50_N800 = np.zeros(len(Range_H50_N800[0,:]))
Ave_Range_H50_N1000 = np.zeros(len(Range_H50_N800[0,:]))
Ave_Range_H50_N1200 = np.zeros(len(Range_H50_N800[0,:]))

Ave_Range_H100_N800 = np.zeros(len(Range_H50_N800[0,:]))
Ave_Range_H100_N1000 = np.zeros(len(Range_H50_N800[0,:]))
Ave_Range_H100_N1200 = np.zeros(len(Range_H50_N800[0,:]))

for i in range(0,len(Range_H50_N800[0,:])):
    
    Ave_Range_H50_N800[i] = (Range_H50_N800[0,i]+Range_H50_N800[1,i]+Range_H50_N800[2,i]+Range_H50_N800[3,i])/4
    Ave_Range_H50_N1000[i] = (Range_H50_N1000[0,i]+Range_H50_N1000[1,i]+Range_H50_N1000[2,i]+Range_H50_N1000[3,i])/4
    Ave_Range_H50_N1200[i] = (Range_H50_N1200[0,i]+Range_H50_N1200[1,i]+Range_H50_N1200[2,i]+Range_H50_N1200[3,i])/4

    Ave_Range_H100_N800[i] = (Range_H100_N800[0,i]+Range_H100_N800[1,i]+Range_H100_N800[2,i]+Range_H100_N800[3,i])/4
    Ave_Range_H100_N1000[i] = (Range_H100_N1000[0,i]+Range_H100_N1000[1,i]+Range_H100_N1000[2,i]+Range_H100_N1000[3,i])/4
    Ave_Range_H100_N1200[i] = (Range_H100_N1200[0,i]+Range_H100_N1200[1,i]+Range_H100_N1200[2,i]+Range_H100_N1200[3,i])/4

AveBatchRange_H50_N800 = np.mean(Ave_Range_H50_N800)
AveBatchRange_H50_N1000 = np.mean(Ave_Range_H50_N1000)
AveBatchRange_H50_N1200 = np.mean(Ave_Range_H50_N1200)

AveBatchRange_H100_N800 = np.mean(Ave_Range_H100_N800)
AveBatchRange_H100_N1000 = np.mean(Ave_Range_H100_N1000)
AveBatchRange_H100_N1200 = np.mean(Ave_Range_H100_N1200)

Y = np.linspace(0, len(XYC_H50_N800[0,:]), num = len(XYC_H50_N800[0,:]))

plot1 = plt.figure(1)

plt.plot(Y,Ave_Range_H50_N800,color='r',label='0.8 mm')
plt.plot(Y,Ave_Range_H50_N1000,color='g',label='1.0 mm')
plt.plot(Y,Ave_Range_H50_N1200,color='b',label='1.2 mm')
plt.title('Averaged Range for Run Height of 50 um')
plt.legend()
plt.grid()
plt.xlabel('Cut Distance (mm)')
plt.ylabel('Range (um)')

plot2 = plt.figure(2)

plt.plot(Y,Ave_Range_H100_N800,color='r',label='0.8 mm')
plt.plot(Y,Ave_Range_H100_N1000,color='g',label='1.0 mm')
plt.plot(Y,Ave_Range_H100_N1200,color='b',label='1.2 mm')
plt.title('Averaged Range for Run Height of 100 um')
plt.legend()
plt.grid()
plt.xlabel('Cut Distance (mm)')
plt.ylabel('Range (um)')

plt.show()
