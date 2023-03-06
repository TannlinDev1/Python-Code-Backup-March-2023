#This script plots frequency vs amplitude data and passes it through a Savitzky-Golay filter for post processing

#import packages
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.signal import savgol_filter

#change directory
os.chdir(r"C:\Users\angus.mcallister\Documents\TFF\Frequency Testing\VG")

#memory preallocation
VG_100um = np.zeros(((512,2,10)))
VG_100um_S = np.zeros((512,10))
VG_100um_SA = np.zeros(512)
N_RS = 1000
VG_100um_RS = np.zeros((N_RS,10))
freq = np.logspace(2,3,N_RS)

#import, apply filter and plot data

for i in range(0,10):
    fname_txt = "VG_100um_" + str(i+1) + ".txt"
    VG_100um[:,:,i] = np.loadtxt(fname=fname_txt,delimiter=",")
    #plt.semilogx(VG_100um[:,0,i],VG_100um[:,1,i])
    VG_100um_S[:,i] = savgol_filter(VG_100um[:,1,i], 11, 3)
    plt.figure(1)
    plt.semilogx(VG_100um[:,0,i],VG_100um_S[:,i])
    plt.grid()
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Amplitude (dB)")
    plt.title("VG - Filtered")
    VG_100um_RS[:, i] = np.interp(freq, VG_100um[:, 0, i], VG_100um_S[:, i])
    plt.figure(2)
    plt.semilogx(freq, VG_100um_RS[:,i])
    plt.grid()
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Amplitude (dB)")
    plt.title("VG - Rescaled")

#average data across samples
VG_100um_ave = np.zeros(1000)

for i in range(0,1000):
    VG_100um_ave[i] = np.average(VG_100um_RS[i,:])

os.chdir(r"C:\Users\angus.mcallister\Documents\TFF\Frequency Testing\MPM")

#memory preallocation
MPM_200um = np.zeros(((512,2,10)))
MPM_200um_S = np.zeros((512,10))
MPM_200um_SA = np.zeros(512)
MPM_200um_RS = np.zeros((N_RS,10))

#import, apply filter and plot data

for i in range(0,10):
    fname_txt = "MPM_" + str(i+1) + ".txt"
    MPM_200um[:,:,i] = np.loadtxt(fname=fname_txt,delimiter=",")
    #plt.semilogx(MPM_200um[:,0,i],MPM_200um[:,1,i])
    MPM_200um_S[:,i] = savgol_filter(MPM_200um[:,1,i], 11, 3)
    plt.figure(3)
    plt.semilogx(MPM_200um[:,0,i],MPM_200um_S[:,i])
    plt.grid()
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Amplitude (dB)")
    plt.title("MPM - Filtered")
    MPM_200um_RS[:, i] = np.interp(freq, MPM_200um[:, 0, i], MPM_200um_S[:, i])
    plt.figure(4)
    plt.semilogx(freq, MPM_200um_RS[:,i])
    plt.grid()
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Amplitude (dB)")
    plt.title("MPM - Rescaled")

#average data across samples
MPM_200um_ave = np.zeros(1000)

for i in range(0,1000):
    MPM_200um_ave[i] = np.average(MPM_200um_RS[i,:])

os.chdir(r"C:\Users\angus.mcallister\Documents\TFF\Frequency Testing\TFF")

#memory preallocation
TFF = np.zeros(((512,2,10)))
TFF_S = np.zeros((1000,10))
TFF_SA = np.zeros(512)
TFF_RS = np.zeros((N_RS,10))

#import, apply filter and plot data

for i in range(0,10):
    fname_txt = "TFF_" + str(i+1) + ".txt"
    TFF[:,:,i] = np.loadtxt(fname=fname_txt,delimiter=",")

    TFF_RS[:, i] = np.interp(freq, TFF[:, 0, i], TFF[:, 1, i])
    plt.figure(5)
    plt.semilogx(freq,TFF_RS[:,i])
    plt.grid()
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Amplitude (dB)")
    plt.title("TFF - Rescaled")

    TFF_S[:,i] = savgol_filter(TFF_RS[:, i], 11, 3)
    plt.figure(6)
    plt.semilogx(freq, TFF_S[:,i])
    plt.grid()
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Amplitude (dB)")
    plt.title("TFF - Smoothed")

#average data across samples
TFF_ave = np.zeros(1000)

for i in range(0,1000):
    TFF_ave[i] = np.average(TFF_S[i,:])

plt.figure(7)
plt.plot(freq, VG_100um_ave,label="VG")
plt.plot(freq,TFF_ave, label="TFF")
plt.plot(freq,MPM_200um_ave, label="MPMM")
plt.grid()
plt.legend()
plt.xlabel("Frequency (Hz)")
plt.ylabel("Amplitude (dB)")
plt.title("Comparison")
