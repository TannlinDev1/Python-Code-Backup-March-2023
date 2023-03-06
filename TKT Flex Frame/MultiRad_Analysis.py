import pandas as pd
import matplotlib.pyplot as plt
import os

os.chdir(r"C:\Users\angus.mcallister\Desktop\CSV Data")

dF_1R = pd.read_csv("dF_1R.csv")
dF_2R = pd.read_csv("dF_2R.csv")
dF_2RB = pd.read_csv("dF_2RB.csv")
dF_2RC = pd.read_csv("dF_2RC.csv")
dF_2RD = pd.read_csv("dF_2RD.csv")
dF_2RE = pd.read_csv("dF_2RE.csv")
dF_2RF = pd.read_csv("dF_2RF.csv")

dF_1R.iloc[:,0] *= 6
dF_2R.iloc[:,0] *= 10
dF_2RB.iloc[:,0] *= 20
dF_2RC.iloc[:,0] *= 20
dF_2RD.iloc[:,0] *= 20
dF_2RE.iloc[:,0] *= 20
dF_2RF.iloc[:,0] *= 20

os.chdir(r"C:\Users\angus.mcallister\Desktop\CSV Data\Stress")
dS_2RB =  pd.read_csv("dS_2RB.csv")
dS_2RC =  pd.read_csv("dS_2RC.csv")
dS_2RD =  pd.read_csv("dS_2RD.csv")
dS_2RE =  pd.read_csv("dS_2RE.csv")
dS_2RF =  pd.read_csv("dS_2RF.csv")

dS_2RB.iloc[:,1] /= 1e6
dS_2RD.iloc[:,1] /= 1e6
dS_2RE.iloc[:,1] /= 1e6
dS_2RF.iloc[:,1] /= 1e6

plt.figure(1)
plt.plot(dF_1R.iloc[:,1], dF_1R.iloc[:,0], label="Single Rad")
plt.plot(dF_2R.iloc[:,1], dF_2R.iloc[:,0], label="Double Rad")
plt.plot(dF_2RB.iloc[:,1], dF_2RB.iloc[:,0], label="Double Rad B")
plt.plot(dF_2RC.iloc[:,1], dF_2RC.iloc[:,0], label="Double Rad C")
plt.plot(dF_2RD.iloc[:,1], dF_2RD.iloc[:,0], label="Double Rad D")
plt.plot(dF_2RE.iloc[:,1], dF_2RE.iloc[:,0], label="Double Rad E")
plt.plot(dF_2RF.iloc[:,1], dF_2RF.iloc[:,0], label="Double Rad F")
plt.legend()
plt.xlabel("Displacement (mm)")
plt.ylabel("Force (N)")
plt.grid()

plt.figure(2)
plt.plot(dS_2RB.iloc[:,0], dS_2RB.iloc[:,1], label="Double Rad B")
plt.plot(dS_2RC.iloc[:,0], dS_2RC.iloc[:,1], label="Double Rad C")
plt.plot(dS_2RD.iloc[:,0], dS_2RD.iloc[:,1], label="Double Rad D")
plt.plot(dS_2RE.iloc[:,0], dS_2RE.iloc[:,1], label="Double Rad E")
plt.plot(dS_2RF.iloc[:,0], dS_2RF.iloc[:,1], label="Double Rad F")
plt.legend()
plt.xlabel("Normalised Length")
plt.ylabel("Von Mises Stress (MPa)")
plt.grid()