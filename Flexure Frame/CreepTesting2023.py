import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

os.chdir(r"\\tann-db1\R&D\TFF\Testing Results\Foil Tension Tests (Final Frame)")

strainSource = pd.read_csv("strain.csv")

strain1 = (strainSource.iloc[:,1]*0.1501-828.05)/584
strain2 = (strainSource.iloc[:,3]*0.1498-783.78)/584
strain3 = (strainSource.iloc[:,5]*0.15-829.807)/584
totalStrain = strain1 + strain2 + strain3

t = np.linspace(0, len(strain1), len(strain1))/60

plt.figure(1)
plt.semilogx(t, strain1, label="Load Cell A")
plt.semilogx(t, strain2, label="Load Cell B")
plt.semilogx(t, strain3, label="Load Cell C")
plt.xlabel("Time (hours)")
plt.ylabel("Foil Tension (N/mm)")
plt.legend()

plt.figure(2)
plt.semilogx(t, totalStrain)
plt.xlabel("Time (hours)")
plt.ylabel("Foil Tension (N/mm)")