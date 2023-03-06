import numpy as np
import matplotlib.pyplot as plt

#Material properties based on Fatigue Resistance of Steels SAE 1045 Steel
D = 0.3
UTS = 510
HB = 215

de = 1.229e-3
s_prime = UTS + 345

E = 193e3
b_estimate = -0.05
c_estimate = -0.7
if HB < 200:
    e_prime = 1
if 200 < HB < 300:
    e_prime = 0.5
if HB > 400:
    e_prime = 0.1

s_prime_UML = 1.5*UTS
b_UML = -0.087
c_UML = -0.58

if UTS / E <= 0.003:
    psi = 1
if UTS / E >= 0.003:
    psi = 1.375 - 125 * UTS / E

s_prime_Hard = 4.25*HB + 225
b_Hard = -0.09
e_prime_Hard = (0.32*(HB)**2-487*HB + 191e3)/E
c_Hard = -0.56

if HB > 50 and HB < 340:
    s_prime_SFM = 3.98 * HB + 285
    e_prime_SFM = 1.5e-6 * HB ** 2.35
    c_prime_SFM = -0.54
if HB > 340 and HB < 700:
    s_prime_SFM = 3.98 * HB + 285
    e_prime_SFM = 1.7e12 * HB ** (-4.78)
    c_prime_SFM = -0.69

N_min = -1
N_max = 9
N_N = 100*((N_max-N_min)+1)
N = np.logspace(N_min,N_max,num=N_N)

dE = np.zeros(N_N)
dE_USE = np.zeros(N_N)
dE_MUSE = np.zeros(N_N)
dE_UML = np.zeros(N_N)
dE_Hard = np.zeros(N_N)
dE_SFM = np.zeros(N_N)


def fatigue_life_estimate():
    for i in range(0,N_N):
        dE[i] = 2*(e_prime*(2*N[i])**c_estimate)+((s_prime/E)*(2*N[i])**b_estimate)
        if dE[i]<de:
            return[N[i]]

def fatigue_life_USE():
    for i in range(0,N_N):
        dE_USE[i] = (N[i]/D)**-0.6 + 3.5*(UTS/E)*N[i]**(-0.12)
        if dE_USE[i]<de:
            return[N[i]]

def fatigue_life_MUSE():
    for i in range(0,N_N):
        dE_MUSE[i] = 0.0266*D**0.155*(UTS/E)**(-0.53)*N[i]**(-0.56) + 1.17*(UTS/E)**(0.832)*N[i]**(-0.09)
        if dE_MUSE[i]<de:
            return[N[i]]

def fatigue_life_UML():
    for i in range(0,N_N):
        e_prime_UML = 0.59*psi
        dE_UML[i] = 2*(e_prime_UML*(2*N[i])**c_UML)+((s_prime_UML/E)*(2*N[i])**b_UML)
        if dE_UML[i]<de:
            return[N[i]]

def fatigue_life_Hard():
    for i in range(0,N_N):
        dE_Hard[i] = 2*(e_prime_Hard*(2*N[i])**c_Hard)+((s_prime_Hard/E)*(2*N[i])**b_Hard)
        if dE_Hard[i]<de:
            return[N[i]]

def fatigue_life_SFM():
    for i in range(0,N_N):
        dE_SFM[i] = 2 * (e_prime_SFM * (2 * N[i]) ** c_prime_SFM) + ((s_prime_SFM / E) * (2 * N[i]) ** b_Hard)
        if dE_SFM[i]<de:
            return[N[i]]

def fatigue_life_SWT():
    UTS = 510
    b = -0.087  # Fatigue strength exponent
    c = -0.58  # Fatigue ductility exponent
    von_mises_stress = 340
    strain_amplitude = 0.15/(2*50)
    s_f = 1.5 * UTS

    if UTS / E <= 0.003:
        psi = 1
    if UTS / E >= 0.003:
        psi = 1.375 - 125 * UTS / E

    e_f = 0.59 * psi

    N_min = 1
    N_max = 10
    N_N = 100 * ((N_max - N_min) + 1)
    N = np.logspace(N_min, N_max, num=N_N)

    for i in range(0, N_N):
        LHS_SWT = von_mises_stress*strain_amplitude*E
        RHS_SWT = (s_f)**2 * (2*N[i])**(2*b) + s_f * e_f * E * (2*N[i])**(b+c)
        if RHS_SWT >= LHS_SWT:
            return N[i]


N_cycles_estimate = fatigue_life_estimate()
N_cycles_USE = fatigue_life_USE()
N_cycles_MUSE = fatigue_life_MUSE()
N_cycles_UML = fatigue_life_UML()
N_cycles_Hard = fatigue_life_Hard()
N_cycles_SFM = fatigue_life_SFM()

plt.figure(1)
plt.loglog(N,dE,label="Empirical Fatigue Model")
plt.loglog(N,dE_UML,label="Uniform Material Law")
plt.loglog(N,dE_Hard,label="Hardness Method")
plt.loglog(N,dE_SFM,label="Segment Fitting Method")

plt.xlabel("Number of Reversals")
plt.ylabel("Strain Amplitude")
plt.grid()
plt.legend()
plt.title("Strain Based Model")

plt.figure(2)
plt.loglog(N,dE_USE,label="Universal Slopes Equation")
plt.loglog(N,dE_MUSE,label="Modified Slopes Equation")

plt.xlabel("Number of Reversals")
plt.ylabel("Strain Amplitude")
plt.grid()
plt.legend()
plt.title("Hardness Based Model")

plt.show()