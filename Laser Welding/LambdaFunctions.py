import numpy as np
import scipy.optimize as scopt
import scipy.special as scsp

q = 100

banana = lambda x: q*(x[1]-x[0]**2)**2+(1-x[0])**2
xopt = scopt.fmin(func=banana, x0=[-1.2,1])

temp_fcn = lambda r: (q/(2*np.pi*k*r))*(scsp.erf(r/np.sqrt(4*a*t))- scsp.erf(r/np.sqrt(4*np.pi*(t - t_p))))


