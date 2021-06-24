import numpy as np

def potential(alpha):
    return lambda x : alpha*np.linalg.norm(x, axis=1)**2

def wavefunction(alpha):
    return lambda x: np.exp(-alpha*x**2)