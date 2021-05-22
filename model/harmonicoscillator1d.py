import numpy as np

def potential(alpha):
    return lambda x : alpha*x**2

def wavefunction(alpha):
    return lambda x : np.exp(-alpha*x**2)

def wavefunction2(alpha):
    return lambda x : np.exp(-2*alpha*x**2)

def force(alpha):
    return lambda x : - 4 * alpha * x

def local_energy(alpha):
    def r(x):
        return alpha + x**2 *(0.5 - 2* alpha**2)
    return r