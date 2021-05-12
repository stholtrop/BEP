from model.variationalmontecarlo import *
from model.auxilary import *
from model.samplers import *

import matplotlib.pyplot as plt
import numpy as np

def wavefunction(alpha):
    return lambda x : np.exp(-alpha*np.linalg.norm(x, axis = 1))

def wavefunction2(alpha):
    return lambda x : np.exp(-2*alpha*np.linalg.norm(x, axis=1))

def force(alpha):
    return lambda x : -2 * alpha * x/np.linalg.norm(x,axis=1)[:,None]

def wavefunction21d(alpha):
    return lambda r : np.exp(-2*alpha*r)

def local_energy(alpha):
    def r(x):
        rx = np.linalg.norm(x, axis=1)[:,None]
        return -1/rx - 0.5*alpha*(alpha - 2/rx)
    return r

# A harmonic oscilator in one dimension with hamiltonian -0.5 Laplace + .5 x**2
bins = 500
for alpha in np.arange(0.8, 1.3, 0.1):
    MC = FokkerPlanckMonteCarlo(lambda N: np.random.rand(N, 3), force(alpha), 0.1)
    samples = MC.generate_samples(400, 40000, 14000)
    print(f"{alpha}:", mc_integrator(local_energy(alpha), samples.get_samples()))
    plt.hist(np.linalg.norm(samples.get_samples(),axis=1), bins=bins, density=True)
    x = np.linspace(0, 10, bins)
    f = wavefunction21d(alpha)(x)*x**2
    C = 10 * np.sum(f)/bins
    plt.plot(x, f/C)
    plt.show()
