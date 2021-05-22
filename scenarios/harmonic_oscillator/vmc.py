from model.variationalmontecarlo import *
from model.auxilary import *
from model.samplers import *

import matplotlib.pyplot as plt

def wavefunction(alpha):
    return lambda x : np.exp(-alpha*x**2)

def wavefunction2(alpha):
    return lambda x : np.exp(-2*alpha*x**2)

def local_energy(alpha):
    def r(x):
        return alpha + x**2 *(0.5 - 2* alpha**2)
    return r

# A harmonic oscilator in one dimension with hamiltonian -0.5 Laplace + .5 x**2
bins = 500
for alpha in np.arange(0.4, 0.64, 0.05):
    psi2 = wavefunction2(alpha)
    MC = VariationalMonteCarlo(psi2, lambda x : x + (np.random.rand(len(x)) - 0.5), lambda N : 2*(np.random.rand(N) - 0.5))
    samples = MC.generate_samples(400, 30000, 4000)
    plt.hist(samples.get_samples(), bins=bins, density=True)
    x = np.linspace(-10, 10, bins)
    f = psi2(x[:])
    C = 20*np.sum(f)/bins
    plt.plot(x, f/C)
    plt.show()
    print(f"{alpha}:", mc_integrator(local_energy(alpha), samples.get_samples()))