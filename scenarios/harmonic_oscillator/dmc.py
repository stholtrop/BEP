from model.diffusionmontecarlo import *
from model.auxilary import *
from model.samplers import *
from model.harmonicoscillator1d import *

import numpy as np

# A harmonic oscilator in one dimension with hamiltonian -0.5 Laplace + .5 x**2
param = 0.5
delta_t = 0.001
alpha = 0.01
E_0 = 0.5

MC = DiffusionMonteCarlo(potential(param), standard_initial_dist(()), delta_t=delta_t, E_0=E_0, alpha=alpha)
sampler = MC.generate_samples(500, 100000, 80000)
print(np.mean(sampler.energies.get_samples()))
denisty_plot(sampler.samples, wavefunction(param))