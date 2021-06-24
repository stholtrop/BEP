from model.diffusionmontecarlo import *
from model.auxilary import *
from model.samplers import *
from model.harmonicoscillator3d import *

import numpy as np

# A harmonic oscilator in one dimension with hamiltonian -0.5 Laplace + .5 x**2
param = 0.5
delta_t = 0.01
alpha = 1
E_0 = 1.5

MC = DiffusionMonteCarlo(potential(param), standard_initial_dist((3,)), delta_t=delta_t, E_0=E_0, alpha=alpha)
sampler = MC.generate_samples(500, 50000, 10000)
print(np.mean(sampler.energies.get_samples()),"+/-", np.std(sampler.energies.get_samples()))
denisty_plot_spherical(sampler.samples, wavefunction(param))