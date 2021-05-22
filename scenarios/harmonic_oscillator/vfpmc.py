from model.variationalmontecarlo import *
from model.auxilary import *
from model.samplers import *
from model.harmonicoscillator import *

import matplotlib.pyplot as plt

# A harmonic oscilator in one dimension with hamiltonian -0.5 Laplace + .5 x**2
for alpha in np.arange(0.4, 0.64, 0.05):
    psi2 = wavefunction2(alpha)
    MC = VariationalFokkerPlanckMonteCarlo(psi2, lambda N : (np.random.rand(N) - 0.5), force(alpha), 0.001)
    samples = MC.generate_samples(400, 30000, 4000)
    print(f"{alpha}:", mc_integrator(local_energy(alpha), samples.get_samples()))
    denisty_plot(samples, psi2)