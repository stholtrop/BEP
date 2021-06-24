from model.variationalmontecarlo import *
from model.auxilary import *
from model.samplers import *
from model.harmonicoscillator1d import *

for alpha in np.arange(0.4, 0.64, 0.05):
    psi2 = wavefunction2(alpha)
    MC = VariationalMonteCarlo(psi2, lambda x : x + (np.random.rand(len(x)) - 0.5), standard_initial_dist(()))
    samples = MC.generate_samples(400, 70000, 40000)
    print(f"{alpha}:", mc_integrator(local_energy(alpha), samples.get_samples()))
    denisty_plot(samples, psi2)