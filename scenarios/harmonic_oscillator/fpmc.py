from model.variationalmontecarlo import *
from model.auxilary import *
from model.samplers import *
from model.harmonicoscillator1d import *

delta_t = 0.001

for alpha in np.arange(0.4, 0.64, 0.05):
    psi2 = wavefunction2(alpha)
    MC = FokkerPlanckMonteCarlo(standard_initial_dist(()), force(alpha), delta_t)
    samples = MC.generate_samples(400, 70000, 40000)
    print(f"{alpha}:", mc_integrator(local_energy(alpha), samples.get_samples()))
    denisty_plot(samples, psi2)