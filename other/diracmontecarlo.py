
import numpy as np
from model.samplers import ListSampler

import matplotlib.pyplot as plt

def potential(x):
    return 1/2*x**2

epochs = 50000
energies = np.zeros((epochs))
c = 1
delta_t = 0.001
rate = 0.1


target_size = 5000
pop_size = target_size
alpha = 1
E_0 = 1.09
E_target = E_0

walkers_pos = np.random.normal(0, 1, pop_size)
walkers_state = np.random.randint(2, size=pop_size) * 2 - 1

sampler = ListSampler()
sampler2 = ListSampler()

for e in range(epochs):
    walkers_state[np.random.rand(pop_size) < rate*delta_t] *= -1
    walkers_pos += walkers_state*c*delta_t
    q = (np.exp(-delta_t*(potential(walkers_pos) - E_target)) + np.random.rand(pop_size)).astype(int)
    walkers_pos = np.repeat(walkers_pos, q, axis=0)
    walkers_state = np.repeat(walkers_state, q, axis=0)
    pop_size = walkers_pos.size
    E_target = E_0 + alpha*np.log(target_size/pop_size)
    energies[e] = E_target
    if e > 5000:
        sampler.add_samples(walkers_pos)
        sampler2.add_samples(walkers_state)
positions = sampler.get_samples()
states = sampler2.get_samples()

plt.hist(sampler.get_samples(), bins=500, density=True, histtype='stepfilled')
#plt.hist(sampler.get_samples()[states==1], bins=500)
#plt.hist(sampler.get_samples()[states==-1], bins=500)
h = plt.ylabel("$\psi$")
h.set_rotation(0)
plt.xlabel("x")
mean_energy = np.average(energies[5000:])
x = np.linspace(-4, 4, 500)
f = np.exp(-x**2/2)
C = 8*np.sum(f)/500
plt.plot(x, f/C, '--')
print("Mean energy:", mean_energy)
print("Std:", np.std(energies[5000:]))
plt.show()