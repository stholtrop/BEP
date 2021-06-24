from model.samplers import ListSampler
import numpy as np
import matplotlib.pyplot as plt


    
def V(x):
    return 0.5*np.linalg.norm(x)**2


target_size = 1000
pop_size = target_size
E_0 = 1.5
alpha = 0.1
E_target = E_0

iter_num = 30000#e6
energies = np.zeros((iter_num))
gamma = 0.5 # see 12.23
h = 0.1 # timestep Delta t equiv h
sigma = np.sqrt(2*gamma*h)
np.random.seed(0)
walkers = np.random.multivariate_normal(np.zeros(3), sigma*np, pop_size)
sampler = ListSampler()
# print(np.random.get_state())

for it in range(iter_num):
    walkers = walkers + np.random.normal(0.0, sigma, pop_size)
    q = np.exp(-h*(V(walkers)-E_target)) + np.random.rand(pop_size)
    q = np.floor(q).astype(int)
    walkers= np.repeat(walkers,q,axis=0)
    pop_size = walkers.size
    # input(pop_size)
    E_target = E_0 + alpha*np.log(target_size/pop_size)
    # input(E_target)
    energies[it] = E_target
    # print(it, walkers.shape, E_target)
    sampler.add_samples(walkers)

print(energies)
input()
plt.hist(sampler.get_samples(), bins=500, density=True)
x = np.linspace(-10, 10, 500)
f = np.exp(-0.5*x**2)
C = 20*np.sum(f)/500
plt.plot(x, f/C)
mean_energy = np.average(energies)
print (mean_energy)
plt.show()