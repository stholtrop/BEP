from model.montecarlo import *
import matplotlib.pyplot as plt

def wavefunction(alpha):
    return lambda x : np.exp(-alpha*x**2)

def wavefunction2(alpha):
    return lambda x : np.exp(-2*alpha*x**2)

def local_energy(alpha, x):
    return alpha + x**2 *(0.5 - 2* alpha**2)

# A harmonic oscilator in one dimension with hamiltonian -0.5 Laplace + .5 x**2

MC = VariationalMonteCarlo(wavefunction2(0.4), lambda x : x + np.random.rand(len(x))*0.05 - 0.025, lambda N : np.random.rand(N))
result = MC.generate_samples(400, 60000, 30000)
print(np.sum(local_energy(0.4, result))/len(result))
plt.hist(result, bins=100)
plt.show()