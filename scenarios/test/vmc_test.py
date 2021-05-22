from model.variationalmontecarlo import *
import matplotlib.pyplot as plt

def function(x):
    return np.exp(-np.linalg.norm(x, axis=1))

dimension = 3

MC = VariationalMonteCarlo(function, lambda x : x + 1*(np.random.rand(*x.shape) - 0.5), lambda N : np.random.rand(N, dimension))
walkers = 400
epochs = 30000
skipped = 4000
bins = 500
result = MC.generate_samples(walkers, epochs, skipped)
plt.hist(np.linalg.norm(result.get_samples(),axis=1), bins=bins, density=True)
x = np.linspace(0, 10, 500)
f = function(x[:,None])*x**(dimension-1)
C = 10*np.sum(f)/bins
plt.plot(x, f/C)
plt.show()
