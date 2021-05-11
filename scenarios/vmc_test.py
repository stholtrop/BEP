from model.montecarlo import *
import matplotlib.pyplot as plt

def function(x):
    return x**1*(1 - x)**4 * ((x > 0) & (x < 1))

MC = VariationalMonteCarlo(function, lambda x : x + np.random.rand(len(x))*0.05 - 0.025, lambda N : np.random.rand(N))
walkers = 400
epochs = 30000
skipped = 4000
bins = 500
result = MC.generate_samples(walkers, epochs, skipped)
plt.hist(result.get_samples(), bins=bins, density=True)
x = np.linspace(0, 1, 500)
f = function(x)
plt.plot(x, 500*f/np.sum(f))
plt.show()