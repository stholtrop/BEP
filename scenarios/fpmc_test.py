from model.montecarlo import *
import matplotlib.pyplot as plt

def function(x):
    return np.exp(-np.abs(x))

    #x(1-x)^4 -> (1-x)^4 - 3x(1-x)^3 = (1-x)^3(1-4x)

def force(x):
    return (x < 0).astype(int) - (x > 0).astype(int)

MC = VariationalFokkerPlanckMonteCarlo(function, lambda N : np.random.normal(size=N), force, 0.1)
walkers = 400
epochs = 30000
skipped = 4000
bins = 500
result = MC.generate_samples(walkers, epochs, skipped)
plt.hist(result.get_samples(), bins=bins, density=True)
x = np.linspace(-10, 10, 500)
f = function(x)
C = 20*np.sum(f)/bins
plt.plot(x, f/C)
plt.show()