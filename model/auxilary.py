import numpy as np
import matplotlib.pyplot as plt

def mc_integrator(function, samples):
    return np.sum(function(samples))/len(samples)

def denisty_plot(sampler, density, bins=500, a=-10, b=10):
    plt.hist(sampler.get_samples(), bins=bins, density=True)
    x = np.linspace(a, b, bins)
    f = density(x)
    C = (b - a)*np.sum(f)/bins
    plt.plot(x, f/C)
    plt.show()

def standard_initial_dist(shape):
    return lambda N : np.random.normal(0, 1, (N, *shape))

