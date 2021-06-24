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

def denisty_plot_spherical(sampler, density, bins=500, a=0, b=6):
    plt.hist(np.linalg.norm(sampler.get_samples(), axis = 1), bins=bins, density=True, histtype='stepfilled')
    x = np.linspace(a, b, bins)
    f = density(x)*x**2
    C = (b - a)*np.sum(f)/bins
    plt.plot(x, f/C, '--')
    plt.xlabel('r')
    h = plt.ylabel('$\psi$')
    h.set_rotation(0)
    plt.show()

def standard_initial_dist(shape):
    return lambda N : np.random.normal(0, 1, (N, *shape))

