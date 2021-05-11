import numpy as np

def mc_integrator(function, samples):
    return np.sum(function(samples))/len(samples)

