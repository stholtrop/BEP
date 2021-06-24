from typing import List
from model.samplers import *
from model.montecarlo import *

import numpy as np

class DiffusionMonteCarlo(MonteCarloMethod):
    def __init__(self, potential, initial_state, delta_t, E_0, alpha):
        self.potential = potential
        self.initial_state = initial_state
        self.delta_t = delta_t
        self.E_0 = np.array([E_0])
        self.alpha = alpha
    
    def generate_samples(self, N, epochs, skipped=0, sampler=None):
        if sampler == None: sampler = JointSampler({"samples" : ListSampler(), "energies" : ListSampler()})
        walkers = self.initial_state(N)
        E_target = self.E_0
        for i in range(epochs):
            walkers += np.random.multivariate_normal(np.zeros(walkers[0].shape), self.delta_t*np.identity(walkers[0].shape[0]), walkers.shape[0])
            q = (np.exp(-self.delta_t*(self.potential(walkers) - E_target)) + np.random.rand(walkers.shape[0])).astype(int)
            walkers = np.repeat(walkers, q, axis=0)
            E_target = self.E_0 + self.alpha*np.log(N/walkers.shape[0])
            if i >= skipped:
                sampler.samples.add_samples(walkers)
                sampler.energies.add_samples(E_target)
        return sampler