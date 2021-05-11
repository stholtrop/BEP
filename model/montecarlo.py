from model.samplers import *

import numpy as np

class MonteCarloMethod:
    """Abstract base class for specific implementations of Monte Carlo methods.
    """
    pass


class VariationalMonteCarlo(MonteCarloMethod):
    def __init__(self, function, transition, initial_state):
        self.function = function
        self.transition = transition
        self.initial_state = initial_state
    
    def generate_samples(self, N, epochs, skipped=0, sampler=ListSampler()):
        walkers = self.initial_state(N)
        vals = self.function(walkers)
        for i in range(epochs):
            new_walkers = self.transition(walkers)
            new_vals = self.function(new_walkers)
            selection = new_vals/vals > np.random.rand(N)
            walkers = np.concatenate((new_walkers[selection], walkers[~selection]))
            vals = self.function(walkers)
            if i >= skipped:
                sampler.add_samples(walkers)
        return sampler

class FokkerPlanckMonteCarlo(MonteCarloMethod):
    def __init__(self, initial_state, force, delta_t):
        self.initial_state = initial_state
        self.force = force
        self.delta_t = delta_t

    def generate_samples(self, N, epochs, skipped=0, sampler=ListSampler()):
        walkers = self.initial_state(N)
        for i in range(epochs):
            walkers += self.force(walkers)*self.delta_t/2 + np.random.normal(0, 1, walkers.shape)*np.sqrt(self.delta_t)
            if i >= skipped:
                sampler.add_samples(np.array(walkers))
        return sampler

class VariationalFokkerPlanckMonteCarlo(MonteCarloMethod):
    def __init__(self, function, initial_state, force, delta_t):
        self.initial_state = initial_state
        self.function = function
        self.force = force
        self.delta_t = delta_t

    def generate_samples(self, N, epochs, skipped=0, sampler=ListSampler()):
        walkers = self.initial_state(N)
        for i in range(epochs):
            walkers += self.force(walkers)*self.delta_t/2 
            vals = self.function(walkers)
            new_walkers = walkers + np.random.normal(0, 1, walkers.shape)*np.sqrt(self.delta_t)
            new_vals = self.function(new_walkers)
            selection = new_vals/vals > np.random.rand(N)
            walkers = np.concatenate((new_walkers[selection], walkers[~selection]))
            if i >= skipped:
                sampler.add_samples(np.array(walkers))
        return sampler