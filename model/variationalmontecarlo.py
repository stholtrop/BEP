from model.samplers import *
from model.montecarlo import MonteCarloMethod

import numpy as np

class VariationalMonteCarlo(MonteCarloMethod):
    def __init__(self, function, transition, initial_state):
        self.function = function
        self.transition = transition
        self.initial_state = initial_state
    
    def generate_samples(self, N, epochs, skipped=0, sampler=None):
        if sampler == None: sampler = ListSampler()
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

    def generate_samples(self, N, epochs, skipped=0, sampler=None):
        if sampler == None: sampler = ListSampler()
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

    def greens_function(self, walkers, new_walkers):
        return np.exp(-np.sum((new_walkers - walkers - self.delta_t/2*self.force(walkers))**2)/2/self.delta_t)/(2*np.pi*self.delta_t)**(3*np.sum(walkers[0].shape)/2)

    def generate_samples(self, N, epochs, skipped=0, sampler=None):
        if sampler == None: sampler = ListSampler()
        walkers = self.initial_state(N)
        for i in range(epochs):
            walkers += self.force(walkers)*self.delta_t/2 
            vals = self.function(walkers)
            new_walkers = walkers + np.random.normal(0, 1, walkers.shape)*np.sqrt(self.delta_t)
            new_vals = self.function(new_walkers)
            selection = self.greens_function(new_walkers, walkers)/self.greens_function(walkers, new_walkers)*new_vals/vals > np.random.rand(N)
            walkers = np.concatenate((new_walkers[selection], walkers[~selection]))
            if i >= skipped:
                sampler.add_samples(np.array(walkers))
        return sampler