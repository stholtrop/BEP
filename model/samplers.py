import numpy as np

class Sampler:
    def add_samples(self, samples):
        pass

    def get_samples(self, samples):
        pass

class ListSampler(Sampler):
    def __init__(self):
        self.cache = None
        self.sample_list = []

    
    def add_samples(self, samples):
        self.sample_list.append(samples)
    
    def get_samples(self):
        if len(self.sample_list) == 0: return self.cache
        if self.cache is not None: self.add_samples(self.cache)
        self.cache = np.concatenate(self.sample_list)
        self.sample_list = []
        return self.cache

class JointSampler(Sampler):
    def __init__(self, samplers):
        self.samplers = samplers

    def __getattr__(self, name: str):
        return self.samplers[name]

class GridSampler(Sampler):
    pass