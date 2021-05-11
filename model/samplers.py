import numpy as np

class Sampler:
    def add_samples(self, samples):
        pass

    def get_samples(self, samples):
        pass

class ListSampler(Sampler):
    def __init__(self, samples=np.array([])):
        self.samples = samples
        self.sample_list = [samples]

    
    def add_samples(self, samples):
        self.sample_list.append(samples)
    
    def get_samples(self):
        if len(self.sample_list) == 1: return self.samples
        self.samples = np.concatenate(self.sample_list)
        self.sample_list = [self.samples]
        return self.samples

class GridSampler(Sampler):
    pass