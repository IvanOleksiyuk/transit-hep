import numpy as np 

class Sin_annealing():
    def __init__(self, amplitude, period, start = None, end = None, min_val = 0):
        self.amplitude = amplitude
        self.start = start
        self.end = end
        self.period = period
        self.min_val = min_val
        
    def __call__(self, batch):
        if (self.start is not None) and batch < self.start:
            return 0
        elif (self.end is not None) and batch > self.end:
            return self.min_val + self.amplitude * np.sin(np.pi/self.period * (self.end - self.start)) ** 2
        else:
            return self.min_val + self.amplitude * np.sin(np.pi/self.period * (batch - self.start)) ** 2