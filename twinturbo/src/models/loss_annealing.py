import numpy as np 

class SinAnnealing():
    def __init__(self, amplitude, period, start = None, end = None, min_val = 0):
        self.amplitude = amplitude
        self.start = start
        self.end = end
        self.period = period
        self.min_val = min_val
        
    def __call__(self, batch):
        if (self.start is not None) and batch < self.start:
            return self.min_val
        elif (self.end is not None) and batch > self.end:
            return self.min_val + self.amplitude * np.sin(np.pi/self.period * (self.end - self.start)) ** 2
        else:
            return self.min_val + self.amplitude * np.sin(np.pi/self.period * (batch - self.start)) ** 2

class SigmoidAnnealing():
    def __init__(self, amplitude = None, start = None, end = None, min_val = 0, steepness=5):
        
        self.start = start
        self.end = end
        self.min_val = min_val
        self.slope = 10/(end - start)
        self.steepness = steepness
        self.amplitude = amplitude
        self.middle = (start + end) / 2
        
    def __call__(self, batch):
        if (self.start is not None) and batch < self.start:
            return self.min_val
        elif (self.end is not None) and batch > self.end:
            return self.min_val + self.amplitude / (1 + np.exp(-self.slope * (self.end - self.middle)))
        else:
            return self.min_val + self.amplitude / (1 + np.exp(-self.slope * (batch - self.middle))) 