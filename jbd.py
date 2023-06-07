#!/usr/bin/env python3

import numpy as np
from scipy.stats import qmc

class ParameterSpace:
    def __init__(self, params):
        self.param_names = list(params.keys())
        self.param_limits_lo = [bounds[0] for bounds in params.values()]
        self.param_limits_hi = [bounds[1] for bounds in params.values()]
        self.dimension = len(self.param_names)

    def sample(self, n, seed=1234):
        sampler = qmc.LatinHypercube(self.dimension, seed=seed)
        samples = sampler.random(n=n) # in [0, 1]
        samples = qmc.scale(samples, self.param_limits_lo, self.param_limits_hi) # in [lo, hi]
        return samples

class Simulation:
    def __init__(self):
        pass

#if __name__ == "__main__":
Pdict = {
    "A_s": (1e-9, 4e-9),
    "Ωc0": (0.15, 0.35),
    "μ0": (-1.0, +1.0),
}
P = ParameterSpace(Pdict)
print(P.sample(5))
