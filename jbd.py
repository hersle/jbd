#!/usr/bin/env python3

import numpy as np
from scipy.stats import qmc
from classy import Class

class ParameterSpace:
    def __init__(self, params):
        self.param_names = list(params.keys())
        self.param_bounds = [(val, val) if isinstance(val, float) else val for val in params.values()]
        self.param_bounds_lo = [bounds[0] for bounds in self.param_bounds]
        self.param_bounds_hi = [bounds[1] for bounds in self.param_bounds]
        self.dimension = len(self.param_names)

    def sample(self, n, seed=1234):
        sampler = qmc.LatinHypercube(self.dimension, seed=seed)
        samples = sampler.random(n=n) # in [0, 1]
        samples = qmc.scale(samples, self.param_bounds_lo, self.param_bounds_hi) # in [lo, hi]
        return samples # TODO: pack in dict with param names?

class Simulation:
    def __init__(self, params):
        self.run_class(params)

    def run_class(self, params):
        params["output"] = "mTk" # needed for Class to compute transfer function
        params["z_pk"] = 100 # redshift after which Class should compute transfer function
        self.cosmology = Class()
        self.cosmology.set(params)
        self.cosmology.compute()
        transfer = self.cosmology.get_transfer(z=10, output_format="camb")
        

#if __name__ == "__main__":
params0 = {
    # TODO: use unicode names
    "h":          0.67,
    "Omega_b":        0.05,
    "Omega_cdm":        0.267,
    "Omega_k":        0.0,
    "T_cmb":        2.7255,
    #"Omega_ncdm":      0.0012, # TODO: non-cold dark matter???
    #"Neff":       3.046,
    "k_pivot":    0.05,
    "A_s":        2.1e-9,
    "n_s":        0.965,
    #"w0":         -1.0, 
    #"wa":         0.0,
    #"mu0":        0.0,     # Only for Geff
    #"log10wBD":   3,       # Only for JBD
    #"log10fofr0": -5.0,    # Only for f(R)
    #"kmax_hmpc":  20.0,
    #"use_physical_parameters": False, " True: specify Ωs0*h^2, h is derived; False: specify Ωs0 and h, ΩΛ0 is derived
    #"cosmology_model": "w0waCDM",
    #"gravity_model": "Geff",
    #"omega_b":    0.05    * 0.67**2,
    #"omega_cdm":  0.267   * 0.67**2,
    #"omega_ncdm": 0.0012  * 0.67**2,
    #"omega_k":    0.0     * 0.67**2,
    #"omega_fld":  0.0     * 0.67**2, # Only needed for JBD
}
params_varying = {
    "A_s": (1e-9, 4e-9),
    "Ωc0": (0.15, 0.35),
    "μ0": (-1.0, +1.0),
}

paramspace = ParameterSpace(params_varying)
params = params0
sim = Simulation(params)
