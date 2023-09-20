#!/usr/bin/env python3

# TODO: does hiclass crash with G0/G very different from 1 with small ω?
# TODO: GR emulators (?): Bacco, CosmicEmu, EuclidEmulator2, references within
# TODO: compare P(k) with fig. 2 on https://journals.aps.org/prd/pdf/10.1103/PhysRevD.97.023520#page=13
# TODO: emulate B / Bfid ≈ 1?
# TODO: run one big box with COLA (fiducial cosmology?) to see if pattern continues to higher k?
# TODO: run one big box with "proper N-body program" to see if COLA is ok
# TODO: emulation https://github.com/renmau/Sesame_pipeline/
# TODO: subtract shotnoise
# TODO: compute AMR of 256 grid on a 4*256 = 1024 grid
# TODO: compute P(k) from COLA *snapshots*

import sim
import plot
import utils

import os
import argparse
import numpy as np
from scipy.stats import qmc
from scipy.interpolate import CubicSpline

parser = argparse.ArgumentParser(prog="main.py")
parser.add_argument("--list-sims", action="store_true", help="list simulations and exit")
parser.add_argument("--list-params", action="store_true", help="list parameters and exit")
parser.add_argument("--fix", metavar="PARAM[=VALUE]", nargs="*", help="parameters to fix", default=[])
parser.add_argument("--vary", metavar="PARAM=VALUE1,VALUE2,...,VALUEN", nargs="?", help="parameter to vary")
parser.add_argument("--sources", metavar="SOURCE", default=["linear-class"], nargs="*", help="P(k) sources (linear-class, nonlinear-class, nonlinear-cola, nonlinear-ramses)")
parser.add_argument("--power", action="store_true", help="plot power spectra and boost")
parser.add_argument("--evolution", action="store_true", help="plot evolution of background and perturbation quantities")
#parser.add_argument("--sample") # TODO: sampling, emulation, ...
parser.add_argument("--test", action="store_true", help="run whatever experimental code is in the test section")
args = parser.parse_args()

class ParameterSpace:
    def __init__(self, params, seed=1234):
        self.param_names = list(params.keys())
        self.param_bounds = [val if isinstance(val, tuple) else (val, val) for val in params.values()]
        self.param_bounds_lo = [bounds[0] for bounds in self.param_bounds]
        self.param_bounds_hi = [bounds[1] for bounds in self.param_bounds]
        self.dimension = len(self.param_names)
        self.sampler = qmc.LatinHypercube(self.dimension, seed=seed)

    def bounds_lo(self): return dict([(name, lo) for name, lo in zip(self.param_names, self.param_bounds_lo)])
    def bounds_hi(self): return dict([(name, hi) for name, hi in zip(self.param_names, self.param_bounds_hi)])

    def sample(self):
        samples = self.sampler.random()[0] # in [0,1)
        for i, (lo, hi) in enumerate(zip(self.param_bounds_lo, self.param_bounds_hi)):
            samples[i] = lo if hi == lo else lo + (hi-lo) * samples[i] # in [lo, hi); or fixed to lo == hi if they are equal (handle this separately to preserve data type)
        samples = dict([(name, sample) for name, sample in zip(self.param_names, samples)]) # e.g. from [0.67, 2.1e-9] to {"h": 0.67, "As": 2.1e-9}
        return samples

    def samples(self, n):
        return [self.sample() for i in range(0, n)]

def θGR_identity(θBD, θBD_all):
    return utils.dictupdate(θBD, remove=["lgω", "G0/G"]) # remove BD-specific parameters

def θGR_different_h(θBD, θBD_all):
    θGR = θGR_identity(θBD, θBD_all)
    θGR["h"] = θBD["h"] * np.sqrt(θBD_all["ϕini"]) # ensure similar Hubble evolution (of E=H/H0) during radiation domination
    return θGR

# All available parameters and their fiducial/default values
# TODO: add explanation/help for each parameter
PARAMS = {
    "h":         0.70,
    "h*√(φini)": 0.70, # TODO: handle! fixing h^2 ϕini
    "ωb0":       0.02,
    "ωc0":       0.13,
    "ωm0":       0.15,
    "ωk0":       0.00, # TODO: cannot handle this?
    "Tγ0":       2.7255,
    "Neff":      3.00,
    "Ase9":      2.00,
    "ns":        1.00,
    "kpivot":    0.05,
    "σ8":        0.80,

    "lgω":       2.0,
    "G0/G":      1.00,

    "z":         0.0, # handled specially

    "zinit":     10.0,
    "Nstep":     30,
    "Npart":     256,
    "Ncell":     256,
    "Lh":        400.0,
    "L":         400.0 / 0.70,
}

# List simulations, if requested
if args.list_sims:
    for simtype in sim.SIMTYPES:
        print(f"Simulations in {simtype.SIMDIR}:")
        simtype.list()
    exit()

if args.list_params:
    print("Available independent parameters and their default values:")
    for param in PARAMS:
        print(f"{param} = {PARAMS[param]}")
    exit()

# Build fixed parameters # TODO: just do --params for both fixing and varying
params = {}
for param in ["h", "ωb0", "ωm0", "ωk0", "Tγ0", "Neff", "ns", "kpivot", "lgω", "G0/G", "zinit", "Nstep", "Npart", "Ncell", "Lh"]: # fix these by default
    params[param] = PARAMS[param] # fix to fiducial value
for fix in args.fix:
    # parse fix == "param" or fix == "param=value"
    param_value = fix.split('=')
    param = param_value[0]
    value = float(param_value[1]) if len(param_value) == 2 else PARAMS[param] # specified or fiducial
    params[param] = value
print("Fixing parameters:")
for param, value in params.items():
    print(f"{param} = {value}")

# Vary parameters, if requested # TODO: make a "generator" for simulations
if args.vary:
    # parse args.vary == "param=value1,value2,...valueN"
    param_values = args.vary.split('=')
    assert len(param_values) == 2, "varying parameter values were not specified"
    param = param_values[0]
    values = [float(value) for value in param_values[1].split(',')]
    sources = args.sources
    stem = "plots/fix_" + '_'.join(args.fix) + f"_vary_{param}"
    plot.plot_power(stem, params, param, values, θGR_different_h, nsims=1, sources=sources)

# Plot evolution of (background) densities
if args.evolution:
    plot.plot_density_evolution("plots/evolution_density.pdf", params, θGR_different_h)

    # Plot evolution of (background) quantities
    def G_G0_BD(bg, params):    return (4+2*10**params["lgω"]) / (3+2*10**params["lgω"]) / bg["phi_smg"]
    def G_G0_GR(bg, params):    return np.ones_like(bg["z"]) # = 1.0, special case for GR
    def H_H0_BD_GR(bg, params): return bg["H [1/Mpc]"] / CubicSpline(np.log10(1/(bg["z"]+1)), bg["H [1/Mpc]"])(0.0) # common to BD and GR
    def D_Di_BD_GR(bg, params): return bg["gr.fac. D"] / CubicSpline(np.log10(1/(bg["z"]+1)), bg["gr.fac. D"])(-10.0) # common to BD and GR
    def f_BD_GR(bg, params):    return bg["gr.fac. f"] # common to BD and GR
    series = [
        ("G", G_G0_BD,    G_G0_GR,    False, "G(a)/G",           0.05, 0.05),
        ("H", H_H0_BD_GR, H_H0_BD_GR, True,  "H(a)/H_0",         5.0,  0.01),
        ("D", D_Di_BD_GR, D_Di_BD_GR, True,  "D(a)",             1.0,  0.1),
        ("f", f_BD_GR,    f_BD_GR,    False, "f(a)",             0.1,  0.01),
    ]
    for q, qBD, qGR, logabs, ylabel, Δyabs, Δyrel in series:
        plot.plot_quantity_evolution(f"plots/evolution_{q}.pdf", params, qBD, qGR, θGR_different_h, qty=q, ylabel=ylabel, logabs=logabs, Δyabs=Δyabs, Δyrel=Δyrel)
exit()

# use this for testing shit
if args.test:
    pass
exit()

# TODO: unfinished shit
if args.sample:
    paramspace = ParameterSpace(params_varying)
    samples = paramspace.samples(500)
    plot.plot_parameter_samples("plots/parameter_samples.pdf", samples, paramspace.bounds_lo(), paramspace.bounds_hi(), paramspace.param_names)
