#!/usr/bin/env python3

# TODO: does hiclass crash with G0/G very different from 1 with small ω?
# TODO: GR emulators (?): Bacco, CosmicEmu, EuclidEmulator2, references within
# TODO: compare P(k) with fig. 2 on https://journals.aps.org/prd/pdf/10.1103/PhysRevD.97.023520#page=13
# TODO: emulate B / Bfid ≈ 1?
# TODO: run one big box with COLA (fiducial cosmology?) to see if pattern continues to higher k?
# TODO: run one big box with "proper N-body program" to see if COLA is ok
# TODO: emulation https://github.com/renmau/Sesame_pipeline/
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
parser.add_argument("--params", metavar="PARAM[=VALUES]", nargs="*", help="parameters to fix or vary", default=[])
parser.add_argument("--transform-h", action="store_true", help="use hGR = hBD * √(ϕini) instead of hGR = hBD")
parser.add_argument("--power", nargs="*", metavar="SOURCE", default=[], help="plot P(k) and B(k) from sources (class, halofit, cola, ramses)")
parser.add_argument("--h-units", action="store_true", help="plot power and boost with P(k/h)*h^3 instead of P(k)")
parser.add_argument("--divide-linear", action="store_true", help="divide by linear prediction")
parser.add_argument("--realizations", metavar="N", type=int, default=1, help="number of universe realizations to simulate per model")
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
    return utils.dictupdate(θBD, remove=["ω", "G0"]) # remove BD-specific parameters

def θGR_different_h(θBD, θBD_all):
    θGR = θGR_identity(θBD, θBD_all)
    θGR["h"] = θBD["h"] * np.sqrt(θBD_all["ϕini"]) # ensure similar Hubble evolution (of E=H/H0) during radiation domination
    return θGR

# All available parameters and their fiducial/default values
PARAMS = {
    "h":         {"fid": 0.70,         "help": "reduced Hubble parameter today = H0 / (100 km/(s*Mpc))"},
    "h*√(φini)": {"fid": 0.70,         "help": "TODO"}, # TODO: handle! fixing h^2 ϕini
    "ωb0":       {"fid": 0.02,         "help": "physical baryon density = ρb0 / (3 * (100 km/(s*Mpc))^2 / (8*π*G))"},
    "ωc0":       {"fid": 0.13,         "help": "physical cold dark matter density = ρc0 / (3 * (100 km/(s*Mpc))^2 / (8*π*G))"},
    "ωm0":       {"fid": 0.15,         "help": "physical matter density = ωb0 + ωc0"},
    "ωk0":       {"fid": 0.00,         "help": "effective curvature density (not handled)"}, # TODO: cannot handle this?
    "Tγ0":       {"fid": 2.7255,       "help": "CMB photon temperature today / K"},
    "Neff":      {"fid": 3.00,         "help": "effective number of neutrino species"},
    "As":        {"fid": 2e-9,         "help": "primordial power spectrum amplitude = As"},
    "ns":        {"fid": 1.00,         "help": "primordial power spectrum spectral index"},
    "kpivot":    {"fid": 0.05,         "help": "wavenumber at which primordial power spectrum amplitude is given"},
    "σ8":        {"fid": 0.80,         "help": "smoothed matter density fluctuation amplitude = σ(R=8 Mpc/h, z=0)"},

    "ω":         {"fid": 100.0,        "help": "Brans-Dicke scalar field coupling"},
    "G0":        {"fid": 1.00,         "help": "gravitational parameter today / G"},

    "zinit":     {"fid": 10.0,         "help": "initial redshift (all N-body simulations)"},
    "Nstep":     {"fid": 30,           "help": "number of timesteps (COLA N-body simulations"},
    "Npart":     {"fid": 256,          "help": "number of particles per dimension (all N-body simulations)"},
    "Ncell":     {"fid": 256,          "help": "number of coarse cells per dimension for (all N-body simulations)"},
    "Lh":        {"fid": 400.0,        "help": "comoving box size / (Mpc/h) (all N-body simulations) = L*h"},
    "L":         {"fid": 400.0 / 0.70, "help": "comoving box size / Mpc (all N-body simulations)"},

    "z":         {"fid": 0.0,          "help": "power spectrum redshift"}, # handled specially
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
        print(f"{param} = {PARAMS[param]['fid']} ({PARAMS[param]['help']})")
    exit()

# Build BD parameters
paramlist = {}
fixparams_default = ["h", "ωk0", "Tγ0", "Neff", "ns", "kpivot", "ω", "G0", "zinit", "Nstep", "Npart", "Ncell", "Lh"]
for param in fixparams_default: # fix these by default
    paramlist[param] = [PARAMS[param]["fid"]] # fix to fiducial value
for param in args.params:
    # parse param on the form "param" or fix == "param=value"
    param_value = param.split('=')
    param = param_value[0]
    values = [int(value) if type(PARAMS[param]["fid"]) == int else float(value) for value in param_value[1].split(',')] if len(param_value) == 2 else [PARAMS[param]["fid"]] # specified or fiducial
    paramlist[param] = values
print("Parameters:")
for param, values in paramlist.items():
    print(f"{param} = {values} (x{len(values)})")

def genparams(paramlist, params={}):
    if len(paramlist) == 0:
        yield params.copy()
    else:
        param, values = paramlist.popitem() # 1) remove one parameter from list of varying parameters
        for value in values:
            params[param] = value # 2) add it to current parameter map
            yield from genparams(paramlist, params)
        paramlist[param] = values # 3) undo 1)

print("Iterating over", 'x'.join(str(len(values)) for _, values in paramlist.items() if len(values) > 1), "parameter combinations")
paramss = list(genparams(paramlist))
varparams = [param for param, vals in paramlist.items() if len(vals)  > 1] # list of varying parameters
fixparams = [param for param, vals in paramlist.items() if len(vals) == 1] # list of fixed   paramaters

# Parameter transformation from BD to GR
θGR = θGR_different_h if args.transform_h else θGR_identity

# Plot power spectra and boost, if requested
if len(args.power) > 0:
    assert len(varparams) <= 1, "can vary at most one parameter at the time"
    varparam = varparams[0] if len(varparams) == 1 else None
    fixparams_nondefault = list(set(fixparams) - set(fixparams_default))
    stem = "plots/power_fix_" + '_'.join(fixparams_nondefault) + (f"_vary_{varparam}" if varparam else "")
    sources = args.power
    params0 = {param: PARAMS[param]["fid"] for param in PARAMS}
    plot.plot_power(stem, params0, paramss, varparam, θGR, nsims=args.realizations, sources=sources, hunits=args.h_units, divlin=args.divide_linear)

# Plot evolution of (background) densities
if args.evolution:
    assert len(paramss) == 1, "will only plot evolution for one set of parameters"
    params = paramss[0]
    plot.plot_density_evolution("plots/evolution_density.pdf", params, θGR)

    # Plot evolution of (background) quantities
    def G_G0_BD(bg, params):    return (4+2*params["ω"]) / (3+2*params["ω"]) / bg["phi_smg"]
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
        plot.plot_quantity_evolution(f"plots/evolution_{q}.pdf", params, qBD, qGR, θGR, qty=q, ylabel=ylabel, logabs=logabs, Δyabs=Δyabs, Δyrel=Δyrel)
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
