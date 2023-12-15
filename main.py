#!/usr/bin/env python3

# TODO: does hiclass crash with G0/G very different from 1 with small ω?
# TODO: GR emulators (?): Bacco, CosmicEmu, EuclidEmulator2, references within
# TODO: compare P(k) with fig. 2 on https://journals.aps.org/prd/pdf/10.1103/PhysRevD.97.023520#page=13
# TODO: emulate B / Bfid ≈ 1?
# TODO: run one big box with COLA (fiducial cosmology?) to see if pattern continues to higher k?
# TODO: run one big box with "proper N-body program" to see if COLA is ok
# TODO: emulation https://github.com/renmau/Sesame_pipeline/
# TODO: compute P(k) from COLA *snapshots*
# TODO: why do sims load multiple times? because simulation is reconstructed for each redshift!
# TODO: smooth B(k) using Savitzky-Golay filter (https://en.wikipedia.org/wiki/Savitzky%E2%80%93Golay_filter)?
# TODO: same "framework" for emulator source as for class/cola/ramses source
# TODO: emulate non-linear correction boost factor
# TODO: plot redshift evolution of modes?
# TODO: use euclid parameter bounds
# TODO: emulate P(k,z) for 0 <= z <= 3 and k <= ???
# TODO: need 1% accuracy for k = 1 Mpc/h ?
# TODO: don't need to re-run with new As when correcting σ8; can simply multiply up the power spectrum of the old run
# TODO: can decrease L instead of increasing N to remove shot noise

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
parser.add_argument("--divide", metavar="SOURCE", default="", help="source to divide by")
parser.add_argument("--subtract-shotnoise", action="store_true", help="subtract shot noise")
parser.add_argument("--realizations", metavar="N", type=int, default=1, help="number of universe realizations to simulate per model")
parser.add_argument("--evolution", action="store_true", help="plot evolution of background and perturbation quantities")
parser.add_argument("--samples", metavar="N", type=int, default=0, help="number of latin hypercube samples to make")
parser.add_argument("--parameter-space", action="store_true", help="plot (varying) parameter space")
parser.add_argument("--test", action="store_true", help="run whatever experimental code is in the test section")
args = parser.parse_args()

class ParameterSpace:
    def __init__(self, params):
        self.params = params

    def combinations(self):
        def traverse(paramlist, params={}):
            if len(paramlist) == 0:
                yield params.copy()
            else:
                param, values = paramlist.popitem() # 1) remove one parameter from list of varying parameters
                for value in values:
                    params[param] = value # 2) add it to current parameter map
                    yield from traverse(paramlist, params)
                paramlist[param] = values # 3) undo 1)
        return list(traverse(self.params))

    def sample(self, n=1, seed=1234, bounds=False):
        paramss = [] # final list of parameter samples to return

        # 1) separate fixed and varying parameters
        fixed, varying = {}, {}
        for param, vals in self.params.items():
            if len(vals) == 1:
                fixed[param] = vals[0]
            elif len(vals) > 1:
                varying[param] = (min(vals), max(vals))
            else:
                raise(f"Unspecified parameter {param}")

        # 2) sample the varying parameters
        dim_varying = len(varying)
        bounds_varying_lo = [minmax[0] for minmax in varying.values()]
        bounds_varying_hi = [minmax[1] for minmax in varying.values()]
        sampler = qmc.LatinHypercube(dim_varying, seed=seed)
        while n > 0:
            params = sampler.random()[0] # list of random numbers in [0,1) for each varying parameter
            for i, (lo, hi) in enumerate(zip(bounds_varying_lo, bounds_varying_hi)): # Python dict iteration order is guaranteed to be in same order as insertion
                params[i] = lo if hi == lo else lo + (hi-lo) * params[i] # map to [lo, hi); or fixed to lo == hi if they are equal (handle this separately to preserve data type)
            params = dict([(param, value) for param, value in zip(varying.keys(), params)]) # convert to dictionary (e.g. from [0.67, 2.1e-9] to {"h": 0.67, "As": 2.1e-9})
            paramss.append(params)
            n -= 1

        # 3) add the fixed parameters
        for i in range(0, len(paramss)):
            paramss[i] |= fixed
        else:
            return paramss

    def bounds(self):
        lo = {param: min(values) for param, values in self.params.items()}
        hi = {param: max(values) for param, values in self.params.items()}
        return lo, hi

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
    "Npart":     {"fid": 512,          "help": "number of particles per dimension (all N-body simulations)"},
    "Ncell":     {"fid": 512,          "help": "number of coarse cells per dimension for (all N-body simulations)"},
    "Lh":        {"fid": 384.0,        "help": "comoving box size / (Mpc/h) (all N-body simulations) = L*h"},
    "L":         {"fid": 384.0 / 0.70, "help": "comoving box size / Mpc (all N-body simulations)"},

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
# TODO: also generate params0 (e.g. --params σ8=0.8 σ8=0.7,0.8,0.9 will use fiducial 0.8 and vary 0.7,0.8,0.9)
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

pspace = ParameterSpace(paramlist)
if args.samples == 0:
    paramss = pspace.combinations()
else:
    paramss = pspace.sample(args.samples, bounds=False)

lo, hi = pspace.bounds()

varparams = [param for param, vals in paramlist.items() if len(vals)  > 1] # list of varying parameters
fixparams = [param for param, vals in paramlist.items() if len(vals) == 1] # list of fixed   paramaters

print("Fixed parameters:")
for param in fixparams:
    print(f"{param} = {paramlist[param][0]}")

print()
print("Varying parameters:")
widths = {param: max([len(str(params[param])) for params in paramss]) for param in varparams}
print(" ".join(f"{param: <{widths[param]}}" for param in varparams))
print(" ".join("-" * widths[param] for param in varparams))
for params in paramss:
    print(" ".join(f"{value: <{widths[param]}}" for param, value in params.items() if param in varparams))

if args.parameter_space:
    plot.plot_parameter_samples("plots/parameter_space.pdf", paramss, lo, hi)

# Parameter transformation from BD to GR
θGR = θGR_different_h if args.transform_h else θGR_identity

# Plot power spectra and boost, if requested
if len(args.power) > 0:
    assert len(varparams) <= 1, "can vary at most one parameter at the time"
    varparam = varparams[0] if len(varparams) == 1 else None
    fixparams_nondefault = sorted(list(set(fixparams) - set(fixparams_default)))
    stem = "plots/power_fix_" + '_'.join(fixparams_nondefault) + (f"_vary_{varparam}" if varparam else "")
    sources = args.power
    params0 = {param: PARAMS[param]["fid"] for param in PARAMS}
    plot.plot_power(stem, params0, paramss, varparam, θGR, nsims=args.realizations, sources=sources, hunits=args.h_units, divide=args.divide, subshot=args.subtract_shotnoise)

# Plot evolution of (background) densities
if args.evolution:
    assert len(paramss) == 1, "will only plot evolution for one set of parameters"
    params = paramss[0]
    plot.plot_density_evolution("plots/evolution_density.pdf", params, θGR)

    def q_from_file(sim, filename, f):
        data = sim.read_data(filename, dict=True)
        return f(data)

    def G_G0_BD(sim):
        return q_from_file(sim, "class/background.dat", lambda bg: (1 / (bg["z"] + 1), (4+2*sim.params["ω"]) / (3+2*sim.params["ω"]) / bg["phi_smg"]))

    def G_G0_GR(sim):
        return q_from_file(sim, "class/background.dat", lambda bg: (1 / (bg["z"] + 1), np.ones_like(bg["z"])))

    def H_H0_BD_GR(sim):
        return q_from_file(sim, "class/background.dat", lambda bg: (1 / (bg["z"] + 1), bg["H [1/Mpc]"] * 2997))

    def D_Di_BD_GR(sim):
        return q_from_file(sim, "cola/gravitymodel_cola_k1.0.txt", lambda data: (data["a"], data["D1(a,k)"] / data["D1(a,k)"][0]))

    def f_BD_GR(sim):
        a, D = q_from_file(sim, "cola/gravitymodel_cola_k1.0.txt", lambda data: (data["a"], data["D1(a,k)"] / data["D1(a,k)"][0]))
        a = a[0:len(a):30] # avoid numerical noise
        D = D[0:len(D):30]
        return a, np.gradient(np.log(D), np.log(a))

    series = [
        ("G", G_G0_BD,    G_G0_GR,    False, "G(a)/G_0",         0.05, 0.05),
        ("H", H_H0_BD_GR, H_H0_BD_GR, True,  "H(a)/(100\,\mathrm{km}/\mathrm{s}\,\mathrm{Mpc})",         5.0,  0.01),
        ("D", D_Di_BD_GR, D_Di_BD_GR, True,  "D(a)/D(10^{-10})", 1.0,  0.1),
        ("f", f_BD_GR,    f_BD_GR,    False, "f(a)",             0.1,  0.01),
    ]
    for q, qBD, qGR, logabs, ylabel, Δyabs, Δyrel in series:
        plot.plot_quantity_evolution(f"plots/evolution_{q}.pdf", params, qBD, qGR, θGR, qty=q, ylabel=ylabel, logabs=logabs, Δyabs=Δyabs, Δyrel=Δyrel)
exit()

# use this for testing shit
if args.test:
    pass
exit()
