#!/usr/bin/env python3

# TODO: does hiclass crash with G0/G very different from 1 with small ω?
# TODO: GR emulators (?): Bacco, CosmicEmu, EuclidEmulator2, references within
# TODO: compare P(k) with fig. 2 on https://journals.aps.org/prd/pdf/10.1103/PhysRevD.97.023520#page=13
# TODO: emulate B / Bfid ≈ 1?
# TODO: run one big box with COLA (fiducial cosmology?) to see if pattern continues to higher k?
# TODO: run one big box with "proper N-body program" to see if COLA is ok
# TODO: allow for "main.py vary z fix σ8 syntax, or similar

import sim
import plot
import utils

import os
import argparse
import numpy as np
from scipy.stats import qmc
from scipy.interpolate import CubicSpline

parser = argparse.ArgumentParser(prog="jbd.py")
parser.add_argument("action", help="action(s) to execute", nargs="+")
args = parser.parse_args()
ACTIONS = vars(args)["action"]

def list_simulations():
    for path in os.scandir(sim.SIMDIR):
        if os.path.isdir(path):
            sim.Simulation(path=path.name, verbose=True, run=False)

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

# TODO: drop in favor of "equal h^2 ϕini" etc.
def θGR_identity(θBD, θBD_all):
    return utils.dictupdate(θBD, remove=["lgω", "G0/G"]) # remove BD-specific parameters

def θGR_different_h(θBD, θBD_all):
    θGR = θGR_identity(θBD, θBD_all)
    θGR["h"] = θBD["h"] * np.sqrt(θBD_all["ϕini"]) # ensure similar Hubble evolution (of E=H/H0) during radiation domination
    return θGR

# Fiducial parameters
params0_GR = {
    # physical parameters
    "h":      0.68,   # class' default
    "ωb0":    0.022,  # class' default (ωb0 = Ωs0*h0^2*ϕ0 ∝ ρb0 in BD, ωb0 = Ωs0*h0^2 ∝ ρb0 in GR)
    "ωc0":    0.120,  # class' default (ωc0 = Ωs0*h0^2*ϕ0 ∝ ρc0 in BD, ωc0 = Ωs0*h0^2 ∝ ρc0 in GR)
    "ωk0":    0.0,    # class' default
    "Tγ0":    2.7255, # class' default
    "Neff":   3.044,  # class' default # TODO: handled correctly in COLA?
    "kpivot": 0.05,   # class' default
    "Ase9":   2.1,    # class' default
    "ns":     0.966,  # class' default

    # computational parameters (max on euclid22-32: Npart=Ncell=1024 with np=16 CPUs)
    "zinit": 10.0,
    "Nstep": 30,
    "Npart": 512,
    "Ncell": 512,
    "Lh":    512.0, # L / (Mpc/h) = L*h / Mpc
}
params0_BD = utils.dictupdate(params0_GR, {
    "lgω":    2.0,    # lowest value to consider (larger values should only be "easier" to simulate?)
    "G0/G":   1.0,    # G0 == G        (ϕ0 = (4+2*ω)/(3+2*ω) * 1/(G0/G))
})

# Plot LHS samples seen through each parameter space face
params_varying = {
    "lgω":    (2.0, 5.0),
    "G0/G":   (0.99, 1.01),
    "h":      (0.63, 0.73),
    "ωb0":    (0.016, 0.028),
    "ωc0":    (0.090, 0.150),
    "Ase9":   (1.6, 2.6),
    "ns":     (0.866, 1.066),
}

# List simulations
if "list" in ACTIONS:
    list_simulations()

if "rcparams" in ACTIONS:
    print("Matplotlib rcParams:")
    print(matplotlib.rcParams.keys())

# Plot evolution of (background) densities
if "evolution" in ACTIONS:
    plot.plot_density_evolution("plots/evolution_density.pdf", params0_BD, θGR_different_h)

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
        plot.plot_quantity_evolution(f"plots/evolution_{q}.pdf", params0_BD, qBD, qGR, θGR_different_h, qty=q, ylabel=ylabel, logabs=logabs, Δyabs=Δyabs, Δyrel=Δyrel)

if "test" in ACTIONS:
    #plot.plot_convergence(f"plots/boost_fiducial.pdf", params0_BD, "lgω", [2.0], nsims=5, θGR=θGR_different_h)
    params0 = utils.dictupdate(params0_BD, {"σ8": 0.6, "Nstep": 10, "Npart": 16, "Ncell": 16}, ["Ase9"])
    #plot.plot_convergence(f"plots/boost_test.pdf", params0, "lgω", [2.0], nsims=1, θGR=θGR_different_h)
    plot.plot_convergence(f"plots/boost_test.pdf", params0, "z", [0.0, 1.0, 2.0, 3.0], nsims=1, θGR=θGR_different_h)
    #plot.plot_convergence(f"plots/boost_test.pdf",   params0_BD, "lgω",  [2.0, 3.0, 4.0],  θGR_different_h)
    exit()

if "compare" in ACTIONS:
    for θGR, suffix1 in zip([θGR_identity, θGR_different_h], ["_sameh", "_diffh"]):
        for divide_linear, suffix2 in zip([False, True], ["", "_divlin"]):
            for logy, suffix3 in zip([False, True], ["", "_log"]):
                plot.plot_power(f"plots/fiducial{suffix1}{suffix2}{suffix3}", params0_BD, "lgω", [2.0, 3.0, 4.0, 5.0], nsims=5, θGR=θGR, divide_linear=divide_linear, logy=logy)

# Convergence plots (computational parameters)
if "convergence" in ACTIONS:
    plot.plot_power("plots/convergence_L",     params0_BD, "Lh",     [256.0, 384.0, 512.0, 768.0, 1024.0], θGR_different_h)
    plot.plot_power("plots/convergence_Npart", params0_BD, "Npart",  [256, 384, 512, 768, 1024],           θGR_different_h)
    plot.plot_power("plots/convergence_Ncell", params0_BD, "Ncell",  [256, 384, 512, 768, 1024],           θGR_different_h)
    plot.plot_power("plots/convergence_Nstep", params0_BD, "Nstep",  [10, 20, 30, 40, 50],                 θGR_different_h)
    plot.plot_power("plots/convergence_zinit", params0_BD, "zinit",  [10.0, 20.0, 30.0],                   θGR_different_h)

# Variation plots (cosmological parameters)
if "variation" in ACTIONS:
    for param, value, prefix in (("Ase9", 2.1, "parametrize_As"), ("σ8", 0.8, "parametrize_s8", )):
        params0 = utils.dictupdate(params0_BD, {param: value}, remove=["Ase9"])
        if True:
            plot.plot_power(f"plots/variation_{prefix}_vary_z",       params0, "z",    [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0],  θGR_different_h)
            plot.plot_power(f"plots/variation_{prefix}_vary_omega",   params0, "lgω",  [2.0, 3.0, 4.0, 5.0],  θGR_different_h)
            plot.plot_power(f"plots/variation_{prefix}_vary_G0",      params0, "G0/G", [0.99, 1.0, 1.01],     θGR_different_h)
            plot.plot_power(f"plots/variation_{prefix}_vary_h",       params0, "h",    [0.63, 0.68, 0.73],    θGR_different_h)
            plot.plot_power(f"plots/variation_{prefix}_vary_omegab0", params0, "ωb0",  [0.016, 0.022, 0.028], θGR_different_h)
            plot.plot_power(f"plots/variation_{prefix}_vary_omegac0", params0, "ωc0",  [0.100, 0.120, 0.140], θGR_different_h)
            plot.plot_power(f"plots/variation_{prefix}_vary_ns",      params0, "ns",   [0.866, 0.966, 1.066], θGR_different_h)
        if "Ase9" in params0:
            plot.plot_power(f"plots/variation_{prefix}_vary_As",      params0, "Ase9", [1.6, 2.1, 2.6],       θGR_different_h)
        if "σ8" in params0:
            plot.plot_power(f"plots/variation_{prefix}_vary_s8",      params0, "σ8",   [0.7, 0.8, 0.9],       θGR_different_h)

        # parametrize with ωm0 and ωb0 (instead of ωc0)
        params0 = utils.dictupdate(params0, remove=["ωc0", "ωb0"])
        plot.plot_power(f"plots/variation_{prefix}_omegam0_vary_omegab0", params0 | {"ωm0": 0.142, "ωb0": 0.022}, "ωb0", [0.016, 0.022, 0.028], θGR_different_h)
        plot.plot_power(f"plots/variation_{prefix}_omegam0_vary_omegac0", params0 | {"ωm0": 0.142, "ωc0": 0.120}, "ωc0", [0.100, 0.120, 0.140], θGR_different_h)
        plot.plot_power(f"plots/variation_{prefix}_omegab0_vary_omegam0", params0 | {"ωm0": 0.142, "ωb0": 0.022}, "ωm0", [0.082, 0.142, 0.202], θGR_different_h)
        #plot.plot_power(f"plots/variation_omegam0_fixed_omegac0", params0 | {"ωm0": 0.142, "ωc0": 0.120}, "ωm0", [0.082, 0.142, 0.202], θGR_different_h) # TODO: need to update values to not get negative mass density

if "sample" in ACTIONS:
    paramspace = ParameterSpace(params_varying)
    samples = paramspace.samples(500)
    plot.plot_parameter_samples("plots/parameter_samples.pdf", samples, paramspace.bounds_lo(), paramspace.bounds_hi(), paramspace.param_names)
