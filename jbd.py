#!/usr/bin/env python3

# TODO: assert CLASS and COLA gives same field, H, etc.
# TODO: Omega_fld, Omega_Lambda, V0 fulfills same role by setting cosmo constant
# TODO: which G is hiclass' density parameters defined with respect to?
# TODO: look at PPN to understand cosmological (large) -> solar system (small) scales of G in JBD
# TODO: example plots, hi-class run: see /mn/stornext/u3/hansw/Herman/WorkingHiClass/plot.py
# TODO: Hans' FML JBD cosmology has not been tested with G/G != 1 !

import os
import re
import shutil
import argparse
import subprocess
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import qmc

parser = argparse.ArgumentParser(prog="jbd.py")
parser.add_argument("--FML", metavar="path/to/FML", default="./FML")
parser.add_argument("--hiclass", metavar="path/to/hiclass", default="./hi_class_public/class")
args = parser.parse_args()

COLAEXEC = os.path.abspath(os.path.expanduser(args.FML + "/FML/COLASolver/nbody"))
CLASSEXEC = os.path.abspath(os.path.expanduser(args.hiclass))

def luastr(var):
    if isinstance(var, bool):
        return str(var).lower() # Lua uses true, not True
    elif isinstance(var, str):
        return '"' + str(var) + '"' # enclose in ""
    elif isinstance(var, list):
        return "{" + ", ".join(luastr(el) for el in var) + "}" # Lua uses {} for lists
    else:
        return str(var) # go with python's string representation

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
        self.params = params
        self.name = self.name()
        self.directory = "sims/" + self.name + "/"

        # make simulation directory
        print(f"Simulating {self.name}")
        os.makedirs(self.directory, exist_ok=True)
        ks, Ps = self.run_class()
        self.params["h"] = self.h()
        self.params["ΩΛ0"] = self.ΩΛ0()
        self.run_cola(ks, Ps)
        assert self.completed()
        print(f"Simulated {self.name}")

    def name(self):
        return f"N{self.params['Npart']}"

    def completed_class(self): return os.path.isfile(self.directory + f"class_pk.dat")
    def completed_cola(self):  return os.path.isfile(self.directory + f"pofk_{self.name}_cb_z0.000.txt")
    def completed(self):       return self.completed_class() and self.completed_cola()

    def write_data(self, filename, cols, colnames=None):
        if isinstance(cols, dict):
            colnames = cols.keys()
            cols = [cols[colname] for colname in colnames]
            return self.write_data(filename, cols, colnames)

        header = None if colnames is None else " ".join(colnames)
        np.savetxt(self.directory + filename, np.transpose(cols), header=header)

    def write_file(self, filename, string):
        with open(self.directory + filename, "w") as file:
            file.write(string)

    def read_data(self, filename):
        data = np.loadtxt(self.directory + filename)
        data = np.transpose(data)
        return data

    def read_file(self, filename):
        with open(self.directory + filename, "r") as file:
            return file.read()

    def read_variable(self, filename, variable, between=" = "):
        pattern = variable + between + r"([-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?)" # e.g. "var = 1.2e-34"
        matches = re.findall(pattern, self.read_file(filename))
        assert len(matches) == 1
        return float(matches[0][0])

    # run a command in the simulation's directory
    # TODO: check/return exit status
    def run_command(self, cmd, log="/dev/null", verbose=True):
        teecmd = subprocess.Popen(["tee", log], stdin=subprocess.PIPE, stdout=None if verbose else subprocess.DEVNULL, cwd=self.directory)
        runcmd = subprocess.Popen(cmd, stdout=teecmd.stdin, stderr=subprocess.STDOUT, cwd=self.directory)

        runcmd.wait() # wait for command to finish
        teecmd.stdin.close() # close stream

    def params_class(self):
        return {
            # cosmological parameters
            # "h": self.params["h"],
            "Omega_b": self.params["Ωb0"],
            "Omega_cdm": self.params["Ωc0"],
            "Omega_k": self.params["Ωk0"],
            "T_cmb": self.params["Tγ0"],
            "N_eff": self.params["Neff"],
            "A_s": self.params["As"],
            "n_s": self.params["ns"],
            "k_pivot": self.params["kpivot"],

            # output control
            "output": "mPk",
            "root": "class_",

            # log verbosity (increase integers to make more talkative)
            "input_verbose": 10,
            "background_verbose": 10,
            "thermodynamics_verbose": 1,
            "perturbations_verbose": 1,
            "spectra_verbose": 1,
            "output_verbose": 1,
        }

    # TODO: use hi_class for generating JBD initial conditions?
    # TODO: which k-values to choose? see https://github.com/lesgourg/class_public/blob/aa92943e4ab86b56970953589b4897adf2bd0f99/explanatory.ini#L1102
    def run_class(self, input="class_input.ini", log="class.log"):
        if not self.completed_class():
            # write input and run class
            self.write_file(input, "\n".join(f"{param} = {str(val)}" for param, val in self.params_class().items()))
            self.run_command([CLASSEXEC, input], log=log, verbose=True)
        assert self.completed_class(), f"ERROR: see {log} for details"

        # get output power spectrum (COLA needs class' output power spectrum, just without comments)
        ks, Ps = self.read_data("class_pk.dat") # k / (h/Mpc); P / (Mpc/h)^3
        ks, Ps = ks * self.params["h"], Ps / self.params["h"]**3 # k / (1/Mpc); P / (Mpc)^3
        return ks, Ps

    def params_cola(self, seed=1234):
        return { # common parameters (for any derived simulation)
            "simulation_name": self.name,
            "simulation_boxsize": self.params["L"],
            "simulation_use_cola": True,
            "simulation_use_scaledependent_cola": False, # only relevant with massive neutrinos?

            "cosmology_Omegab": self.params["Ωb0"],
            "cosmology_OmegaCDM": self.params["Ωc0"],
            "cosmology_OmegaK": self.params["Ωk0"],
            "cosmology_OmegaLambda": self.params["ΩΛ0"],
            "cosmology_Neffective": self.params["Neff"],
            "cosmology_TCMB_kelvin": self.params["Tγ0"],
            "cosmology_As": self.params["As"],
            "cosmology_ns": self.params["ns"],
            "cosmology_kpivot_mpc": self.params["kpivot"],
            "cosmology_OmegaMNu": 0.0,

            "particle_Npart_1D": self.params["Npart"],

            "timestep_nsteps": [self.params["NT"]],

            "ic_random_seed": seed,
            "ic_initial_redshift": self.params["zinit"],
            "ic_nmesh" : self.params["Npart"],
            "ic_type_of_input": "powerspectrum", # transferinfofile only relevant with massive neutrinos?
            "ic_input_filename": "power_spectrum_today.dat",
            "ic_input_redshift": 0.0, # TODO: feed initial power spectrum directly instead of backscaling?

            "force_nmesh": self.params["Nmesh"],

            "output_folder": ".",
            "output_redshifts": [0.0],
        }

    def run_cola(self, ks, Ps, np=1, verbose=True, ic="power_spectrum_today.dat", input="cola_input.lua", log="cola.log"):
        if not self.completed_cola():
            self.write_data(ic, {"k/(h/Mpc)": ks/self.params["h"], "P/(Mpc/h)^3": Ps*self.params["h"]**3}) # COLA wants "h-units"
            self.write_file(input, "\n".join(f"{param} = {luastr(val)}" for param, val in self.params_cola().items()))
            cmd = ["mpirun", "-np", str(np), COLAEXEC, input] if np > 1 else [COLAEXEC, input]
            self.run_command(cmd, log=log, verbose=True)
        assert self.completed_cola(), f"ERROR: see {log} for details"

    def power_spectrum(self):
        assert self.completed()

        # pk: "total matter"
        # pk_cb: "cdm+b"
        # pk_lin: "total matter" (linear?)

        data = self.read_data(f"pofk_{self.name}_cb_z0.000.txt")
        k, P, Plin = data[0], data[1], data[2]
        return k, P, Plin

class GRSimulation(Simulation):
    def name(self):
        return "GR_" + Simulation.name(self)

    def params_cola(self):
        return Simulation.params_cola(self) | { # combine dictionaries
            "cosmology_h": self.params["h"],
            "cosmology_model": "LCDM",
            "gravity_model": "GR",
        }

    def   h(self): return params["h"]
    def ΩΛ0(self): return self.read_variable("class.log", "Omega_Lambda")

class JBDSimulation(Simulation):
    def name(self):
        return "JBD_" + Simulation.name(self)

    def params_class(self):
        return Simulation.params_class(self) | { # combine dictionaries
            "gravity_model": "brans_dicke", # select JBD gravity
            "Omega_Lambda": 0, # rather include Λ through potential term
            "Omega_fld": 0, # no dark energy fluid
            "Omega_smg": -1, # automatic modified gravity
            "parameters_smg": f"NaN, {params['wBD']}, 1, 0", # Λ (in JBD potential?), ωBD, Φini (guess), Φ′ini≈0 (fixed)
            "M_pl_today_smg": 1.0, # TODO: vary G/G
            "a_min_stability_test_smg": 1e-6, # BD has early-time instability, so lower tolerance to pass stability checker
            "write background": "yes",
        }

    def params_cola(self):
        return Simulation.params_cola(self) | { # combine dictionaries
            "gravity_model": "JBD",
            "cosmology_model": "JBD",
            "cosmology_h": 1.0, # h is a derived quantity in JBD cosmology, but FML needs an arbitrary nonzero value for initial calculations
            "cosmology_JBD_wBD": self.params["wBD"],
            "cosmology_JBD_GeffG_today": 1.0, # TODO: vary
            "cosmology_JBD_Omegabh2": self.params["Ωb0"] * self.params["h"]**2,
            "cosmology_JBD_OmegaCDMh2": self.params["Ωc0"] * self.params["h"]**2,
            "cosmology_JBD_OmegaLambdah2": self.params["ΩΛ0"] * self.params["h"]**2,
            "cosmology_JBD_OmegaKh2": self.params["Ωk0"] * self.params["h"]**2,
            "cosmology_JBD_OmegaMNuh2": 0.0,
        }

    def validate(self):
        pass # TODO: validate consistency from class, cola, etc.

    def    h(self): return self.read_data("class_background.dat")[3,-1] * 3e8 / 1e3 / 100
    def  ΩΛ0(self): return self.read_variable("class.log", "Lambda")
    def Φini(self): return self.read_variable("class.log", "phi_ini")

class SimulationPair:
    def __init__(self, params):
        self.sim_gr = GRSimulation(params)
        self.sim_bd = JBDSimulation(params)

    def power_spectrum_ratio(self):
        k_gr, P_gr, _ = self.sim_gr.power_spectrum()
        k_bd, P_bd, _ = self.sim_bd.power_spectrum()

        assert np.all(k_gr == k_bd), "simulations output different k-values"
        k = k_gr

        return k, P_bd / P_gr

def plot_power_spectrum(filename, k, Ps, labels):
    fig, ax = plt.subplots()
    ax.set_xlabel("$\log_{10} [k / (h/Mpc)]$")
    ax.set_ylabel("$\log_{10} [P / (Mpc/h)^3]$")
    for (P, label) in zip(Ps, labels):
        ax.plot(np.log10(k), np.log10(P), label=label)
    ax.legend()
    fig.savefig(filename)
    print(f"Plotted {filename}")

def plot_power_spectrum_ratio(filename, k, P1P2s, labels, ylabel=r"$P_\mathrm{JBD} / P_\mathrm{\Lambda CDM}$"):
    fig, ax = plt.subplots()
    ax.set_xlabel("$\log_{10} [k / (h/Mpc)]$")
    ax.set_ylabel(ylabel)
    ax.set_ylim(0.99, 1.01)
    ax.axhline(1.0, color="gray", linestyle="dashed")
    for (P1P2, label) in zip(P1P2s, labels):
        ax.plot(np.log10(k), P1P2, label=label)
    ax.legend()
    fig.savefig(filename)
    print(f"Plotted {filename}")

params0 = {
    # physical parameters
    "h":      0.67,
    "Ωb0":    0.05,
    "Ωc0":    0.267,
    "Ωk0":    0.0,
    "Tγ0":    2.7255,
    "Neff":   3.046,
    "kpivot": 0.05,
    "As":     2.1e-9,
    "ns":     0.965,

    "wBD": 1e3, # for JBD

    # computational parameters (cheap, for testing)
    "zinit": 10,
    "L": 256,
    "Npart": 64,
    "Nmesh": 64,
    "NT": 10,

    # computational parameters (expensive, for results)
    #"zinit": 30,
    #"L": 350.0,
    #"Npart": 128,
    #"Nmesh": 128,
    #"NT": 30,
}
#params0["ΩΛ0"] = 1 - params0["Ωb0"] - params0["Ωc0"] - params0["Ωk0"]
params_varying = {
    "As": (1e-9, 4e-9),
    "Ωc0": (0.15, 0.35),
    "μ0": (-1.0, +1.0),
}

paramspace = ParameterSpace(params_varying)
params = params0

sim = JBDSimulation(params)
print(f"ΩΛ0 = {sim.ΩΛ0()}")
print(f"Φini = {sim.Φini()}")
print(f"h = {sim.h()}")
sim = GRSimulation(params)
#k, P, Plin = sim.power_spectrum()
#plot_power_spectrum("plots/power_spectrum.pdf", k, [P, Plin], ["full (\"Pcb\")", "linear (\"Pcb_linear\")"])

#sims = SimulationPair(params)
#k, Pbd_Pgr = sims.power_spectrum_ratio()
