#!/usr/bin/env python3

import os
import shutil
import subprocess
import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import qmc
from classy import Class

COLAEXEC = os.path.expanduser("~/FML/FML/COLASolver/nbody")

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
        self.name = self.name(params)
        self.directory = "sims/" + self.name + "/"

        # make simulation directory
        print(f"Simulating {self.name}")
        if not self.completed():
            os.makedirs(self.directory, exist_ok=True)
            self.run_class(params)
            self.run_cola(params)

        if self.completed():
            print(f"Simulated {self.name}")

    def name(self, params):
        return f"N{params['Npart']}"

    def completed(self):
        return os.path.isfile(self.directory + f"pofk_{self.name}_cb_z0.000.txt")

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

    def run_class(self, params):
        # TODO: use hi_class for generating JBD initial conditions?
        params_class = {
            "output": "mTk,mPk", # needed for Class to compute transfer function # TODO: only mPk?
            "z_pk": 100, # redshift after which Class should compute transfer function

            "h": params["h"],
            "Omega_b": params["Ωb0"],
            "Omega_cdm": params["Ωc0"],
            "Omega_k": params["Ωk0"],
            "T_cmb": params["Tγ0"],
            "N_eff": params["Neff"],
            "A_s": params["As"],
            "n_s": params["ns"],
            "k_pivot": params["kpivot"],
        }
        self.cosmology = Class()
        self.cosmology.set(params_class)
        self.cosmology.compute()

        z1, z2 = params["zinit"]+0.1, 0.0 # TODO: FML warns about starting before zinit if z1=zinit
        a1, a2 = 1/(z1+1), 1/(z2+1)
        as_ = 10 ** np.linspace(np.log10(a1), np.log10(a2), 100)
        zs  = 1/as_ - 1
        zs  = np.flip(zs) # TODO: from 0 ???
        nsnaps = len(zs)

        transfer = self.cosmology.get_transfer(output_format="camb")
        ks = np.array(transfer["k (h/Mpc)"]) * params["h"] # k / (1/Mpc)
        Ps = np.array([self.cosmology.pk(k, z=0) for k in ks]) # P / (Mpc)^3
        self.pspecfile = "initial_power_spectrum.txt"
        self.write_data(self.pspecfile, {"k/(h/Mpc)": ks / params["h"], "P/(Mpc/h)^3": Ps * params["h"]**3}) # COLA wants "h-units"

    def params_cola(self, params, seed=1234):
        return { # common parameters (for any derived simulation)
            "simulation_name": self.name,
            "simulation_boxsize": params["L"],
            "simulation_use_cola": True,
            "simulation_use_scaledependent_cola": False, # only relevant with massive neutrinos?

            "cosmology_Omegab": params["Ωb0"],
            "cosmology_OmegaCDM": params["Ωc0"],
            "cosmology_OmegaK": params["Ωk0"],
            "cosmology_OmegaLambda": params["ΩΛ0"],
            "cosmology_Neffective": params["Neff"],
            "cosmology_TCMB_kelvin": params["Tγ0"],
            "cosmology_As": params["As"],
            "cosmology_ns": params["ns"],
            "cosmology_kpivot_mpc": params["kpivot"],
            "cosmology_OmegaMNu": 0.0,

            "particle_Npart_1D": params["Npart"],

            "timestep_nsteps": [params["NT"]],

            "ic_random_seed": seed,
            "ic_initial_redshift": params["zinit"],
            "ic_nmesh" : params["Npart"],
            "ic_type_of_input": "powerspectrum", # transferinfofile only relevant with massive neutrinos?
            "ic_input_filename": self.pspecfile,
            "ic_input_redshift": 0.0, # TODO: feed initial power spectrum directly instead of backscaling?

            "force_nmesh": params["Nmesh"],

            "output_folder": ".",
            "output_redshifts": [0.0],
        }

    def run_cola(self, params, np=1):
        colainfile = "cola_input.lua"
        self.write_file(colainfile, "\n".join(f"{param} = {luastr(val)}" for param, val in self.params_cola(params).items()))

        cmd = [COLAEXEC, colainfile]
        if np > 1:
            cmd = ["mpirun", "-np", str(np)]
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, cwd=self.directory)
        colalogfile = "cola.log"
        self.write_file(colalogfile, proc.stdout.decode())
        assert proc.returncode == 0, f"ERROR: see {colalogfile} for details"

    def power_spectrum(self):
        assert self.completed()

        # pk: "total matter"
        # pk_cb: "cdm+b"
        # pk_lin: "total matter" (linear?)

        data = self.read_data(f"pofk_{self.name}_cb_z0.000.txt")
        k, P, Plin = data[0], data[1], data[2]
        return k, P, Plin

class GRSimulation(Simulation):
    def name(self, params):
        return "GR_" + Simulation.name(self, params)

    def params_cola(self, params):
        return Simulation.params_cola(self, params) | { # combine dictionaries
            "cosmology_h": params["h"],
            "cosmology_model": "LCDM",
            "gravity_model": "GR",
        }

class JBDSimulation(Simulation):
    def name(self, params):
        return "JBD_" + Simulation.name(self, params)

    def params_cola(self, params):
        return Simulation.params_cola(self, params) | { # combine dictionaries
            "cosmology_model": "JBD",
            "cosmology_h": params["h"], # h is a derived quantity in JBD cosmology, but FML needs an arbitrary nonzero value for initial calculations
            "cosmology_JBD_wBD": params["wBD"],
            "cosmology_JBD_GeffG_today": params["h"],
            "cosmology_JBD_Omegabh2": params["Ωb0"] * params["h"]**2,
            "cosmology_JBD_OmegaCDMh2": params["Ωc0"] * params["h"]**2,
            "cosmology_JBD_OmegaLambdah2": params["ΩΛ0"] * params["h"]**2,
            "cosmology_JBD_OmegaKh2": params["Ωk0"] * params["h"]**2,
            "cosmology_JBD_OmegaMNuh2": 0.0,
            "gravity_model": "JBD",
        }

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
    "NT": 30,

    # computational parameters (expensive, for results)
    #"zinit": 30,
    #"L": 350.0,
    #"Npart": 128,
    #"Nmesh": 128,
    #"NT": 30,
}
params0["ΩΛ0"] = 1 - params0["Ωb0"] - params0["Ωc0"] - params0["Ωk0"]
params_varying = {
    "As": (1e-9, 4e-9),
    "Ωc0": (0.15, 0.35),
    "μ0": (-1.0, +1.0),
}

paramspace = ParameterSpace(params_varying)
params = params0

#sim = JBDSimulation(params)
#k, P, Plin = sim.power_spectrum()
#plot_power_spectrum("plots/power_spectrum.pdf", k, [P, Plin], ["full (\"Pcb\")", "linear (\"Pcb_linear\")"])

sims = SimulationPair(params)
k, Pbd_Pgr = sims.power_spectrum_ratio()
plot_power_spectrum_ratio("plots/power_spectrum_ratio.pdf", k, [Pbd_Pgr], ["ratio"])
