#!/usr/bin/env python3

# TODO: assert CLASS and COLA gives same field, H, etc.
# TODO: Omega_fld, Omega_Lambda, V0 fulfills same role by setting cosmo constant
# TODO: which G is hiclass' density parameters defined with respect to?
# TODO: look at PPN to understand cosmological (large) -> solar system (small) scales of G in BD
# TODO: example plots, hi-class run: see /mn/stornext/u3/hansw/Herman/WorkingHiClass/plot.py
# TODO: Hans' FML BD cosmology has not been tested with G/G != 1 !
# TODO: compare P(k) with fig. 2 on https://journals.aps.org/prd/pdf/10.1103/PhysRevD.97.023520#page=13
# TODO: don't output snapshot

import os
import re
import json
import hashlib
import shutil
import argparse
import subprocess
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.stats import qmc

parser = argparse.ArgumentParser(prog="jbd.py")
parser.add_argument("--nbody", metavar="path/to/FML/FML/COLASolver/nbody", default="./FML/FML/COLASolver/nbody")
parser.add_argument("--class", metavar="path/to/hi_class_public/class", default="./hi_class_public/class")
args = parser.parse_args()

COLAEXEC = os.path.abspath(os.path.expanduser(vars(args)["nbody"]))
CLASSEXEC = os.path.abspath(os.path.expanduser(vars(args)["class"]))

def dictjson(dict, sort=False, unicode=False):
    return json.dumps(dict, sort_keys=sort, ensure_ascii=not unicode)

def dicthash(dict):
    return hashlib.md5(dictjson(dict, sort=True).encode('utf-8')).hexdigest() # https://stackoverflow.com/a/10288255

def luastr(var):
    if isinstance(var, bool):
        return str(var).lower() # Lua uses true, not True
    elif isinstance(var, str):
        return '"' + str(var) + '"' # enclose in ""
    elif isinstance(var, list):
        return "{" + ", ".join(luastr(el) for el in var) + "}" # Lua uses {} for lists
    else:
        return str(var) # go with python's string representation

# Utility function for verifying that two quantities q1 and q2 are (almost) the same
def check_values_are_close(q1, q2, a1=None, a2=None, name="", atol=0, rtol=0, verbose=True, plot=False):
    are_arrays = isinstance(q1, np.ndarray) and isinstance(q2, np.ndarray)
    if are_arrays:
        # If q1 and q2 are function values at a1 and a2,
        # first interpolate them to common values of a
        # and compare them there
        if a1 is not None and a2 is not None:
            a = a1 if np.min(a1) > np.min(a2) else a2 # for the comparison, use largest a-values
            q1 = np.interp(a, a1, q1)
            q2 = np.interp(a, a2, q2)

        if plot: # for debugging
            plt.plot(np.log10(a), q1)
            plt.plot(np.log10(a), q2)
            plt.savefig("check.png")

        # If q1 and q2 are function values (now at common a1=a2=a),
        # pick out scalars for which the deviation is greatest
        i = np.argmax(np.abs(q1 - q2))
        a  = a[i]
        q1 = q1[i]
        q2 = q2[i]

    # do same test as np.isclose: https://numpy.org/doc/stable/reference/generated/numpy.isclose.html
    # (use atol != 0 for quantities close to zero, and rtol otherwise)
    tol = atol + np.abs(q2) * rtol
    are_close = np.abs(q1 - q2) < tol

    if verbose:
        print(f"q1 = {name}_class = {q1:e}" + (f" (picked values with greatest difference at a = {a})" if are_arrays else ""))
        print(f"q2 = {name}_cola  = {q2:e}" + (f" (picked values with greatest difference at a = {a})" if are_arrays else ""))
        print("^^ PASSED" if are_close else "FAILED", f"test |q1-q2| = {np.abs(q1-q2):.2e} < {tol:.2e} = tol = atol + rtol*|q2| with atol={atol:.1e}, rtol={rtol:.1e}")

    assert are_close, f"{name} is not consistent in CLASS and COLA"

def list_simulations():
    for path in os.scandir("sims/"):
        if os.path.isdir(path):
            print(f"{path.name}: {BDSimulation(path=path.name).params}")

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
    def __init__(self, params=None, path=None):
        if path is not None:
            self.directory = "sims/" + path + "/"
            params = json.loads(self.read_file("parameters.json"))
            return Simulation.__init__(self, params=params)

        self.params = params.copy() # will be modified
        self.name = self.name()
        self.directory = "sims/" + self.name + "/"
        #self.rename_legacy() # TODO: move legacy directory
        #return # TODO: remove

        # initialize simulation, validate input, create working directory, write parameters
        print(f"Simulating {self.name} with independent parameters:")
        print("\n".join(f"{param} = {self.params[param]}" for param in sorted(self.params)))
        self.validate_input()
        os.makedirs(self.directory, exist_ok=True)
        self.write_file("parameters.json", dictjson(self.params, unicode=True))

        # create initial conditions with CLASS, store derived parameters, run COLA simulation
        ks, Ps = self.run_class()
        self.run_cola(ks, Ps, np=16)

        # verify successful completion
        assert self.completed()
        self.validate_output()
        print(f"Simulated {self.name}")

    # unique string identifier for the simulation
    # TODO: create unique hash from parameters: 
    # TODO: return array of names (to look for renaming etc.)
    # TODO: also output JSON dict with parameters
    def name(self):
        return dicthash(self.params)

    def names_old(self):
        return [f"NP{self.params['Npart']}_NM{self.params['Ncell']}_NS{self.params['Nstep']}_L{self.params['L']}"]

    def rename_legacy(self):
        for name_old in self.names_old():
            path_old = f"sims/{name_old}"
            print("candidate ", path_old)
            if os.path.isdir(path_old):
                print(f"want to rename {path_old} -> {self.directory}")
                #os.rename(path_old, self.directory)
                return
        print("no rename")

    # whether CLASS has been run
    def completed_class(self):
        return os.path.isfile(self.directory + f"class_pk.dat")

    # whether COLA has been run
    def completed_cola(self):
        return os.path.isfile(self.directory + f"pofk_{self.name}_cb_z0.000.txt")

    # whether CLASS and COLA has been run
    def completed(self):
        return self.completed_class() and self.completed_cola()

    # check that the combination of parameters passed to the simulation is allowed
    def validate_input(self):
        assert "ωΛ0" not in self.params, "derived parameter ωΛ0 is specified"

    # check that the output from CLASS and COLA is consistent
    def validate_output(self):
        print("Checking consistency between quantities computed separately by CLASS and COLA/FML:")

        # Read background tables and their scale factors (which we use as the free time variable)
        bg_class = self.read_data("class_background.dat")
        bg_cola  = self.read_data(f"cosmology_{self.name}.txt")
        z_class  = bg_class[0]
        a_class  = 1 / (1 + z_class)
        a_cola   = bg_cola[0]

        # Compare E = H/H0
        H_class = bg_class[3]
        E_class = H_class / H_class[-1] # E = H/H0 (assuming final value is at a=1)
        E_cola  = bg_cola[1]
        check_values_are_close(E_class, E_cola, a_class, a_cola, name="(H/H0)", rtol=1e-4)

        # Compare ΩΛ0
        ΩΛ0_class = self.read_variable("class.log", "Lambda")
        ΩΛ0_cola  = self.read_variable("cola.log", "OmegaLambda      ", between=" : ")
        check_values_are_close(ΩΛ0_class, ΩΛ0_cola, name="ΩΛ0", rtol=1e-4)

    # save a data file associated with the simulation
    def write_data(self, filename, cols, colnames=None):
        if isinstance(cols, dict):
            colnames = cols.keys()
            cols = [cols[colname] for colname in colnames]
            return self.write_data(filename, cols, colnames)

        header = None if colnames is None else " ".join(colnames)
        np.savetxt(self.directory + filename, np.transpose(cols), header=header)

    # load a data file associated with the simulation
    def read_data(self, filename):
        data = np.loadtxt(self.directory + filename)
        data = np.transpose(data)
        return data

    # save a file associated with the simulation
    def write_file(self, filename, string):
        with open(self.directory + filename, "w", encoding="utf-8") as file:
            file.write(string)

    # load a file associated with the simulation
    def read_file(self, filename):
        with open(self.directory + filename, "r", encoding="utf-8") as file:
            return file.read()

    # read a string like "variable = numerical_value" and return the numerical value
    def read_variable(self, filename, variable, between=" = "):
        pattern = variable + between + r"([-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?)" # e.g. "var = 1.2e-34"
        matches = re.findall(pattern, self.read_file(filename))
        assert len(matches) == 1, f"found {len(matches)} != 1 matches: {matches}"
        return float(matches[0][0])

    # run a command in the simulation's directory
    def run_command(self, cmd, log="/dev/null", verbose=True):
        teecmd = subprocess.Popen(["tee", log], stdin=subprocess.PIPE, stdout=None if verbose else subprocess.DEVNULL, cwd=self.directory)
        runcmd = subprocess.Popen(cmd, stdout=teecmd.stdin, stderr=subprocess.STDOUT, cwd=self.directory)

        # TODO: check/return exit status
        runcmd.wait() # wait for command to finish
        teecmd.stdin.close() # close stream

    # dictionary of parameters that should be passed to CLASS
    def params_class(self):
        return {
            # cosmological parameters
            "h": self.params["h"],
            "Omega_b": self.params["ωb0"] / self.params["h"]**2,
            "Omega_cdm": self.params["ωc0"] / self.params["h"]**2,
            "Omega_k": self.params["ωk0"] / self.params["h"]**2,
            "T_cmb": self.params["Tγ0"],
            "N_eff": self.params["Neff"],
            "A_s": self.params["As"],
            "n_s": self.params["ns"],
            "k_pivot": self.params["kpivot"],

            # output control
            "output": "mPk",
            "write background": "yes",
            "root": "class_",
            "P_k_max_1/Mpc": 11.0, # output linear power spectrum to fill my plots

            # log verbosity (increase integers to make more talkative)
            "input_verbose": 10,
            "background_verbose": 10,
            "thermodynamics_verbose": 1,
            "perturbations_verbose": 1,
            "spectra_verbose": 1,
            "output_verbose": 1,
        }

    # run CLASS and return today's matter power spectrum
    # TODO: use hi_class for generating BD initial conditions?
    # TODO: which k-values to choose? see https://github.com/lesgourg/class_public/blob/aa92943e4ab86b56970953589b4897adf2bd0f99/explanatory.ini#L1102
    def run_class(self, input="class_input.ini", log="class.log"):
        if not self.completed_class():
            # write input and run class
            self.write_file(input, "\n".join(f"{param} = {str(val)}" for param, val in self.params_class().items()))
            self.run_command([CLASSEXEC, input], log=log, verbose=True)
        assert self.completed_class(), f"ERROR: see {log} for details"

        # get output power spectrum (COLA needs class' output power spectrum, just without comments)
        # TODO: which h does hiclass use here?
        # TODO: set the "non-used" h = 1.0 to avoid division?
        return self.power_spectrum(linear=True)

    # dictionary of parameters that should be passed to COLA
    def params_cola(self, seed=1234):
        return { # common parameters (for any derived simulation)
            "simulation_name": self.name,
            "simulation_boxsize": self.params["L"], # TODO: convert to physical (instead of h-based boxsize), since h can differ?
            "simulation_use_cola": True,
            "simulation_use_scaledependent_cola": False, # only relevant with massive neutrinos?

            "cosmology_Omegab": self.params["ωb0"] / self.params["h"]**2,
            "cosmology_OmegaCDM": self.params["ωc0"] / self.params["h"]**2,
            "cosmology_OmegaK": self.params["ωk0"] / self.params["h"]**2,
            "cosmology_Neffective": self.params["Neff"],
            "cosmology_TCMB_kelvin": self.params["Tγ0"],
            "cosmology_As": self.params["As"],
            "cosmology_ns": self.params["ns"],
            "cosmology_kpivot_mpc": self.params["kpivot"],
            "cosmology_OmegaMNu": 0.0,

            "particle_Npart_1D": self.params["Npart"],

            "timestep_nsteps": [self.params["Nstep"]],

            "ic_random_seed": seed,
            "ic_initial_redshift": self.params["zinit"],
            "ic_nmesh" : self.params["Ncell"],
            "ic_type_of_input": "powerspectrum", # transferinfofile only relevant with massive neutrinos?
            "ic_input_filename": "power_spectrum_today.dat",
            "ic_input_redshift": 0.0, # TODO: feed initial power spectrum directly instead of backscaling?

            "force_nmesh": self.params["Ncell"],

            "output_folder": ".",
            "output_redshifts": [0.0],
        }

    # run COLA simulation from back-scaling today's matter power spectrum (from CLASS)
    def run_cola(self, khs, Phs, np=1, verbose=True, ic="power_spectrum_today.dat", input="cola_input.lua", log="cola.log"):
        if not self.completed_cola():
            self.write_data(ic, {"k/(h/Mpc)": khs, "P/(Mpc/h)^3": Phs}) # COLA wants "h-units" # TODO: give cola the actual used h for ICs?
            self.write_file(input, "\n".join(f"{param} = {luastr(val)}" for param, val in self.params_cola().items()))
            cmd = ["mpirun", "-np", str(np), COLAEXEC, input] if np > 1 else [COLAEXEC, input]
            self.run_command(cmd, log=log, verbose=True)
            shutil.rmtree(self.directory + f"snapshot_{self.name}_z0.000/") # delete particle data # TODO: delete big snapshot/particle files?
        assert self.completed_cola(), f"ERROR: see {log} for details"

    def power_spectrum(self, linear=False):
        if linear:
            assert self.completed_class()
            khs, Phs = self.read_data("class_pk.dat") # k / (h/Mpc); P / (Mpc/h)^3 (in "h-units")
        else:
            # pk: "total matter"
            # pk_cb: "cdm+b"
            # pk_lin: "total matter" (linear?)
            assert self.completed_cola()
            data = self.read_data(f"pofk_{self.name}_cb_z0.000.txt")
            khs, Phs = data[0], data[1]
        return khs, Phs

class GRSimulation(Simulation):
    #def name(self):
        #return "GR_" + Simulation.name(self)

    #def names_old(self):
        #return ["GR_" + name_old for name_old in Simulation.names_old(self)]

    def params_cola(self):
        return Simulation.params_cola(self) | { # combine dictionaries
            "cosmology_h": self.params["h"],
            "cosmology_model": "LCDM",
            "gravity_model": "GR",
        }

class BDSimulation(Simulation):
    #def name(self):
        #return "BD_" + Simulation.name(self)

    #def names_old(self):
        #return ["BD_" + name_old for name_old in Simulation.names_old(self)]

    def validate_input(self):
        Simulation.validate_input(self)

    def params_class(self):
        ω = self.params["ω"]
        return Simulation.params_class(self) | { # combine dictionaries
            "gravity_model": "brans_dicke", # select BD gravity
            "Omega_Lambda": 0, # rather include Λ through potential term (first entry in parameters_smg)
            "Omega_fld": 0, # no dark energy fluid
            "Omega_smg": -1, # automatic modified gravity
            "parameters_smg": f"NaN, {ω}, 1, 0", # ΩΛ0 (fill with cosmological constant), ω, Φini (arbitrary initial guess), Φ′ini≈0 (fixed)
            "M_pl_today_smg": (4+2*ω)/(3+2*ω) / self.params["Geff/G"],
            "a_min_stability_test_smg": 1e-6, # BD has early-time instability, so lower tolerance to pass stability checker
            "output_background_smg": 2, # >= 2 needed to output phi to background table (https://github.com/miguelzuma/hi_class_public/blob/16ae0f6ccfcee513146ec36b690678f34fb687f4/source/background.c#L3031)
        }

    def params_cola(self):
        return Simulation.params_cola(self) | { # combine dictionaries
            "gravity_model": "JBD",
            "cosmology_model": "JBD",
            "cosmology_h": self.params["h"],
            "cosmology_JBD_wBD": self.params["ω"],
            "cosmology_JBD_GeffG_today": self.params["Geff/G"],
        }

    def validate_output(self):
        Simulation.validate_output(self) # do any validation in parent class

        # Read background tables and their scale factors (which we use as the free time variable)
        bg_class = self.read_data("class_background.dat")
        bg_cola  = self.read_data(f"cosmology_{self.name}.txt")
        z_class  = bg_class[0]
        a_class  = 1 / (1 + z_class)
        a_cola   = bg_cola[0]

        H_class = bg_class[3]

        # Compare ϕ
        ϕ_class = bg_class[25]
        ϕ_cola = bg_cola[9]
        check_values_are_close(ϕ_class, ϕ_cola, a_class, a_cola, name="ϕ", rtol=1e-5)

        # Compare dlogϕ/dloga
        dϕ_dη_class       = bg_class[26]
        dlogϕ_dloga_class = dϕ_dη_class / ϕ_class / (H_class * a_class)
        dlogϕ_dloga_cola  = bg_cola[10]
        check_values_are_close(dlogϕ_dloga_class, dlogϕ_dloga_cola, a_class, a_cola, name="dlogϕ/dloga", atol=1e-4)

    # derived parameters
    def  ΩΛ0(self): return self.read_variable("class.log", "Lambda")
    def Φini(self): return self.read_variable("class.log", "phi_ini")

class SimulationPair:
    def __init__(self, sim1, sim2):
        self.sim1 = sim1
        self.sim2 = sim2

    #def __init__(self, params, wGR=1e6):
        #self.sim_gr = BDSimulation(params | {"ω": ωGR}) # TODO: use BD with large w, or a proper GR simulation?
        #self.sim_bd = BDSimulation(params)

    # TODO: how to handle different ks in best way?
    # TODO: more natural (more similar ks) if plotted in normal units (not in h-units)?
    # TODO: for linear, specify exact ks to class?
    def power_spectrum_ratio(self, linear=False):
        k1, P1 = self.sim1.power_spectrum(linear)
        k2, P2 = self.sim2.power_spectrum(linear)

        # TODO: use non-h-units for k, since I am comparing two theories with different h!!!
        k1 = k1 / self.sim1.params["h"]
        k2 = k2 / self.sim2.params["h"]
        P1 = P1 * self.sim1.params["h"]**3
        P2 = P2 * self.sim2.params["h"]**3

        #assert np.all(np.isclose(k1, k2, atol=1e-2)), f"simulations output different k-values: max(abs(k1-k2)) = {np.max(np.abs(k1-k2))}"
        #k1 = k1 / self.sim1.params["h"]
        #k2 = k2 / self.sim2.params["h"]
        #k = k1

        #kmin = np.maximum(k1[0],  k2[0])
        #kmax = np.minimum(k1[-1], k2[-1])

        # get common (average) ks and interpolate P there
        k = (k1 + k2) / 2
        P1 = np.interp(k, k1, P1)
        P2 = np.interp(k, k2, P2)

        return k, P1 / P2

def plot_sequence(filename, paramss, labelfunc = lambda params: None, colorfunc = lambda params: "black", plot_linear = True):
    fig, ax = plt.subplots()
    ax.set_xlabel(r"$\lg[k / (1/\mathrm{Mpc})]$")
    ax.set_ylabel(r"$P_\mathrm{BD} / P_\mathrm{GR}$")

    dy = 0.05 # spacing between major y ticks
    ymin = 0.96
    ymax = 1.0
    for params_bd in paramss:
        params_gr = params_bd.copy()
        del params_gr["ω"] # remove BD-specific parameters
        del params_gr["Geff/G"] # remove BD-specific parameters

        sim_bd = BDSimulation(params_bd)
        sim_gr = GRSimulation(params_gr)
        sims   = SimulationPair(sim_bd, sim_gr)

        if plot_linear:
            k, P1_P2 = sims.power_spectrum_ratio(linear=True)
            k, P1_P2 = k[k>0.9e-2], P1_P2[k>0.9e-2] # cut away k < 10^-2 / Mpc
            ax.plot(np.log10(k), P1_P2, linewidth=1, color=colorfunc(params_bd), alpha=0.5, linestyle="dashed", label=None, zorder=1) # TODO: create dashed black dummy legend before/after

        k, P1_P2 = sims.power_spectrum_ratio(linear=False)
        ax.plot(np.log10(k), P1_P2, linewidth=1, color=colorfunc(params_bd), alpha=1.0, linestyle="solid", label=labelfunc(params_bd), zorder=2)

        ymin = np.minimum(ymin, np.min(P1_P2))
        ymax = np.maximum(ymax, np.max(P1_P2))

    ymax = np.ceil( np.maximum(ymax, np.max(P1_P2)) / dy) * dy # round to nearest full tick
    ymin = np.floor(np.minimum(ymin, np.min(P1_P2)) / dy) * dy # round to nearest full tick

    ax.set_xlim(-2, +1)
    ax.set_ylim(ymin, ymax)
    ax.set_xticks([-2, -1, 0, 1])
    ax.set_yticks(np.linspace(ymin, ymax, int(np.round((ymax-ymin)/dy)) + 1))
    ax.xaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator(10))
    ax.yaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator(int(np.round(dy / 0.01)))) # one minor tick per 0.01

    ax.legend()
    ax.grid()
    fig.tight_layout()
    fig.savefig(filename)
    print("Plotted", filename)

def plot_convergence(filename, params0, param, vals, lfunc=None, cfunc=None, **kwargs):
    paramss = (params0 | {param: val} for val in vals) # generate all parameter combinations

    if lfunc is None:
        def lfunc(v):
            return f"{param} = {v}"
    def labelfunc(params):
        return lfunc(params[param])

    # linear color C = A*v + B so C(v0) = 0.5 (black) and C(vmin) = 0.0 (blue) *or* C(vmax) = 1.0 (red)
    # (black = neutral = fiducial; red = warm = greater; blue = cold = smaller)
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("blueblackred", ["#0000ff", "#000000", "#ff0000"], N=256)
    if cfunc is None:
        def cfunc(v):
            return v # identity
    def colorfunc(params):
        v    = cfunc(params[param])  # current  (transformed) value
        v0   = cfunc(params0[param]) # fiducial (transformed) value
        vmin = np.min(cfunc(vals))   # minimum  (transformed) value
        vmax = np.max(cfunc(vals))   # maximum  (transformed) value
        A = 0.5 / (vmax - v0) if vmax - v0 > v0 - vmin else 0.5 / (v0 - vmin) # if/else saturates one end of color spectrum
        B = 0.5 - A * v0
        return cmap(A * v + B)

    plot_sequence(filename, paramss, labelfunc, colorfunc, **kwargs)

# Fiducial parameters
params0_GR = {
    # physical parameters
    "h":      0.68,   # class' default
    "ωb0":    0.022,  # class' default
    "ωc0":    0.120,  # class' default
    "ωk0":    0.0,    # class' default
    "Tγ0":    2.7255, # class' default
    "Neff":   3.044,  # class' default # TODO: handled correctly in COLA?
    "kpivot": 0.05,   # class' default
    "As":     2.1e-9, # class' default
    "ns":     0.966,  # class' default

    # computational parameters (max on euclid22-32: Npart=Ncell=1024 with np=16 CPUs)
    "zinit": 10.0,
    "Nstep": 30,
    "Npart": 512,
    "Ncell": 512,
    "L":     512.0,
}
params0_BD = params0_GR | {
    "ω": 1e2, # lowest value to consider (larger values should only be "easier" to simulate?)
    "Geff/G": 1.0
}

params_varying = {
    "As": (1e-9, 4e-9),
    "Ωc0": (0.15, 0.35),
    "μ0": (-1.0, +1.0),
}

# TODO: split up into different "run files"

# List simulations
#list_simulations()
#exit()

#paramspace = ParameterSpace(params_varying)
#params = params0

# Check that CLASS and COLA outputs consistent background cosmology parameters
# for a "non-standard" BD cosmology with small wBD and Geff/G != 1
# (using cheap COLA computational parameters, so the simulation finishes near-instantly)
#GRSimulation(params0_GR | {"Npart": 0, "Ncell": 16, "Nstep": 0, "L": 4})
#BDSimulation(params0_BD | {"Npart": 0, "Ncell": 16, "Nstep": 0, "L": 4, "ω": 1e2, "Geff/G": 1.1})
#exit()

#BDSimulation(params0_BD)
#GRSimulation(params0_GR)
#exit()

# Convergence plots (computational parameters)
plot_convergence("plots/convergence_L.pdf",     params0_BD, "L",      [256.0, 384.0, 512.0, 768.0, 1024.0], lfunc=lambda L: f"$L = {L:.0f}\,\mathrm{{Mpc}}$")
plot_convergence("plots/convergence_Npart.pdf", params0_BD, "Npart",  [256, 384, 512, 768, 1024],           lfunc=lambda Npart: f"$N_\mathrm{{part}} = {Npart}^3$")
plot_convergence("plots/convergence_Ncell.pdf", params0_BD, "Ncell",  [256, 384, 512, 768, 1024],           lfunc=lambda Ncell: f"$N_\mathrm{{cell}} = {Ncell}^3$")
plot_convergence("plots/convergence_Nstep.pdf", params0_BD, "Nstep",  [10, 20, 30, 40, 50],                 lfunc=lambda Nstep: f"$N_\mathrm{{step}} = {Nstep}$")
plot_convergence("plots/convergence_zinit.pdf", params0_BD, "zinit",  [10.0, 20.0, 30.0],                   lfunc=lambda zinit: f"$z_\mathrm{{init}} = {zinit:.0f}$")

# Variation plots (cosmological parameters)
plot_convergence("plots/convergence_omega.pdf",   params0_BD, "ω",      [1e2, 1e3, 1e4, 1e5],               lfunc=lambda ω: f"$\omega = 10^{{{np.log10(ω):.0f}}}$", cfunc=lambda ω: np.log10(ω))
plot_convergence("plots/convergence_Geff.pdf",    params0_BD, "Geff/G", [0.95, 1.0, 1.05],                  lfunc=lambda Geff_G: f"$G_\mathrm{{eff}}/G = {Geff_G:.02f}$")
plot_convergence("plots/convergence_h.pdf",       params0_BD, "h",      [0.63, 0.68, 0.73],                 lfunc=lambda h: f"$h = {h}$")
plot_convergence("plots/convergence_omegab0.pdf", params0_BD, "ωb0",    [0.016, 0.022, 0.028],              lfunc=lambda ωb0: f"$\omega_{{b0}} = {ωb0}$")
plot_convergence("plots/convergence_omegac0.pdf", params0_BD, "ωc0",    [0.090, 0.120, 0.150],              lfunc=lambda ωc0: f"$\omega_{{c0}} = {ωc0}$")
plot_convergence("plots/convergence_As.pdf",      params0_BD, "As",    [1.6e-9, 2.1e-9, 2.7e-9],            lfunc=lambda As:  f"$A_s = {np.round(As/1e-9, 1)} \cdot 10^{{-9}}$")
exit()

#sims = SimulationPair(params)
#k, Pbd_Pgr = sims.power_spectrum_ratio()
