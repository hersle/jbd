#!/usr/bin/env python3

# TODO: does hiclass crash with G0/G very different from 1 with small ω?
# TODO: run "properly" with and without hGR parameter transformation
# TODO: GR emulators (?): Bacco, CosmicEmu, EuclidEmulator2, references within
# TODO: compare boost with nonlinear prediction from hiclass' hmcode?
# TODO: look at PPN to understand cosmological (large) -> solar system (small) scales of G in BD
# TODO: compare P(k) with fig. 2 on https://journals.aps.org/prd/pdf/10.1103/PhysRevD.97.023520#page=13
# TODO: don't output snapshot
# TODO: fix GeffG in FML (currently in large-ω limit)

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
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.stats import qmc
from scipy.interpolate import interp1d, CubicSpline

matplotlib.rcParams |= {
    "text.usetex": True,
    "font.size": 9,
    "figure.figsize": (6.0, 4.0), # default (6.4, 4.8)
    "grid.linewidth": 0.3,
    "grid.alpha": 0.2,
    "legend.labelspacing": 0.3,
    "legend.columnspacing": 1.5,
    "legend.handlelength": 1.5,
    "legend.handletextpad": 0.3,
    "legend.frameon": False,
    "xtick.top": True,
    "ytick.right": True,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "axes.xmargin": 0,
    "axes.ymargin": 0,
}

parser = argparse.ArgumentParser(prog="jbd.py")
parser.add_argument("action", help="action(s) to execute", nargs="+")
parser.add_argument("--nbody", metavar="path/to/FML/FML/COLASolver/nbody", default="./FML/FML/COLASolver/nbody")
parser.add_argument("--class", metavar="path/to/hi_class_public/class", default="./hi_class_public/class")
parser.add_argument("--simdir", metavar="path/to/simulation/directory/", default="./sims/")
args = parser.parse_args()

ACTIONS = vars(args)["action"]
COLAEXEC = os.path.abspath(os.path.expanduser(vars(args)["nbody"]))
CLASSEXEC = os.path.abspath(os.path.expanduser(vars(args)["class"]))
SIMDIR = vars(args)["simdir"].rstrip("/") + "/" # enforce trailing /

def to_nearest(number, nearest, mode="round"):
    if mode == "round":
        func = np.round
    elif mode == "ceil":
        func = np.ceil
    elif mode == "floor":
        func = np.floor
    else:
        assert f"unknown mode {mode}"
    return func(np.round(number / nearest, 7)) * nearest # round to many digits first to eliminate floating point errors (this is only used for plotting purposes, anyway)

def ax_set_ylim_nearest(ax, Δy):
    ymin, ymax = ax.get_ylim()
    ymin = to_nearest(ymin, Δy, "floor")
    ymax = to_nearest(ymax, Δy, "ceil")
    ax.set_ylim(ymin, ymax)
    ax.set_yticks(np.linspace(ymin, ymax, int(np.round((ymax-ymin)/Δy))+1)) # TODO: set minor ticks every 1, 0.1, 0.01 etc. here?
    return ymin, ymax

def dictupdate(dict, add={}, remove=[]):
    dict = dict.copy() # don't modify input!
    for key in remove:
        del dict[key]
    for key in add:
        dict[key] = add[key]
    return dict

def dictkeycount(dict, keys, number=None):
    if number is None: number = len(keys)
    return len(set(dict).intersection(set(keys))) == number

def dictjson(dict, sort=False, unicode=False):
    return json.dumps(dict, sort_keys=sort, ensure_ascii=not unicode)

def jsondict(jsonstr):
    return json.loads(jsonstr)

def hashstr(str):
    return hashlib.md5(str.encode('utf-8')).hexdigest()

def hashdict(dict):
    return hashstr(dictjson(dict, sort=True)) # https://stackoverflow.com/a/10288255

def params2seeds(params, n=None):
    rng = np.random.default_rng(int(hashdict(params), 16)) # deterministic random number generator from simulation parameters
    seeds = rng.integers(0, 2**31-1, size=n, dtype=int) # output python (not numpy) ints to make compatible with JSON dict hashing
    return int(seeds) if n is None else [int(seed) for seed in seeds]

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

# propagate error in f(x1, x2, ...) given
# * df_dx = [df_dx1, df_dx2, ...] (list of numbers): derivatives df/dxi evaluated at mean x
# * xs    = [x1s,    x2s,    ...] (list of lists of numbers): list of observations xis for each variable xi
# (for reference,       see e.g. https://en.wikipedia.org/wiki/Propagation_of_uncertainty#Non-linear_combinations)
# (for an introduction, see e.g. https://veritas.ucd.ie/~apl/labs_master/docs/2020/DA/Matrix-Methods-for-Error-Propagation.pdf)
def propagate_error(df_dx, xs):
    return np.sqrt(np.transpose(df_dx) @ np.cov(xs) @ df_dx)

def list_simulations():
    for path in os.scandir(SIMDIR):
        if os.path.isdir(path):
            Simulation(path=path.name, verbose=True, run=False)

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

class Simulation:
    def __init__(self, params=None, seed=None, path=None, verbose=True, run=True):
        if path is not None:
            self.directory = SIMDIR + path + "/"
            params = jsondict(self.read_file("parameters.json"))
            seed = params.pop("seed") # remove seed key
            constructor = BDSimulation if "lgω" in params else GRSimulation
            constructor(params=params, seed=seed, verbose=verbose, run=run)
            return None

        if seed is None:
            seed = params2seeds(params)

        self.params = dictupdate(params, {"seed": seed}) # make seed part of parameters only internally
        self.name = self.name()
        self.directory = SIMDIR + self.name + "/"

        if verbose:
            print(f"{self.directory}: {self.params}", end="") # print independend parameters
            if self.completed():
                params_derived = self.params_extended()
                params_derived = dict(set(params_derived.items()) - set(self.params.items()))
                print(f" -> {params_derived}", end="") # print dependend/derived parameters
            print() # end line

        # initialize simulation, validate input, create working directory, write parameters
        os.makedirs(self.directory, exist_ok=True)
        self.write_file("parameters.json", dictjson(self.params, unicode=True))

        # create initial conditions with CLASS, store derived parameters, run COLA simulation
        # TODO: be lazy
        if run and not self.completed():
            self.validate_input(params)

            # parametrize with ωm0 and ωb0 (letting ωc0 be derived)
            if "ωb0" not in self.params:
                self.params["ωb0"] = self.params["ωm0"] - self.params["ωc0"]
            elif "ωc0" not in self.params:
                self.params["ωc0"] = self.params["ωm0"] - self.params["ωb0"]
            # else: "ωm0" not in self.params, so both "ωb0" and "ωc0" are specified

            # find As that gives desired σ8
            if "σ8" in self.params: # overrides Ase9
                σ8_target = self.params["σ8"]
                Ase9 = 1.0 # initial guess
                while True: # "σ8" not in self.params_extended() or np.abs(self.params_extended()["σ8"] - σ8_target) > 1e-5:
                    self.params["Ase9"] = Ase9
                    k, P = self.run_class()
                    σ8 = self.params_extended()["σ8"]
                    if np.abs(σ8 - σ8_target) < 1e-10:
                        break
                    Ase9 = (σ8_target/σ8)**2 * Ase9 # exploit σ8^2 ∝ As to fixed-point iterate efficiently (should only require one iteration)
            else:
                k, P = self.run_class()

            self.run_cola(k, P, np=16)
            self.write_file("parameters_extended.json", dictjson(self.params_extended(), unicode=True))

            self.validate_output()
            assert self.completed() # verify successful completion

    # unique string identifier for the simulation
    def name(self):
        return hashdict(self.params)

    # whether CLASS has been run
    def completed_class(self):
        return os.path.isfile(self.directory + f"class_pk.dat")

    # whether COLA has been run
    def completed_cola(self):
        return os.path.isfile(self.directory + f"pofk_{self.name}_cb_z0.000.txt")

    # whether CLASS and COLA has been run
    def completed(self):
        return os.path.isfile(self.directory + "parameters_extended.json")

    # check that the combination of parameters passed to the simulation is allowed
    def validate_input(self, params):
        assert dictkeycount(params, {"ωb0", "ωc0", "ωm0"}, 2), "specify two of ωb0, ωc0, ωm0"
        assert dictkeycount(params, {"Ase9", "σ8"}, 1), "specify one of Ase9, σ8"
        assert dictkeycount(params, {"ωΛ0"}, 0), "don't specify derived parameter ωΛ0"

    # check that the output from CLASS and COLA is consistent
    def validate_output(self):
        print("Checking consistency between quantities computed separately by CLASS and COLA/FML:")

        # Read background quantities
        z_class, H_class = self.read_data("class_background.dat", dict=True, cols=("z", "H [1/Mpc]"))
        a_cola, E_cola = self.read_data(f"cosmology_{self.name}.txt", dict=True, cols=("a", "H/H0"))
        a_class  = 1 / (1 + z_class)

        # Compare E = H/H0
        E_class = H_class / H_class[-1] # E = H/H0 (assuming final value is at a=1)
        check_values_are_close(E_class, E_cola, a_class, a_cola, name="(H/H0)", rtol=1e-4)

        # Compare ΩΛ0
        ΩΛ0_class = self.read_variable("class.log", "Lambda = ")
        ΩΛ0_cola  = self.read_variable("cola.log", "OmegaLambda       : ")
        check_values_are_close(ΩΛ0_class, ΩΛ0_cola, name="ΩΛ0", rtol=1e-4)

        # Compare σ8 today
        σ8_class = self.read_variable("class.log", "sigma8=")
        σ8_cola  = self.read_variable("cola.log",  "Sigma\(R = 8 Mpc/h, z = 0.0 \) :\s+")
        check_values_are_close(σ8_class, σ8_cola, name="σ8", rtol=1e-3)

    # save a data file associated with the simulation
    def write_data(self, filename, cols, colnames=None):
        if isinstance(cols, dict):
            colnames = cols.keys()
            cols = [cols[colname] for colname in colnames]
            return self.write_data(filename, cols, colnames)

        header = None if colnames is None else " ".join(colnames)
        np.savetxt(self.directory + filename, np.transpose(cols), header=header)

    # load a data file associated with the simulation
    def read_data(self, filename, dict=False, cols=None):
        data = np.loadtxt(self.directory + filename)
        data = np.transpose(data) # index by [icol, irow] (or just [icol] to get a whole column)

        # if requested, read header and generate {header[icol]: data[icol]} dictionary
        if dict:
            with open(self.directory + filename) as file:
                # the header line (with column titles) is always the last commented line
                while True:
                    line = file.readline()
                    if not line.startswith("#"):
                        break
                    header = line

                header = header.lstrip('#').lstrip().rstrip() # remove starting comment sign (#) and starting and trailing whitespace (spaces and newlines)

                header = re.split(r"\s{2,}", header) # split only on >=2 spaces because class likes to have spaces in column titles :/
                header = header[:len(data)] # remove any trailing whitespace because class outputs trailing whitespace :/
                assert len(header) == len(data) # make sure the hacks we do because of class seem consistent :/
                header = [head[1+head.find(":"):] for head in header] # ugly hack because class outputs weird and ugly column titles :/

            data = {header[col]: data[col] for col in range(0, len(data))}

        # if requested, return only requested column numbers (or titles)
        if cols is not None:
            data = tuple(data[col] for col in cols)

        return data

    # save a file associated with the simulation
    def write_file(self, filename, string):
        with open(self.directory + filename, "w", encoding="utf-8") as file:
            file.write(string)

    # load a file associated with the simulation
    def read_file(self, filename):
        with open(self.directory + filename, "r", encoding="utf-8") as file:
            return file.read()

    # read a string like "[prefix][number]" from a file and return number
    # example: if file contains "Omega_Lambda = 1.23", read_variable(filename, "Omega_Lambda = ") returns 1.23
    def read_variable(self, filename, prefix):
        regex = prefix + r"([-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?)"
        matches = re.findall(regex, self.read_file(filename))
        assert len(matches) == 1, f"found {len(matches)} ≠ 1 matches {matches} for regex \"{regex}\" in file {filename}"
        return float(matches[0][0])

    # run a command in the simulation's directory
    def run_command(self, cmd, log="/dev/null", verbose=True):
        teecmd = subprocess.Popen(["tee", log], stdin=subprocess.PIPE, stdout=None if verbose else subprocess.DEVNULL, cwd=self.directory)
        runcmd = subprocess.Popen(cmd, stdout=teecmd.stdin, stderr=subprocess.STDOUT, cwd=self.directory)

        # TODO: check/return exit status?
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
            "A_s": self.params["Ase9"] / 1e9,
            "n_s": self.params["ns"],
            "k_pivot": self.params["kpivot"],

            "YHe": 0.25,

            # output control
            "output": "mPk",
            "write background": "yes",
            "root": "class_",
            "P_k_max_h/Mpc": 100.0, # output linear power spectrum to fill my plots

            # log verbosity (increase integers to make more talkative)
            "input_verbose": 10,
            "background_verbose": 10,
            "thermodynamics_verbose": 2,
            "perturbations_verbose": 2,
            "spectra_verbose": 2,
            "output_verbose": 2,
        }

    # run CLASS and return today's matter power spectrum
    def run_class(self, input="class_input.ini", log="class.log"):
        # write input and run class
        self.write_file(input, "\n".join(f"{param} = {str(val)}" for param, val in self.params_class().items()))
        self.run_command([CLASSEXEC, input], log=log, verbose=True)
        assert self.completed_class(), f"ERROR: see {log} for details"

        # get output power spectrum (COLA needs class' output power spectrum, just without comments)
        return self.power_spectrum(linear=True, hunits=True) # COLA wants power spectrum in h units

    # dictionary of parameters that should be passed to COLA
    def params_cola(self):
        return { # common parameters (for any derived simulation)
            "simulation_name": self.name,
            "simulation_boxsize": self.params["Lh"], # TODO: flexibly give L with or without hunits (using parameter reduction function?) # h factors out of sim equations, so instead of L = Lphys / Mpc, it wants Lsim = Lphys / (Mpc/h) = L * h
            "simulation_use_cola": True,
            "simulation_use_scaledependent_cola": False, # TODO: only relevant with massive neutrinos?

            "cosmology_h": self.params["h"],
            "cosmology_Omegab": self.params["ωb0"] / self.params["h"]**2,
            "cosmology_OmegaCDM": self.params["ωc0"] / self.params["h"]**2,
            "cosmology_OmegaK": self.params["ωk0"] / self.params["h"]**2,
            "cosmology_Neffective": self.params["Neff"],
            "cosmology_TCMB_kelvin": self.params["Tγ0"],
            "cosmology_As": self.params["Ase9"] / 1e9,
            "cosmology_ns": self.params["ns"],
            "cosmology_kpivot_mpc": self.params["kpivot"],
            "cosmology_OmegaMNu": 0.0,

            "particle_Npart_1D": self.params["Npart"],

            "timestep_nsteps": [self.params["Nstep"]],

            # TODO: look into σ8 normalization stuff
            "ic_random_field_type": "gaussian",
            "ic_random_seed": self.params["seed"],
            "ic_fix_amplitude": True, # use P(k) when generating Gaussian random field # TODO: (?)
            "ic_use_gravity_model_GR": False, # don't use GR for backscaling P(k) in MG runs; instead be consistent with gravity model
            "ic_initial_redshift": self.params["zinit"],
            "ic_nmesh" : self.params["Ncell"],
            "ic_type_of_input": "powerspectrum", # transferinfofile only relevant with massive neutrinos?
            "ic_input_filename": "power_spectrum_today.dat",
            "ic_input_redshift": 0.0, # TODO: feed initial power spectrum directly instead of backscaling? Hans said someone incorporated this into his code?

            "force_nmesh": self.params["Ncell"],

            "output_folder": ".",
            "output_redshifts": [0.0],
            "output_particles": False,
            "pofk_nmesh": self.params["Ncell"], # TODO: ???
        }

    # run COLA simulation from back-scaling today's matter power spectrum (from CLASS)
    def run_cola(self, khs, Phs, np=1, verbose=True, ic="power_spectrum_today.dat", input="cola_input.lua", log="cola.log"):
        self.write_data(ic, {"k/(h/Mpc)": khs, "P/(Mpc/h)^3": Phs}) # COLA wants "h-units" # TODO: give cola the actual used h for ICs?
        self.write_file(input, "\n".join(f"{param} = {luastr(val)}" for param, val in self.params_cola().items()))
        cmd = ["mpirun", "-np", str(np), COLAEXEC, input] if np > 1 else [COLAEXEC, input] # TODO: ssh and run?
        self.run_command(cmd, log=log, verbose=True)

        assert self.completed_cola(), f"ERROR: see {log} for details"

    def power_spectrum(self, linear=False, hunits=False):
        k_h, Ph3 = self.read_data("class_pk.dat" if linear else f"pofk_{self.name}_cb_z0.000.txt", cols=(0, 1))

        if hunits:
            return k_h, Ph3
        else:
            k = k_h * self.params["h"]    # k / (1/Mpc)
            P = Ph3 / self.params["h"]**3 # P / Mpc^3
            return k, P

    # extend independent parameters used to run the sim with its derived parameters
    def params_extended(self):
        params_ext = self.params.copy()
        del params_ext["seed"] # only used internally; don't expose outside
        params_ext["σ8"] = self.read_variable("class.log", "sigma8=") # TODO: sigma8
        if "Ase9" not in params_ext:
            params_ext["Ase9"] = jsondict(self.read_file("parameters_extended.json"))["Ase9"] # TODO: find better way!
        if "ωb0" not in params_ext:
            params_ext["ωb0"] = params_ext["ωm0"] - params_ext["ωc0"]
        elif "ωc0" not in params_ext:
            params_ext["ωc0"] = params_ext["ωm0"] - params_ext["ωb0"]
        elif "ωm0" not in params_ext:
            params_ext["ωm0"] = params_ext["ωb0"] + params_ext["ωc0"]
        return params_ext

class GRSimulation(Simulation):
    def params_cola(self):
        return dictupdate(Simulation.params_cola(self), {
            "cosmology_model": "LCDM",
            "gravity_model": "GR",
        })

class BDSimulation(Simulation):
    def validate_input(self, params):
        Simulation.validate_input(self, params)

    def params_class(self):
        ω = 10 ** self.params["lgω"]
        return dictupdate(Simulation.params_class(self), {
            "gravity_model": "brans_dicke", # select BD gravity
            "Omega_Lambda": 0, # rather include Λ through potential term (first entry in parameters_smg; should be equivalent)
            "Omega_fld": 0, # no dark energy fluid
            "Omega_smg": -1, # automatic modified gravity
            "parameters_smg": f"NaN, {ω}, 1, 0", # ΩΛ0 (fill with cosmological constant), ω, Φini (arbitrary initial guess), Φ′ini≈0 (fixed)
            "M_pl_today_smg": (4+2*ω)/(3+2*ω) / self.params["G0/G"], # see https://github.com/HAWinther/hi_class_pub_devel/blob/3160be0e0482ac2284c20b8878d9a81efdf09f2a/gravity_smg/gravity_models_smg.c#L462
            "a_min_stability_test_smg": 1e-6, # BD has early-time instability, so lower tolerance to pass stability checker
            "output_background_smg": 2, # >= 2 needed to output phi to background table (https://github.com/miguelzuma/hi_class_public/blob/16ae0f6ccfcee513146ec36b690678f34fb687f4/source/background.c#L3031)
        })

    def params_cola(self):
        return dictupdate(Simulation.params_cola(self), {
            "gravity_model": "JBD",
            "cosmology_model": "JBD",
            "cosmology_JBD_wBD": 10 ** self.params["lgω"],
            "cosmology_JBD_GeffG_today": self.params["G0/G"],
        })

    def validate_output(self):
        Simulation.validate_output(self) # do any validation in parent class

        # Read background tables and their scale factors (which we use as the free time variable)
        z_class, H_class, ϕ_class, dϕ_dη_class = self.read_data("class_background.dat", dict=True, cols=("z", "H [1/Mpc]", "phi_smg", "phi'"))
        a_cola, ϕ_cola, dlogϕ_dloga_cola = self.read_data(f"cosmology_{self.name}.txt", dict=True, cols=("a", "phi", "dlogphi/dloga"))
        a_class  = 1 / (1 + z_class)

        # Compare ϕ
        check_values_are_close(ϕ_class, ϕ_cola, a_class, a_cola, name="ϕ", rtol=1e-5)

        # Compare dlogϕ/dloga
        dlogϕ_dloga_class = dϕ_dη_class / ϕ_class / (H_class * a_class) # convert by chain rule
        check_values_are_close(dlogϕ_dloga_class, dlogϕ_dloga_cola, a_class, a_cola, name="dlogϕ/dloga", atol=1e-4)

    def params_extended(self):
        params = Simulation.params_extended(self)
        params["ϕini"] = self.read_variable("class.log", "phi_ini = ")
        params["ϕ0"]   = self.read_variable("class.log", "phi_0 = ")
        params["ΩΛ0"]  = self.read_variable("class.log", "Lambda = ") / params["ϕ0"] # ρΛ0 / (3*H0^2*ϕ0/8*π)
        params["ωΛ0"]  = params["ΩΛ0"] * params["h"]**2 * params["ϕ0"]            # ∝ ρΛ0
        return params

# TODO: rather use a SimulationPairGroup when running pairs with same initial seed
class SimulationGroup:
    def __init__(self, simtype, params, nsims, seeds=None):
        if seeds is None:
            seeds = params2seeds(params, nsims)
        self.sims = [simtype(params, seed) for seed in seeds] # run simulations with all seeds

        # extend independent parameters with derived parameters
        self.params = self.sims[0].params_extended() # loop checks they are equal across sims
        for sim in self.sims:
            assert set(sim.params_extended().items()) == set(self.params.items()), f"simulations with same independent parameters have different dependent parameters"

    def __iter__(self): yield from self.sims
    def __len__(self): return len(self.sims)
    def __getitem__(self, key): return self.sims[key]

    def power_spectra(self, **kwargs):
        ks, Ps = [], []
        for sim in self:
            k, P = sim.power_spectrum(**kwargs)
            assert len(ks) == 0 or np.all(k == ks[0]), "group simulations output different k"
            ks.append(k)
            Ps.append(P)

        k = ks[0] # common wavenumbers for all simulations (by assertion)
        Ps = np.array(Ps) # 2D numpy array P[isim, ik]
        return k, Ps

    def power_spectrum(self, **kwargs):
        k, Ps = self.power_spectra(**kwargs)
        P  = np.mean(Ps, axis=0) # average            over simulations (for each k)
        ΔP = np.std( Ps, axis=0) # standard deviation over simulations (for each k)
        assert not linear or np.all(np.isclose(ΔP, 0.0, rtol=0, atol=1e-12)), "group simulations output different linear power spectra" # linear power spectrum is independent of seeds
        return k, P, ΔP

class SimulationGroupPair:
    def __init__(self, params_BD, params_BD_to_GR, nsims=1):
        seeds = params2seeds(params_BD, nsims) # BD parameters is a superset, so use them to make common seeds for BD and GR

        self.params_BD = params_BD
        self.sims_BD = SimulationGroup(BDSimulation, params_BD, nsims, seeds=seeds)

        self.params_GR = params_BD_to_GR(params_BD, self.sims_BD.params) # θGR = θGR(θBD)
        self.sims_GR = SimulationGroup(GRSimulation, self.params_GR, nsims, seeds=seeds)

        self.nsims = nsims

    def power_spectrum_ratio(self, linear=False):
        kBD, PBDs = self.sims_BD.power_spectra(linear=linear, hunits=False) # kBD / (1/Mpc), PBD / Mpc^3
        kGR, PGRs = self.sims_GR.power_spectra(linear=linear, hunits=False) # kGR / (1/Mpc), PGR / Mpc^3
        hBD = self.sims_BD.params["h"]
        hGR = self.sims_GR.params["h"]

        assert linear or np.all(np.isclose(kBD/hBD, kGR/hGR)) or self.sims_BD.params["Lh"] != self.sims_GR.params["Lh"], f"simulations with equal L*h should output equal non-linear k/h"
        PGRs = interp1d(kGR, PGRs, axis=1, kind="cubic", bounds_error=False, fill_value=np.nan)(kBD*hGR/hBD) # interpolate PGR(kGR) to kBD*hGR/hBD, # TODO: avoid? unnecessary in NL case. specify class' k-values?
        kGR = kBD * hGR/hBD # so we compare modes with kGR/hGR == kBD/hBD (has no effect on code, just for clarification)
        k = kBD * hBD # take the BD wavenumbers as the reference wavenumbers # TODO: or (kBD+kGR)/2 and interpolate both PBD and PGR?

        # from a statistical viewpoint,
        # we view P(k) as a random variable with samples from each simulation,
        # so it is more natural to index Ps[ik] == Ps[ik,:]
        PBDs = np.transpose(PBDs)
        PGRs = np.transpose(PGRs)

        # boost (of means)
        B = np.mean(PBDs/PGRs, axis=1)

        # boost error (propagate from errors in PBD and PGR)
        PBD = np.mean(PBDs, axis=1) # average over simulations
        PGR = np.mean(PGRs, axis=1) # average over simulations
        dB_dPBD =    1 / PGR    # dB/dPBD evaluated at means
        dB_dPGR = -PBD / PGR**2 # dB/dPGR evaluated at means
        ΔB = np.array([propagate_error([dB_dPBD[ik], dB_dPGR[ik]], [PBDs[ik], PGRs[ik]]) for ik in range(0, len(k))])

        # uncomment to compare matrix error propagation to manual expression (for one k value, to check it is correct)
        # (see formula for f=A/B at https://en.wikipedia.org/wiki/Propagation_of_uncertainty#Example_formulae)
        #σsq = np.cov([PBDs[0], PGRs[0]])
        #ΔB_matrix = ΔB[0]
        #ΔB_manual = B[0] * np.sqrt(σsq[0,0]/PBD[0]**2 + σsq[1,1]/PGR[0]**2 - 2*σsq[0,1]/(PBD[0]*PGR[0]))
        #assert np.isclose(ΔB_matrix, ΔB_manual), "error propagation is wrong"

        return k, B, ΔB

def θGR_identity(θBD, θBDext):
    return dictupdate(θBD, remove=["lgω", "G0/G"]) # remove BD-specific parameters

def θGR_different_h(θBD, θBDext):
    θGR = θGR_identity(θBD, θBDext)
    θGR["h"] = θBD["h"] * np.sqrt(θBDext["ϕini"]) # TODO: ensure similar Hubble evolution (of E=H/H0) during radiation domination
    return θGR

def plot_power_spectra(filename, sims, labelfunc = lambda params: None, colorfunc = lambda params: "black"):
    fig, ax = plt.subplots(figsize=(3.0, 2.7))
    ax.set_xlabel(r"$\lg[k / (1/\mathrm{Mpc})]$")
    ax.set_ylabel(r"$\lg[P / \mathrm{Mpc}^3]$")

    k, P, ΔP = sims.power_spectrum(linear=True)
    ax.plot(np.log10(k), np.log10(P), color="black", alpha=0.5, linewidth=1, linestyle="dashed")

    for i, sim in enumerate(sims):
        k, P = sim.power_spectrum(linear=False)
        ax.plot(np.log10(k), np.log10(P), color="black", alpha=0.5, linewidth=0.1, linestyle="solid")

    k, P, ΔP = sims.power_spectrum(linear=False)
    ax.fill_between(np.log10(k), np.log10(P-ΔP), np.log10(P+ΔP), color="black", alpha=0.2, edgecolor=None)
    ax.plot(        np.log10(k), np.log10(P),                    color="black", alpha=1.0, linewidth=1.0, linestyle="solid")

    ax.set_xlim(-2, 1)
    ax.set_ylim(1, 4)
    ax.set_xticks([-2, -1, 0, 1])  # also sets xlims
    ax.set_yticks([1, 2, 3, 4]) # also sets xlims
    ax.xaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator(10))
    ax.yaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator(10)) # one minor tick per 0.01

    fig.tight_layout()
    fig.savefig(filename)
    print("Plotted", filename)

# TODO: plot_single and plot_pair generic functions?

def plot_sequence(filename, paramss, nsims, θGR, labeltitle=None, labelfunc = lambda params: None, colorfunc = lambda params: "black", ax=None, divide_linear = False, logy = False):
    fig, ax = plt.subplots(figsize=(3.0, 2.7))

    ax.set_xlabel(r"$\lg\left[k_\mathrm{BD} / (1/\mathrm{Mpc})\right]$")
    ylabel = r"B(k_\mathrm{BD})"
    if divide_linear: ylabel = f"{ylabel} / B_\mathrm{{linear}}(k_\mathrm{{BD}})"
    if logy:          ylabel = f"\lg [ |{ylabel}| - 1 ]"
    ax.set_ylabel(f"${ylabel}$")

    # Dummy legend plot
    ax2 = ax.twinx() # use invisible twin axis to create second legend
    ax2.get_yaxis().set_visible(False) # make invisible
    ax2.fill_between([-4, -4], [0, 1], alpha=0.2, color="black", edgecolor=None,                  label=r"$B = \langle B \rangle \pm \Delta B$")
    ax2.plot(        [-4, -4], [0, 1], alpha=1.0, color="black", linestyle="solid",  linewidth=1, label=r"$B = \langle B \rangle$")
    ax2.plot(        [-4, -4], [0, 1], alpha=0.5, color="black", linestyle="dashed", linewidth=1, label=r"$B = B_\mathrm{lin}$")
    ax2.legend(loc="upper left", bbox_to_anchor=(-0.02, 0.97))

    for params_BD in paramss:
        sims = SimulationGroupPair(params_BD, θGR, nsims)

        klin, Blin, _  = sims.power_spectrum_ratio(linear=True)
        k,    B   , ΔB = sims.power_spectrum_ratio(linear=False)

        klin, Blin = klin[klin > 1e-3], Blin[klin > 1e-3]
        k, B, ΔB = k[k > 1e-3], B[k > 1e-3], ΔB[k > 1e-3]

        y    = B    / np.interp(k, klin, Blin) if divide_linear else B
        Δy   = ΔB   / np.interp(k, klin, Blin) if divide_linear else ΔB
        ylin = Blin /                    Blin  if divide_linear else Blin

        def T(y): return np.log10(np.abs(y-1)) if logy else y

        ax.plot(np.log10(klin),      T(ylin),          color=colorfunc(params_BD), alpha=1.0, linewidth=1, linestyle="dashed", label=None)
        ax.plot(np.log10(k),         T(y),             color=colorfunc(params_BD), alpha=1.0, linewidth=1, linestyle="solid",  label=None)
        ax.fill_between(np.log10(k), T(y-Δy), T(y+Δy), color=colorfunc(params_BD), alpha=0.2, edgecolor=None) # error band

    Δy = 1.0 if logy else 0.05
    ymin, ymax = ax.get_ylim()
    #ymin = np.log10(0.95) if logy else 0.95
    #ymax = np.log10(1.05) if logy else 1.05
    ymin = to_nearest(ymin, Δy, "floor")
    ymax = to_nearest(ymax, Δy, "ceil")

    ax.set_xticks([-3, -2, -1, 0, 1])
    ax.set_yticks(np.linspace(ymin, ymax, int(np.round((ymax-ymin)/Δy))+1)) # every Δy
    ax.set_xlim(-3, +1)
    ax.set_ylim(ymin, ymax)
    ax.xaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator(10)) # 10 minor ticks
    ax.yaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator(10)) # 10 minor ticks

    labels = [f"${labelfunc(params_BD)}$" for params_BD in paramss]
    colors = [colorfunc(params_BD) for params_BD in paramss]
    cax  = make_axes_locatable(ax).append_axes("top", size="7%", pad="0%") # align colorbar axis with plot
    cmap = matplotlib.colors.ListedColormap(colors)
    cbar = plt.colorbar(matplotlib.cm.ScalarMappable(cmap=cmap), cax=cax, orientation="horizontal")
    cbar.ax.set_title(labeltitle)
    cax.xaxis.set_ticks_position("top")
    cax.xaxis.set_tick_params(direction="out")
    cax.xaxis.set_ticks(np.linspace(0.5/len(labels), 1-0.5/len(labels), len(labels)), labels=labels)

    ax.grid(which="both")
    fig.tight_layout(pad=0)
    fig.savefig(filename)
    print("Plotted", filename)

def plot_convergence(filename, params0, param, vals, θGR, nsims=5, lfunc=None, cfunc=None, **kwargs):
    paramss = [dictupdate(params0, {param: val}) for val in vals] # generate all parameter combinations

    if lfunc is None:
        def lfunc(val): return f"{val}"
    def labelfunc(params): return lfunc(params[param])

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

    plot_sequence(filename, paramss, nsims, θGR, latex_labels[param], labelfunc, colorfunc, **kwargs)

def plot_quantity_evolution(filename, params0_BD, qty_BD, qty_GR, θGR, qty="", ylabel="", logabs=False, Δyrel=None, Δyabs=None):
    sims = SimulationGroupPair(params0_BD, θGR)

    bg_BD = sims.sims_BD[0].read_data("class_background.dat", dict=True)
    bg_GR = sims.sims_GR[0].read_data("class_background.dat", dict=True)

    # want to plot 1e-10 <= a <= 1, so cut away a < 1e-11
    bg_BD = {key: bg_BD[key][1/(bg_BD["z"]+1) > 1e-11] for key in bg_BD}
    bg_GR = {key: bg_GR[key][1/(bg_GR["z"]+1) > 1e-11] for key in bg_GR}

    a_BD = 1 / (bg_BD["z"] + 1) # = 1 / (z + 1)
    a_GR = 1 / (bg_GR["z"] + 1) # = 1 / (z + 1)

    aeq_BD = 1 / (sims.sims_BD[0].read_variable("class.log", "radiation/matter equality at z = ") + 1) # = 1 / (z + 1)
    aeq_GR = 1 / (sims.sims_GR[0].read_variable("class.log", "radiation/matter equality at z = ") + 1) # = 1 / (z + 1)

    fig, (ax1, ax2) = plt.subplots(2, 1, gridspec_kw={'height_ratios': (1, 1.618)}, figsize=(3.0, 3.5), sharex=True)
    ax2.set_xlabel(r"$\lg a$")
    ax2.set_ylabel(f"$\lg[{ylabel}]$" if logabs else f"${ylabel}$")

    ax1.axhline(1.0, color="gray", linestyle="dashed", linewidth=0.5)
    ax1.plot(np.log10(a_BD), qty_BD(bg_BD, sims.params_BD) / qty_GR(bg_GR, sims.params_GR), color="black")
    ax1.set_ylabel(f"${qty}_\mathrm{{BD}}(a) / {qty}_\mathrm{{GR}}(a)$")

    ax2.plot(np.log10(a_BD), np.log10(qty_BD(bg_BD, sims.params_BD)) if logabs else qty_BD(bg_BD, sims.params_BD), label=f"${qty}(a) = {qty}_\mathrm{{BD}}(a)$", color="blue")
    ax2.plot(np.log10(a_GR), np.log10(qty_GR(bg_GR, sims.params_GR)) if logabs else qty_GR(bg_GR, sims.params_GR), label=f"${qty}(a) = {qty}_\mathrm{{GR}}(a)$", color="red")

    ax2.set_xlim(-10, 0)
    ax2.set_xticks(np.linspace(-10, 0, 11))

    if Δyrel: ax_set_ylim_nearest(ax1, Δyrel)
    if Δyabs: ax_set_ylim_nearest(ax2, Δyabs)

    ymin, ymax = ax2.get_ylim()

    # mark some times
    for a_BD_GR, label in zip(((aeq_BD, aeq_GR),), (r"$\rho_r = \rho_m$",)):
        for a, color, dashoffset in zip(a_BD_GR, ("blue", "red"), (0, 5)):
            for ax in [ax1, ax2]:
                ax.axvline(np.log10(a), color=color, linestyle=(dashoffset, (5, 5)), alpha=0.5, linewidth=1.0)
        ax2.text(np.log10(np.average(a_BD_GR)) - 0.10, ymin + 0.5*(ymax-ymin), label, va="bottom", rotation=90, rotation_mode="anchor")

    ax1.xaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator(10))
    ax2.xaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator(10))
    ax1.yaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator(10))
    ax2.yaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator(10))

    ax1.yaxis.set_label_coords(-0.15, 0.5) # manually position to align for different invocations of this function
    ax2.yaxis.set_label_coords(-0.15, 0.5) # manually position to align for different invocations of this function

    ax2.legend()
    fig.subplots_adjust(left=0.17, right=0.99, bottom=0.10, top=0.98, hspace=0.1) # trial and error to get consistent plot layout...
    fig.savefig(filename)
    print("Plotted", filename)

def plot_density_evolution(filename, params0_BD, θGR):
    sims = SimulationGroupPair(params0_BD, θGR)

    z_BD, ργ_BD, ρν_BD, ρb_BD, ρc_BD, ρΛϕ_BD, ρcrit_BD = sims.sims_BD[0].read_data("class_background.dat", dict=True, cols=("z", "(.)rho_g", "(.)rho_ur", "(.)rho_b", "(.)rho_cdm", "(.)rho_smg",    "(.)rho_crit"))
    z_GR, ργ_GR, ρν_GR, ρb_GR, ρc_GR, ρΛ_GR,  ρcrit_GR = sims.sims_GR[0].read_data("class_background.dat", dict=True, cols=("z", "(.)rho_g", "(.)rho_ur", "(.)rho_b", "(.)rho_cdm", "(.)rho_lambda", "(.)rho_crit"))

    aeq_BD = 1 / (sims.sims_BD[0].read_variable("class.log", "radiation/matter equality at z = ") + 1) # = 1 / (z + 1)
    aeq_GR = 1 / (sims.sims_GR[0].read_variable("class.log", "radiation/matter equality at z = ") + 1) # = 1 / (z + 1)

    a_BD   = 1 / (z_BD + 1)
    Ωr_BD  = (ργ_BD + ρν_BD) / ρcrit_BD
    Ωm_BD  = (ρb_BD + ρc_BD) / ρcrit_BD
    ΩΛϕ_BD =  ρΛϕ_BD          / ρcrit_BD
    Ω_BD   = Ωr_BD + Ωm_BD + ΩΛϕ_BD

    a_GR   = 1 / (z_GR + 1)
    Ωr_GR  = (ργ_GR + ρν_GR) / ρcrit_GR
    Ωm_GR  = (ρb_GR + ρc_GR) / ρcrit_GR
    ΩΛ_GR  =  ρΛ_GR          / ρcrit_GR
    Ω_GR   = Ωr_GR + Ωm_GR + ΩΛ_GR

    fig, ax = plt.subplots(figsize=(6.0, 3.0))
    ax.plot(np.log10(a_BD), Ωr_BD,  color="C0",    linestyle="solid", alpha=0.8, label=r"$\Omega_r^\mathrm{BD}$")
    ax.plot(np.log10(a_BD), Ωm_BD,  color="C1",    linestyle="solid", alpha=0.8, label=r"$\Omega_m^\mathrm{BD}$")
    ax.plot(np.log10(a_BD), ΩΛϕ_BD, color="C2",    linestyle="solid", alpha=0.8, label=r"$\Omega_\Lambda^\mathrm{BD}+\Omega_\phi^\mathrm{BD}$")
    ax.plot(np.log10(a_BD), Ω_BD,   color="black", linestyle="solid", alpha=0.8, label=r"$\Omega^\mathrm{BD}$")
    ax.plot(np.log10(a_GR), Ωr_GR,  color="C0",    linestyle="dashed",  alpha=0.8, label=r"$\Omega_r^\mathrm{GR}$")
    ax.plot(np.log10(a_GR), Ωm_GR,  color="C1",    linestyle="dashed",  alpha=0.8, label=r"$\Omega_m^\mathrm{GR}$")
    ax.plot(np.log10(a_GR), ΩΛ_GR,  color="C2",    linestyle="dashed",  alpha=0.8, label=r"$\Omega_\Lambda^\mathrm{GR}$")
    ax.plot(np.log10(a_GR), Ω_GR,   color="black", linestyle="dashed",  alpha=0.8, label=r"$\Omega^\mathrm{GR}$")
    ax.set_xlabel(r"$\lg a$")
    ax.set_xlim(-7, 0)
    ax.set_ylim(0.0, 1.1)
    ax_set_ylim_nearest(ax, 0.1)
    ax.set_xticks(np.linspace(-7, 0, 8))
    ax.xaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator(10)) # minor ticks
    ax.yaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator(10)) # minor ticks
    ax.legend(ncol=2)
    fig.tight_layout()
    fig.savefig(filename)
    print("Plotted", filename)

def plot_parameter_samples(filename, samples, lo, hi, labels):
    params = list(lo.keys())
    dimension = len(params)

    fig, axs = plt.subplots(dimension-1, dimension-1, figsize=(6.0, 6.0))
    #fig = matplotlib.figure.Figure(figsize=(6.0, 6.0))
    #axs = fig.subplots(dimension-1, dimension-1)
    for iy in range(1, dimension):
        paramy = params[iy]
        sy = [sample[paramy] for sample in samples]
        for ix in range(0, dimension-1):
            paramx = params[ix]
            sx = [sample[paramx] for sample in samples]

            ax = axs[iy-1,ix]

            # plot faces (p1, p2); but not degenerate faces (p2, p1) or "flat faces" with p1 == p2
            if ix >= iy:
                ax.set_visible(False) # hide subplot
                continue

            ax.set_xlabel(latex_labels[paramx] if iy == dimension-1 else "")
            ax.set_ylabel(latex_labels[paramy] if ix == 0           else "")
            ax.set_xlim(lo[paramx], hi[paramx]) # [lo, hi]
            ax.set_ylim(lo[paramy], hi[paramy]) # [lo, hi]
            ax.set_xticks([lo[paramx], np.round((lo[paramx]+hi[paramx])/2, 10), hi[paramx]]) # [lo, mid, hi]
            ax.set_yticks([lo[paramy], np.round((lo[paramy]+hi[paramy])/2, 10), hi[paramy]]) # [lo, mid, hi]
            ax.set_xticklabels([f"${xtick}$" if iy == dimension-1 else "" for xtick in ax.get_xticks()], rotation=90) # rotation=45, ha="right", rotation_mode="anchor") # only on very bottom
            ax.set_yticklabels([f"${ytick}$" if ix == 0           else "" for ytick in ax.get_yticks()], rotation= 0) # rotation=45, ha="right", rotation_mode="anchor") # only on very left
            ax.xaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator(10)) # minor ticks
            ax.yaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator(10)) # minor ticks
            ax.xaxis.set_label_coords(+0.5, -0.5) # manually align xlabels vertically   (due to ticklabels with different size)
            ax.yaxis.set_label_coords(-0.5, +0.5) # manually align ylabels horizontally (due to ticklabels with different size)
            ax.grid()

            ax.scatter(sx, sy, 2.0, c="black", edgecolors="none")

    fig.suptitle(f"$\\textrm{{${len(samples)}$ Latin hypercube samples}}$")
    fig.tight_layout(pad=0, rect=(0.02, 0.02, 1.0, 1.0))
    fig.savefig(filename)
    print("Plotted", filename)

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
params0_BD = dictupdate(params0_GR, {
    "lgω":    2.0,    # lowest value to consider (larger values should only be "easier" to simulate?)
    "G0/G":   1.0,    # G0 == G        (ϕ0 = (4+2*ω)/(3+2*ω) * 1/(G0/G))
})

latex_labels = {
    "h":      r"$h$",
    "ωb0":    r"$\omega_{b0}$",
    "ωc0":    r"$\omega_{c0}$",
    "ωm0":    r"$\omega_{m0}$",
    "Lh":     r"$L / (\mathrm{Mpc}/h)$",
    "Npart":  r"$N_\mathrm{part}$",
    "Ncell":  r"$N_\mathrm{cell}$",
    "Nstep":  r"$N_\mathrm{step}$",
    "zinit":  r"$z_\mathrm{init}$",
    "lgω":    r"$\lg \omega$",
    "G0/G":   r"$G_0/G$",
    "Ase9":   r"$A_s / 10^{-9}$",
    "ns":     r"$n_s$",
    "σ8":     r"$\sigma(R=8\,\mathrm{Mpc}/h,\,z=0)$",
}
# TODO: make similar dict for value formatting?

# TODO: split up into different "run files"

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

# Check that CLASS and COLA outputs consistent background cosmology parameters
# for a "non-standard" BD cosmology with small wBD and G0/G != 1
# (using cheap COLA computational parameters, so the simulation finishes near-instantly)
#GRSimulation(params0_GR | {"Npart": 0, "Ncell": 16, "Nstep": 0, "Lh": 4})
#BDSimulation(params0_BD | {"Npart": 0, "Ncell": 16, "Nstep": 0, "Lh": 4, "lgω": 2.0, "G0/G": 1.1})
#exit()

#BDSimulation(params0_BD)
#GRSimulation(params0_GR)
#exit()

if "rcparams" in ACTIONS:
    print("Matplotlib rcParams:")
    print(matplotlib.rcParams.keys())

# List simulations
if "list" in ACTIONS:
    list_simulations()

# Power spectrum plots
#sims = SimulationGroup(BDSimulation, params0_BD, 1)
#sims = SimulationGroup(GRSimulation, params0_GR, 1)
#exit()
#plot_power_spectra("plots/power_spectra_fiducial.pdf", sims)
#exit()

# Plot evolution of (background) densities
if "evolution" in ACTIONS:
    plot_density_evolution("plots/evolution_density.pdf", params0_BD, θGR_different_h)

    # Plot evolution of (background) quantities
    def G_G0_BD(bg, params):    return (4+2*10**params["lgω"]) / (3+2*10**params["lgω"]) / bg["phi_smg"]
    def G_G0_GR(bg, params):    return np.ones_like(bg["z"]) # = 1.0, special case for GR
    def H_H0_BD_GR(bg, params): return bg["H [1/Mpc]"] / CubicSpline(np.log10(1/(bg["z"]+1)), bg["H [1/Mpc]"])(0.0) # common to BD and GR
    def D_Di_BD_GR(bg, params): return bg["gr.fac. D"] / CubicSpline(np.log10(1/(bg["z"]+1)), bg["gr.fac. D"])(-10.0) # common to BD and GR
    def f_BD_GR(bg, params):    return bg["gr.fac. f"] # common to BD and GR
    series = [
        ("G", G_G0_BD,    G_G0_GR,    False, "G(a)/G",           0.05, 0.05),
        ("H", H_H0_BD_GR, H_H0_BD_GR, True,  "H(a)/H_0",         5.0,  0.01),
        ("D", D_Di_BD_GR, D_Di_BD_GR, True,  "D(a)/D(10^{-10})", 1.0,  0.1),
        ("f", f_BD_GR,    f_BD_GR,    False, "f(a)",             0.1,  0.01),
    ]
    for q, qBD, qGR, logabs, ylabel, Δyabs, Δyrel in series:
        plot_quantity_evolution(f"plots/evolution_{q}.pdf", params0_BD, qBD, qGR, θGR_different_h, qty=q, ylabel=ylabel, logabs=logabs, Δyabs=Δyabs, Δyrel=Δyrel)

#plot_convergence(f"plots/boost_fiducial.pdf", params0_BD, "lgω", [2.0], nsims=5, θGR=θGR_different_h, paramlabel=latex_labels["lgω"])
#exit()

if "compare" in ACTIONS:
    for θGR, suffix1 in zip([θGR_identity, θGR_different_h], ["_sameh", "_diffh"]):
        for divide_linear, suffix2 in zip([False, True], ["", "_divlin"]):
            for logy, suffix3 in zip([False, True], ["", "_log"]):
                plot_convergence(f"plots/boost_fiducial{suffix1}{suffix2}{suffix3}.pdf", params0_BD, "lgω", [2.0, 3.0, 4.0, 5.0], nsims=5, θGR=θGR, divide_linear=divide_linear, logy=logy)

# Convergence plots (computational parameters)
if "convergence" in ACTIONS:
    plot_convergence("plots/convergence_L.pdf",     params0_BD, "Lh",     [256.0, 384.0, 512.0, 768.0, 1024.0], θGR_different_h, lfunc=lambda Lh:    f"${Lh:.0f}$",  cfunc=lambda Lh: np.log2(Lh))
    plot_convergence("plots/convergence_Npart.pdf", params0_BD, "Npart",  [256, 384, 512, 768, 1024],           θGR_different_h, lfunc=lambda Npart: f"${Npart}^3$", cfunc=lambda N: np.log2(N))
    plot_convergence("plots/convergence_Ncell.pdf", params0_BD, "Ncell",  [256, 384, 512, 768, 1024],           θGR_different_h, lfunc=lambda Ncell: f"${Ncell}^3$", cfunc=lambda N: np.log2(N))
    plot_convergence("plots/convergence_Nstep.pdf", params0_BD, "Nstep",  [10, 20, 30, 40, 50],                 θGR_different_h, lfunc=lambda Nstep: f"${Nstep}$")
    plot_convergence("plots/convergence_zinit.pdf", params0_BD, "zinit",  [10.0, 20.0, 30.0],                   θGR_different_h, lfunc=lambda zinit: f"${zinit:.0f}$")

# Variation plots (cosmological parameters)
if "variation" in ACTIONS:
    for param, value, prefix in (("Ase9", 2.1, "parametrize_As"), ("σ8", 0.8, "parametrize_s8")):
        params0 = dictupdate(params0_BD, {param: value})
        if True:
            plot_convergence(f"plots/variation_{prefix}_vary_omega.pdf",   params0, "lgω",  [2.0, 3.0, 4.0, 5.0],  θGR_different_h, lfunc=lambda lgω: f"${lgω}$")
            plot_convergence(f"plots/variation_{prefix}_vary_G0.pdf",      params0, "G0/G", [0.99, 1.0, 1.01],     θGR_different_h, lfunc=lambda G0_G: f"${G0_G:.02f}$")
            plot_convergence(f"plots/variation_{prefix}_vary_h.pdf",       params0, "h",    [0.63, 0.68, 0.73],    θGR_different_h, lfunc=lambda h: f"${h}$")
            plot_convergence(f"plots/variation_{prefix}_vary_omegab0.pdf", params0, "ωb0",  [0.016, 0.022, 0.028], θGR_different_h, lfunc=lambda ωb0: f"${ωb0}$")
            plot_convergence(f"plots/variation_{prefix}_vary_omegac0.pdf", params0, "ωc0",  [0.100, 0.120, 0.140], θGR_different_h, lfunc=lambda ωc0: f"${ωc0}$")
            plot_convergence(f"plots/variation_{prefix}_vary_ns.pdf",      params0, "ns",   [0.866, 0.966, 1.066], θGR_different_h, lfunc=lambda ns:  f"${ns}$")
        if "Ase9" in params0:
            plot_convergence(f"plots/variation_{prefix}_vary_As.pdf",      params0, "Ase9", [1.6, 2.1, 2.6],       θGR_different_h, lfunc=lambda Ase9: f"${Ase9}$")
        if "σ8" in params0:
            plot_convergence(f"plots/variation_{prefix}_vary_s8.pdf",      params0, "σ8",   [0.7, 0.8, 0.9],       θGR_different_h, lfunc=lambda σ8: f"${σ8}$")

        # parametrize with ωm0 and ωb0 (instead of ωc0)
        params0 = dictupdate(params0, remove=["ωc0", "ωb0"])
        plot_convergence(f"plots/variation_{prefix}_omegam0_vary_omegab0.pdf", params0 | {"ωm0": 0.142, "ωb0": 0.022}, "ωb0", [0.016, 0.022, 0.028], θGR_different_h, lfunc=lambda ωb0: f"${ωb0}$")
        plot_convergence(f"plots/variation_{prefix}_omegam0_vary_omegac0.pdf", params0 | {"ωm0": 0.142, "ωc0": 0.120}, "ωc0", [0.100, 0.120, 0.140], θGR_different_h, lfunc=lambda ωc0: f"${ωc0}$")
        plot_convergence(f"plots/variation_{prefix}_omegab0_vary_omegam0.pdf", params0 | {"ωm0": 0.142, "ωb0": 0.022}, "ωm0", [0.082, 0.142, 0.202], θGR_different_h, lfunc=lambda ωm0: f"${ωm0}$")
        #plot_convergence(f"plots/variation_omegam0_fixed_omegac0.pdf", params0 | {"ωm0": 0.142, "ωc0": 0.120}, "ωm0", [0.082, 0.142, 0.202], θGR_different_h, lfunc=lambda ωm0: f"${ωm0}$")

if "sample" in ACTIONS:
    paramspace = ParameterSpace(params_varying)
    samples = paramspace.samples(500)
    plot_parameter_samples("plots/parameter_samples.pdf", samples, paramspace.bounds_lo(), paramspace.bounds_hi(), latex_labels)
