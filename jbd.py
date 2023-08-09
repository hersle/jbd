#!/usr/bin/env python3

# TODO: look at PPN to understand cosmological (large) -> solar system (small) scales of G in BD
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
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.stats import qmc

print("Matplotlib default rcParams:")
print(matplotlib.rcParams.keys())
matplotlib.rcParams |= {
    "text.usetex": True,
    "font.size": 9,
    "figure.figsize": (6.0, 4.0), # default (6.4, 4.8)
    "grid.linewidth": 0.3,
    "grid.alpha": 0.2,
    "legend.labelspacing": 0.2,
    "legend.columnspacing": 1.5,
    "legend.handlelength": 1.5,
    "legend.handletextpad": 0.3,
    "legend.frameon": False,
    "xtick.top": True,
    "ytick.right": True,
    "xtick.direction": "in",
    "ytick.direction": "in",
}

parser = argparse.ArgumentParser(prog="jbd.py")
parser.add_argument("--nbody", metavar="path/to/FML/FML/COLASolver/nbody", default="./FML/FML/COLASolver/nbody")
parser.add_argument("--class", metavar="path/to/hi_class_public/class", default="./hi_class_public/class")
args = parser.parse_args()

COLAEXEC = os.path.abspath(os.path.expanduser(vars(args)["nbody"]))
CLASSEXEC = os.path.abspath(os.path.expanduser(vars(args)["class"]))

def dictjson(dict, sort=False, unicode=False):
    return json.dumps(dict, sort_keys=sort, ensure_ascii=not unicode)

def hashstr(str):
    return hashlib.md5(str.encode('utf-8')).hexdigest()

def hashdict(dict):
    return hashstr(dictjson(dict, sort=True)) # https://stackoverflow.com/a/10288255

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
    for path in os.scandir("sims/"):
        if os.path.isdir(path):
            print(f"{path.name}: {BDSimulation(path=path.name).params}")

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
    def __init__(self, params=None, path=None):
        if path is not None:
            self.directory = "sims/" + path + "/"
            params = json.loads(self.read_file("parameters.json"))
            return Simulation.__init__(self, params=params)

        self.params = params.copy() # will be modified
        self.name = self.name()
        self.directory = "sims/" + self.name + "/"

        # initialize simulation, validate input, create working directory, write parameters
        print(f"Simulating {self.name} with independent parameters:")
        print("\n".join(f"{param} = {self.params[param]}" for param in sorted(self.params)))
        self.validate_input()
        os.makedirs(self.directory, exist_ok=True)
        self.write_file("parameters.json", dictjson(self.params, unicode=True))

        # create initial conditions with CLASS, store derived parameters, run COLA simulation
        # TODO: be lazy
        k, P = self.run_class()
        self.run_cola(k, P, np=16)

        # verify successful completion
        assert self.completed()
        self.validate_output()
        print(f"Simulated {self.name} with derived parameters:")
        params_extended = self.params_extended()
        print("\n".join(f"{param} = {params_extended[param]}" for param in sorted(params_extended)))

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
    def run_class(self, input="class_input.ini", log="class.log"):
        if not self.completed_class():
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
        if not self.completed_cola():
            self.write_data(ic, {"k/(h/Mpc)": khs, "P/(Mpc/h)^3": Phs}) # COLA wants "h-units" # TODO: give cola the actual used h for ICs?
            self.write_file(input, "\n".join(f"{param} = {luastr(val)}" for param, val in self.params_cola().items()))
            cmd = ["mpirun", "-np", str(np), COLAEXEC, input] if np > 1 else [COLAEXEC, input] # TODO: ssh and run?
            self.run_command(cmd, log=log, verbose=True)
        assert self.completed_cola(), f"ERROR: see {log} for details"

    def power_spectrum(self, linear=False, hunits=False):
        if linear:
            assert self.completed_class()
            data = self.read_data("class_pk.dat")
        else:
            assert self.completed_cola()
            data = self.read_data(f"pofk_{self.name}_cb_z0.000.txt") # pk: "total matter"; pk_cb: "cdm+b"; pk_lin: "total matter" (linear?)

        k, P = data[0], data[1] # k / (h/Mpc); P / (Mpc/h)^3 (in "h-units", common to CLASS and COLA)
        if not hunits:
            k = k / self.params["h"]    # k / Mpc
            P = P * self.params["h"]**3 # P / Mpc^3

        return k, P

    # extend independent parameters used to run the sim with its derived parameters
    def params_extended(self):
        return self.params.copy()

class GRSimulation(Simulation):
    def params_cola(self):
        return Simulation.params_cola(self) | { # combine dictionaries
            "cosmology_model": "LCDM",
            "gravity_model": "GR",
        }

class BDSimulation(Simulation):
    def validate_input(self):
        Simulation.validate_input(self)

    def params_class(self):
        ω = 10 ** self.params["lgω"]
        return Simulation.params_class(self) | { # combine dictionaries
            "gravity_model": "brans_dicke", # select BD gravity
            "Omega_Lambda": 0, # rather include Λ through potential term (first entry in parameters_smg; should be equivalent)
            "Omega_fld": 0, # no dark energy fluid
            "Omega_smg": -1, # automatic modified gravity
            "parameters_smg": f"NaN, {ω}, 1, 0", # ΩΛ0 (fill with cosmological constant), ω, Φini (arbitrary initial guess), Φ′ini≈0 (fixed)
            "M_pl_today_smg": (4+2*ω)/(3+2*ω) / self.params["G0/G"], # see https://github.com/HAWinther/hi_class_pub_devel/blob/3160be0e0482ac2284c20b8878d9a81efdf09f2a/gravity_smg/gravity_models_smg.c#L462
            "a_min_stability_test_smg": 1e-6, # BD has early-time instability, so lower tolerance to pass stability checker
            "output_background_smg": 2, # >= 2 needed to output phi to background table (https://github.com/miguelzuma/hi_class_public/blob/16ae0f6ccfcee513146ec36b690678f34fb687f4/source/background.c#L3031)
        }

    def params_cola(self):
        return Simulation.params_cola(self) | { # combine dictionaries
            "gravity_model": "JBD",
            "cosmology_model": "JBD",
            "cosmology_JBD_wBD": 10 ** self.params["lgω"],
            "cosmology_JBD_GeffG_today": self.params["G0/G"],
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

    def params_extended(self):
        params = Simulation.params_extended(self)
        params["ϕini"] = self.read_variable("class.log", "phi_ini")
        params["ϕ0"]   = self.read_variable("class.log", "phi_0")
        params["ΩΛ0"]  = self.read_variable("class.log", "Lambda") / params["ϕ0"] # ρΛ0 / (3*H0^2*ϕ0/8*π)
        params["ωΛ0"]  = params["ΩΛ0"] * params["h"]**2 * params["ϕ0"]            # ∝ ρΛ0
        return params

# TODO: rather use a SimulationPairGroup when running pairs with same initial seed
class SimulationGroup:
    def __init__(self, simtype, params0, nsims, hash=None):
        if hash is None:
            hash = hashdict(params0) # unique hash of base (without seed) simulation parameters
            hash = int(hash, 16)     # convert MD5 hexadecimal (base 16) hash to integer (needed to seed numpy's rng)
        rng = np.random.default_rng(hash) # deterministic random number generator from simulation parameters
        seeds = rng.integers(0, 2**31-1, size=nsims, dtype=int) # will be the same for the same simulation parameters
        seeds = [int(seed) for seed in seeds] # convert to python ints to make compatible with JSON dict hashing
        self.sims = [simtype(params0 | {"seed": seed}) for seed in seeds] # run simulations with all seeds

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
    def __init__(self, params_BD, nsims):
        # choose hash so each simulation in PBD/PGR is run with the same seed and thus similar initial conditions
        hash = hashdict(params_BD) # BD parameters is a superset, so use them to make a hash for both BD and GR
        hash = int(hash, 16) # make an integer out of it

        self.sims_BD = SimulationGroup(BDSimulation, params_BD, nsims, hash=hash)
        params_BD_derived = self.sims_BD[0].params_extended()
        del params_BD_derived["seed"]

        # make sure all BD simulations (except the appended random seed) have the same derived parameters
        for sim_BD in self.sims_BD:
            params_BD_derived_candidate = sim_BD.params_extended()
            del params_BD_derived_candidate["seed"]
            assert params_BD_derived_candidate == params_BD_derived

        params_GR = parameters_BD_to_GR(params_BD, params_BD_derived) # θGR = θGR(θBD)
        self.sims_GR = SimulationGroup(GRSimulation, params_GR, nsims, hash=hash)

        self.nsims = nsims

    def power_spectrum_ratio(self, linear=False):
        kBD, PBDs = self.sims_BD.power_spectra(linear=linear, hunits=True) # kBD / (hBD/Mpc), PBD / (Mpc/hBD)^3
        kGR, PGRs = self.sims_GR.power_spectra(linear=linear, hunits=True) # kGR / (hGR/Mpc), PGR / (Mpc/hGR)^3

        if linear:
            # class outputs linear P at slightly different k/(h/Mpc) in BD and GR,
            # so interpolate PGR(kGR) -> PGR(kBD)
            # TODO: avoid interpolation by make class output linear P at same kBD and kGR (in hunits)
            for isim in range(0, self.nsims):
                PGRs[isim,:] = np.interp(kBD, kGR, PGRs[isim,:]) # interpolate PGR(kGR) to kBD values # TODO: avoid
        else:
            # kBD / hBD == kGR / hGR if the simulations are run with equal L / (Mpc/h)
            assert np.all(kBD == kGR), f"simulations output different k-values: {kBD[:3]}...{kBD[-3:]} vs {kGR[:3]}...{kGR[-3:]}"
        k = kBD / self.sims_BD[0].params["h"] # take the BD wavenumbers as the reference wavenumbers

        # from a statistical viewpoint,
        # we view P(k) as a random variable with samples from each simulation,
        # so it is more natural to index Ps[ik] == Ps[ik,:]
        PBDs = np.transpose(PBDs)
        PGRs = np.transpose(PGRs)

        # boost (of means)
        PBD = np.mean(PBDs, axis=1) # average over simulations
        PGR = np.mean(PGRs, axis=1) # average over simulations
        B = np.mean(PBDs/PGRs, axis=1)

        # boost error (propagate from errors in PBD and PGR)
        dB_dPBD =    1 / PGR    # dB/dPBD evaluated at means
        dB_dPGR = -PBD / PGR**2 # dB/dPGR evaluated at means
        ΔB = np.array([propagate_error([dB_dPBD[ik], dB_dPGR[ik]], [PBDs[ik], PGRs[ik]]) for ik in range(0, len(k))])

        # uncomment to compare matrix error propagation to manual expression (for one k value, to check it is correct)
        # (see formula for f=A/B at https://en.wikipedia.org/wiki/Propagation_of_uncertainty#Example_formulae)
        #σsq = np.cov([PBDs[0], PGRs[0]])
        #ΔB_matrix = ΔB[0]
        #ΔB_manual = B[0] * np.sqrt(σsq[0,0]/PBD[0]**2 + σsq[1,1]/PGR[0]**2 - 2*σsq[0,1]/(PBD[0]*PGR[0]))
        #assert np.isclose(ΔB_matrix, ΔB_manual), "error propagation is wrong"

        # remove h factoring # TODO: correct?
        hBD = self.sims_BD[0].params["h"]
        hGR = self.sims_GR[0].params["h"]
        B  *= (hGR / hBD)**3
        ΔB *= (hGR / hBD)**3

        return k, B, ΔB

def parameters_BD_to_GR(params_BD, params_BD_derived):
    params_GR = params_BD.copy() # don't modify input

    del params_GR["lgω"]  # remove BD-specific parameter
    del params_GR["G0/G"] # remove BD-specific parameter
    params_GR["h"] = params_BD["h"] * np.sqrt(params_BD_derived["ϕini"]) # TODO: ensure similar Hubble evolution (of E=H/H0) during radiation domination

    return params_GR

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

def plot_sequence(filename, paramss, nsims, labeltitle=None, labelfunc = lambda params: None, colorfunc = lambda params: "black", ax=None, divide_linear = False):
    fig, ax = plt.subplots(figsize=(3.0, 2.7))

    ax.set_xlabel(r"$\lg\left[k_\mathrm{BD} / (1/\mathrm{Mpc})\right]$")
    ax.set_ylabel(r"$B(k_\mathrm{BD})" + (" / B_\mathrm{linear}(k_\mathrm{BD})" if divide_linear else "") + "$")

    # Dummy legend plot
    ax2 = ax.twinx() # use invisible twin axis to create second legend
    ax2.get_yaxis().set_visible(False) # make invisible
    ax2.fill_between([-3, -2], [0, 1], alpha=0.2, color="black", edgecolor=None,                  label=r"$B = \langle B \rangle \pm \Delta B$")
    ax2.plot(        [-3, -2], [0, 1], alpha=1.0, color="black", linestyle="solid",  linewidth=1, label=r"$B = \langle B \rangle$")
    ax2.plot(        [-3, -2], [0, 1], alpha=0.5, color="black", linestyle="dashed", linewidth=1, label=r"$B = B_\mathrm{lin}$")
    ax2.legend(loc="upper left", bbox_to_anchor=(-0.02, 0.97))

    for params_BD in paramss:
        sims = SimulationGroupPair(params_BD, nsims)

        klin, Blin, _  = sims.power_spectrum_ratio(linear=True)
        k,    B   , ΔB = sims.power_spectrum_ratio(linear=False)

        y = B
        Δy = ΔB
        ylin = Blin
        if divide_linear:
            y  /= np.interp(k, klin, Blin)
            Δy /= np.interp(k, klin, Blin)
            ylin /= Blin

        ax.plot(np.log10(klin), ylin, color=colorfunc(params_BD), alpha=1.0, linewidth=1, linestyle="dashed", label=None)
        ax.plot(np.log10(k),    y,    color=colorfunc(params_BD), alpha=1.0, linewidth=1, linestyle="solid",  label=None)
        ax.fill_between(np.log10(k), y-Δy, y+Δy, color=colorfunc(params_BD), alpha=0.2, edgecolor=None) # error band

    ax.set_xlim(-2, +1)
    ax.set_ylim(0.94-1e-10, 1.06+1e-10) # +/- 1e-10 shows first and last minor tick
    ax.set_xticks([-2, -1, 0, 1])
    ax.set_yticks([0.95, 1.0, 1.05])
    ax.xaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator(10)) # 10 minor ticks
    ax.yaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator(10)) # 10 minor ticks

    labels = [labelfunc(params_BD) for params_BD in paramss]
    colors = [colorfunc(params_BD) for params_BD in paramss]
    cax  = make_axes_locatable(ax).append_axes("top", size="7%", pad="0%") # align colorbar axis with plot
    cmap = matplotlib.colors.ListedColormap(colors)
    cbar = plt.colorbar(matplotlib.cm.ScalarMappable(cmap=cmap), cax=cax, orientation="horizontal")
    cbar.ax.set_title(labeltitle)
    cax.xaxis.set_ticks_position("top")
    cax.xaxis.set_ticks(np.linspace(0.5/len(labels), 1-0.5/len(labels), len(labels)), labels=labels)

    ax.grid(which="both")
    fig.tight_layout(pad=0)
    fig.savefig(filename)
    print("Plotted", filename)

def plot_convergence(filename, params0, param, vals, nsims=5, paramlabel=None, lfunc=None, cfunc=None, **kwargs):
    paramss = [params0 | {param: val} for val in vals] # generate all parameter combinations

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

    plot_sequence(filename, paramss, nsims, paramlabel, labelfunc, colorfunc, **kwargs)

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
    "Lh":    512.0, # L / (Mpc/h)
}
params0_BD = params0_GR | {
    "lgω":    2.0,    # lowest value to consider (larger values should only be "easier" to simulate?)
    "G0/G":   1.0,    # G0 == G        (ϕ0 = (4+2*ω)/(3+2*ω) * 1/(G0/G))
}

latex_labels = {
    "h":      r"$h$",
    "ωb0":    r"$\omega_{b0}$",
    "ωc0":    r"$\omega_{c0}$",
    "Lh":     r"$L / (\mathrm{Mpc}/h)$",
    "Npart":  r"$N_\mathrm{part}$",
    "Ncell":  r"$N_\mathrm{cell}$",
    "Nstep":  r"$N_\mathrm{step}$",
    "zinit":  r"$z_\mathrm{init}$",
    "lgω":    r"$\lg \omega$",
    "G0/G":   r"$G_0/G$",
    "Ase9":   r"$A_s / 10^{-9}$",
    "ns":     r"$n_s$",
}

# TODO: split up into different "run files"

# List simulations
#list_simulations()
#exit()

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
#paramspace = ParameterSpace(params_varying)
#samples = paramspace.samples(500)
#plot_parameter_samples("plots/parameter_samples.pdf", samples, paramspace.bounds_lo(), paramspace.bounds_hi(), latex_labels)
#exit()

# Check that CLASS and COLA outputs consistent background cosmology parameters
# for a "non-standard" BD cosmology with small wBD and G0/G != 1
# (using cheap COLA computational parameters, so the simulation finishes near-instantly)
#GRSimulation(params0_GR | {"Npart": 0, "Ncell": 16, "Nstep": 0, "Lh": 4})
#BDSimulation(params0_BD | {"Npart": 0, "Ncell": 16, "Nstep": 0, "Lh": 4, "lgω": 2.0, "G0/G": 1.1})
#exit()

#BDSimulation(params0_BD)
#GRSimulation(params0_GR)
#exit()

# Power spectrum plots
#sims = SimulationGroup(BDSimulation, params0_BD, 1)
#sims = SimulationGroup(GRSimulation, params0_GR, 1)
#exit()
#plot_power_spectra("plots/power_spectra_fiducial.pdf", sims)
#exit()

plot_convergence("plots/boost_fiducial.pdf", params0_BD, "lgω", [2.0], nsims=1)
exit()

# Convergence plots (computational parameters)
plot_convergence("plots/convergence_L.pdf",     params0_BD, "Lh",     [256.0, 384.0, 512.0, 768.0, 1024.0], paramlabel=latex_labels["Lh"],    lfunc=lambda Lh:    f"${Lh:.0f}$", cfunc=lambda Lh: np.log2(Lh))
plot_convergence("plots/convergence_Npart.pdf", params0_BD, "Npart",  [256, 384, 512, 768, 1024],           paramlabel=latex_labels["Npart"], lfunc=lambda Npart: f"${Npart}^3$")
plot_convergence("plots/convergence_Ncell.pdf", params0_BD, "Ncell",  [256, 384, 512, 768, 1024],           paramlabel=latex_labels["Ncell"], lfunc=lambda Ncell: f"${Ncell}^3$")
plot_convergence("plots/convergence_Nstep.pdf", params0_BD, "Nstep",  [10, 20, 30, 40, 50],                 paramlabel=latex_labels["Nstep"], lfunc=lambda Nstep: f"${Nstep}$")
plot_convergence("plots/convergence_zinit.pdf", params0_BD, "zinit",  [10.0, 20.0, 30.0],                   paramlabel=latex_labels["zinit"], lfunc=lambda zinit: f"${zinit:.0f}$")

# Variation plots (cosmological parameters)
plot_convergence("plots/convergence_omega.pdf",   params0_BD, "lgω",    [2.0, 3.0, 4.0, 5.0],     paramlabel=latex_labels["lgω"],    lfunc=lambda lgω: f"${lgω}$")
plot_convergence("plots/convergence_G0.pdf",      params0_BD, "G0/G",   [0.99, 1.0, 1.01],        paramlabel=latex_labels["G0/G"],   lfunc=lambda G0_G: f"${G0_G:.02f}$")
plot_convergence("plots/convergence_h.pdf",       params0_BD, "h",      [0.63, 0.68, 0.73],       paramlabel=latex_labels["h"],      lfunc=lambda h: f"${h}$")
plot_convergence("plots/convergence_omegab0.pdf", params0_BD, "ωb0",    [0.016, 0.022, 0.028],    paramlabel=latex_labels["ωb0"],    lfunc=lambda ωb0: f"${ωb0}$")
plot_convergence("plots/convergence_omegac0.pdf", params0_BD, "ωc0",    [0.090, 0.120, 0.150],    paramlabel=latex_labels["ωc0"],    lfunc=lambda ωc0: f"${ωc0}$")
plot_convergence("plots/convergence_As.pdf",      params0_BD, "Ase9",   [1.6, 2.1, 2.6],          paramlabel=latex_labels["Ase9"],   lfunc=lambda Ase9: f"${Ase9}$")
plot_convergence("plots/convergence_ns.pdf",      params0_BD, "ns",     [0.866, 0.966, 1.066],    paramlabel=latex_labels["ns"],     lfunc=lambda ns:  f"${ns}$")
exit()
