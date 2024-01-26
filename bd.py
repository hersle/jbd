#!/usr/bin/env python3

import os
import re
import subprocess
import numpy as np
import argparse
from scipy.interpolate import CubicSpline

# TODO: optionally prefix with MPI stuff
DATADIR = "./data"
HICLASSEXEC = os.path.abspath(os.path.expandvars("$HOME/local/hi_class_public/class"))
EE2EXEC = os.path.abspath(os.path.expandvars("$HOME/local/EuclidEmulator2-pywrapper/ee2.exe"))

assert os.path.isdir(DATADIR), f"Data directory {DATADIR} does not exist; please create it!"
assert os.path.isfile(HICLASSEXEC), "hi_class executable {HICLASSEXEC} does not exist!"
assert os.path.isfile(EE2EXEC), "EuclidEmulator2 executable {EE2EXEC} does not exist!"

parser = argparse.ArgumentParser(prog="bd.py", argument_default=argparse.SUPPRESS) # suppress unspecified arguments (instead of including them with None values)
parser.add_argument("-z", "--z",   metavar="VAL", type=float, required=True, help="redshift") # TODO: take multiple redshifts?
parser.add_argument("-w", "--ω",   metavar="VAL", type=float, required=True, help="Brans-Dicke coupling")
parser.add_argument("-G", "--G0",  metavar="VAL", type=float, required=True, help="gravitational parameter today in units of Newton's constant (G0/GN)")
parser.add_argument("-m", "--ωm0", metavar="VAL", type=float, required=True, help="total matter density parameter today")
parser.add_argument("-b", "--ωb0", metavar="VAL", type=float, required=True, help="baryon density parameter today")
parser.add_argument("-H", "--h",   metavar="VAL", type=float, required=True, help="reduced Hubble parameter today")
parser.add_argument("-n", "--ns",  metavar="VAL", type=float, required=True, help="primordial power spectrum spectral index")

# specify As XOR σ8 (see https://stackoverflow.com/a/11155124)
group = parser.add_mutually_exclusive_group(required=True)
group.add_argument("-A", "--As",   metavar="VAL", type=float,                help="primordial power spectrum amplitude")
group.add_argument("-8", "--σ8",   metavar="VAL", type=float,                help="sigma8 today") # TODO: require this or As!

# run a command in the simulation's directory
def run_command(cmd, log=None, subdir=""):
    if log:
        cmd = f"{cmd} | tee {log}"
    print(f"Running {cmd}")
    subprocess.run(cmd, shell=True, check=True) # https://stackoverflow.com/a/45988305

def read_data(filename):
    return np.transpose(np.loadtxt(filename)) # transpose to index whole columns with [icol]

def write_file(filename, content):
    with open(filename, "w", encoding="utf-8") as file:
        file.write(content)

def read_file(filename):
    with open(filename, "r", encoding="utf-8") as file:
        return file.read()

# read a string like "[prefix][number]" from a file and return number
# example: if file contains "Omega_Lambda = 1.23", read_variable(filename, "Omega_Lambda = ") returns 1.23
# TODO: optimize for file, return on first match?
def read_variable(filename, prefix):
    regex = prefix + r"([-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?)"
    matches = re.findall(regex, read_file(filename))
    assert len(matches) == 1 # TODO: error message
    return float(matches[0][0])

class Universe:
    klin, Plin = None, None # linear power spectrum
    k,    P    = None, None # nonlinear power spectrum

    def __init__(self, params):
        self.params = params
        self.power_spectrum_linear() # need to run to calculate derived parameters
        self.params |= self.derived_parameters() # update with parameters derived from running class

    # TODO: return, write and read lines instead?
    def input_class(self):
        return '\n'.join([
            # cosmological parameters
            f"h = {self.params['h']}",
            f"Omega_b = {self.params['ωb0'] / self.params['h']**2}",
            f"Omega_cdm = {(self.params['ωm0'] - self.params['ωb0']) / self.params['h']**2}",
            f"A_s = {self.params['As']}" if 'As' in self.params else f"sigma8 = {self.params['σ8']}",
            f"n_s = {self.params['ns']}",

            # output control
            f"output = mPk", # output matter power spectrum P(k,z)
            f"z_pk = {self.params['z']}",
            f"root = data/class_",
            f"P_k_max_h/Mpc = 100.0", # output linear power spectrum to fill my plots

            # log verbosity (increase integers to make more talkative)
            f"input_verbose = 1", # log A_s
            f"background_verbose = 3", # log phi_ini and phi_0 (needed in BD)
            f"spectra_verbose = 1", # log sigma8
            f"", # final newline
        ])

    def power_spectrum_linear(self, infile=f"{DATADIR}/class_input.ini", logfile=f"{DATADIR}/class_log.txt"):
        if self.klin is None or self.Plin is None:
            write_file(infile, self.input_class())
            run_command(f"{HICLASSEXEC} {infile}", log=logfile)
            k_h, Ph3 = read_data(f"{DATADIR}/class_pk.dat") # k/(h/Mpc), P/(Mpc/h)^3
            self.klin = k_h * self.params["h"] # k/(1/Mpc)
            self.Plin = Ph3 / self.params["h"]**3 # P/Mpc^3
        return self.klin, self.Plin

    def derived_parameters(self):
        dparams = {}
        if "As" not in self.params:
            dparams["As"] = read_variable(f"{DATADIR}/class_log.txt", "A_s = ")
        if "σ8" not in self.params:
            dparams["σ8"] = read_variable(f"{DATADIR}/class_log.txt", "sigma8=")
        return dparams

class GRUniverse(Universe):
    def __init__(self, params):
        Universe.__init__(self, params)

    def power_spectrum_nonlinear(self):
        if self.k is None or self.P is None:
            h = self.params["h"]
            Ωb0 = self.params["ωb0"] / h**2
            Ωm0 = self.params["ωm0"] / h**2
            ns = self.params["ns"]
            As = self.params["As"]
            z = self.params["z"]
            run_command(f"{EE2EXEC} -b {Ωb0} -m {Ωm0} -n {ns} -H {h} -A {As} -z {z} -W -1.0 -w 0.0 -s 0.0 -d {DATADIR} -o ee2_boost", log=f"{DATADIR}/ee2_log.txt")
            outfile = f"{DATADIR}/ee2_boost0.dat"

            k_h, B = read_data(outfile) # k/(h/Mpc), Pnonlin/Plin
            k = k_h * self.params["h"] # k/(1/Mpc)
            self.k = self.klin
            self.P = CubicSpline(k, B, extrapolate=False)(self.klin) * self.Plin
            self.P[np.isnan(self.P)] = self.Plin[np.isnan(self.P)] # copy nans from Plin # TODO: Make sure no high-k nans

        return self.k, self.P

class BDUniverse(Universe):
    def __init__(self, params):
        Universe.__init__(self, params)

    def derived_parameters(self):
        return Universe.derived_parameters(self) | {
            "ϕini": read_variable(f"{DATADIR}/class_log.txt", "phi_ini = "),
        }

    def input_class(self):
        return Universe.input_class(self) + '\n'.join([
            f"gravity_model = brans_dicke", # select BD gravity
            f"Omega_Lambda = 0", # rather include Λ through potential term (first entry in parameters_smg; should be equivalent)
            f"Omega_fld = 0", # no dark energy fluid
            f"Omega_smg = -1", # automatic modified gravity
            f"parameters_smg = NaN, {self.params['ω']}, 1.0, 0.0", # ΩΛ0 (fill with cosmological constant), ω, Φini (arbitrary initial guess), Φ′ini≈0 (fixed)
            f"M_pl_today_smg = {(4+2*self.params['ω'])/(3+2*self.params['ω']) / self.params['G0']}", # see https://github.com/HAWinther/hi_class_pub_devel/blob/3160be0e0482ac2284c20b8878d9a81efdf09f2a/gravity_smg/gravity_models_smg.c#L462
            f"a_min_stability_test_smg = 1e-6", # BD has early-time instability, so lower tolerance to pass stability checker
            f"output_background_smg = 2", # >= 2 needed to output phi to background table (https://github.com/miguelzuma/hi_class_public/blob/16ae0f6ccfcee513146ec36b690678f34fb687f4/source/background.c#L3031)
            f"" # final newline
        ])

    def power_spectrum_nonlinear(self):
        if self.k is None or self.P is None:
            BD = self # this class instance is the BD universe (just for symmetry with GR variable)

            # transform parameters from BD to GR
            GR = GRUniverse({
                "z":   BD.params["z"],
                "ωm0": BD.params["ωm0"],
                "ωb0": BD.params["ωb0"],
                "h":   BD.params["h"] * np.sqrt(BD.params["ϕini"]), # transformed Hubble parameter!
                "ns":  BD.params["ns"],
                "σ8":  BD.params["σ8"], # same σ8, but different As!
            })

            # get good handles to all spectra that will be used below
            kGR,    PGR    = GR.power_spectrum_nonlinear()
            kGRlin, PGRlin = GR.power_spectrum_linear()
            kBDlin, PBDlin = BD.power_spectrum_linear()
            hGR, hBD = GR.params["h"], BD.params["h"]

            kBD = kGR # final k/(1/Mpc) # TODO: what is best choice?

            # calculate non-linear boost B = PBD(k/h)/PGR(k/h) != PBD(k)/PGR(k)
            # with linear spectra at common k/h = kBD/hBD values
            PBDlin = CubicSpline(kBDlin/hBD, PBDlin, extrapolate=False)(kBD/hBD) # P(k) -> P(k/h)
            PGRlin = CubicSpline(kGRlin/hGR, PGRlin, extrapolate=False)(kBD/hBD) # P(k) -> P(k/h)
            B = PBDlin / PGRlin # PBD(k/h) / PGR(k/h)
            PBD = B * PGR

            # remove NaNs due to possibly (slightly) different k-ranges
            self.k = kBD[np.isfinite(PBD)]
            self.P = PBD[np.isfinite(PBD)]

        return self.k, self.P

if __name__ == "__main__":
    args = parser.parse_args()

    paramsBD = vars(args)
    BD = BDUniverse(paramsBD)
    k, P = BD.power_spectrum_linear()
    assert len(k) == len(P)

    print("# k/(1/Mpc)     P/Mpc^3")
    for i in range(0, len(k)):
        print(f"{k[i]:.5e} {P[i]:.5e}")
