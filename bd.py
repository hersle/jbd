#!/usr/bin/env python3

import os
import re
import sys
import argparse
import subprocess
import numpy as np
from scipy.interpolate import CubicSpline

parser = argparse.ArgumentParser(prog="bd.py", formatter_class=argparse.ArgumentDefaultsHelpFormatter) # suppress unspecified arguments (instead of including them with None values)

# cosmological parameters
parser.add_argument("-z",          metavar="VAL", type=float, required=True, nargs="+", help="redshift(s)")
parser.add_argument("-w", "--ω",   metavar="VAL", type=float, required=True,            help="Brans-Dicke coupling")
parser.add_argument("-G", "--G0",  metavar="VAL", type=float, required=True, default=1, help="gravitational parameter today in units of Newton's constant [G0/GN]")
parser.add_argument("-H", "--h",   metavar="VAL", type=float, required=True,            help="reduced Hubble parameter today [H0/(100km/(s*Mpc))]")
parser.add_argument("-m", "--ωm0", metavar="VAL", type=float, required=True,            help="physical matter density parameter today [ρm0*h^2/(3*H0^2/8*π*GN)]")
parser.add_argument("-b", "--ωb0", metavar="VAL", type=float, required=True,            help="physical baryon density parameter today [ρb0*h^2/(3*H0^2/8*π*GN)]")
parser.add_argument("-n", "--ns",  metavar="VAL", type=float, required=True,            help="primordial power spectrum spectral index")
group = parser.add_mutually_exclusive_group(required=True) # specify As XOR σ8
group.add_argument("-A", "--As",   metavar="VAL", type=float, default=argparse.SUPPRESS, help="primordial power spectrum amplitude")
group.add_argument("-8", "--σ8",   metavar="VAL", type=float, default=argparse.SUPPRESS, help="density fluctuation amplitude over 8Mpc/h scale today")

# directory and executable paths, verbosity and output file
parser.add_argument("--data",    metavar="PATH", default=".",       help="working directory for intermediate data/log files")
parser.add_argument("--hiclass", metavar="PATH", default="class",   help="hi_class executable")
parser.add_argument("--ee2",     metavar="PATH", default="ee2.exe", help="EuclidEmulator2 executable")
parser.add_argument("--output",  metavar="PATH", default="-",       help="output filename")
parser.add_argument("--verbose", action="store_true", help="print commands run and their output")

# desired power spectrum
parser.add_argument("spectrum", metavar="SPECTRUM", type=str, help="desired nonlinear/linear power spectrum PGR/PBD/PGRlin/PBDlin")

args = parser.parse_args()

# run a command and optionally log its output to a file
def run_command(cmd, log=None):
    if args.verbose:
        print(f"Running command {cmd}") # user wants to see what is run
    run = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, encoding="utf-8") # combine stderr -> stdout
    if run.returncode != 0:
        print(run.stdout) # print output
        run.check_returncode() # then raise exception
    if args.verbose:
        print(run.stdout) # user wants to see output
    if log:
        write_file(log, run.stdout) # log outpuit to file

# read tabular data from a file
def read_data(filename):
    return np.transpose(np.loadtxt(filename)) # transpose to index whole columns with [icol]

# write a string to a file
def write_file(filename, content):
    with open(filename, "w", encoding="utf-8") as file:
        file.write(content)

# read a string from a file
def read_file(filename):
    with open(filename, "r", encoding="utf-8") as file:
        return file.read()

# read a string like "[prefix][number]" from a file and return number
# example: if file contains "phi_ini = 1.23", then read_variable(filename, "phi_ini = ") returns 1.23
def read_variable(filename, prefix):
    regex = prefix + r"([-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?)" # the latter matches a number
    matches = re.findall(regex, read_file(filename))
    assert len(matches) == 1, f"found {len(matches)} != 1 for variable \"{prefix}\" in {filename}"
    return float(matches[0][0])

class GRUniverse:
    klin, Plin = None, None # linear power spectrum (saved to avoid repeated calculation after constructor)

    def __init__(self, params):
        self.params = params
        self.power_spectrum_linear() # need to run hi_class to calculate derived parameters
        self.params |= self.derived_parameters() # update with parameters derived from running hi_class

    # returns content of input file to run hiclass with
    def input_class(self):
        return '\n'.join([
            # cosmological parameters
            f"h = {self.params['h']}",
            f"omega_b = {self.params['ωb0']}",
            f"omega_cdm = {self.params['ωm0'] - self.params['ωb0']}",
            f"A_s = {self.params['As']}" if 'As' in self.params else f"sigma8 = {self.params['σ8']}",
            f"n_s = {self.params['ns']}",

            # output control
            f"output = mPk", # output matter power spectrum P(k,z)
            f"z_pk = {','.join(str(z) for z in self.params['z'])}",
            f"root = {args.data}/class_",
            f"P_k_max_h/Mpc = 20.0",

            # log verbosity (increase integers to make more talkative)
            f"input_verbose = 1", # log A_s
            f"background_verbose = 3", # log phi_ini and phi_0 (needed in BD)
            f"spectra_verbose = 1", # log sigma8
        ]) + '\n' # final newline

    # calculates linear power spectrum from hi_class
    def power_spectrum_linear(self, k=None, infile=f"{args.data}/class_input.ini", logfile=f"{args.data}/class_log.txt"):
        if self.klin is None or self.Plin is None:
            write_file(infile, self.input_class())
            run_command([args.hiclass, infile], log=logfile)

            # class outputs different files depending on whether there is one or more redshifts :/
            if len(self.params["z"]) == 1:
                filenames = [f"{args.data}/class_pk.dat"]
            else:
                filenames = (f"{args.data}/class_z{i}_pk.dat" for i, _ in enumerate(self.params["z"], start=1))

            # read output k, P at all redshifts z
            self.klin, self.Plin = [], []
            for filename in filenames:
                klin, Plin = read_data(filename) # k/(h/Mpc), P/(Mpc/h)^3
                self.klin.append(klin) # k/(h/Mpc)
                self.Plin.append(Plin / self.params["h"]**3) # P/Mpc^3
            self.klin, self.Plin = np.array(self.klin), np.array(self.Plin) # convert to numpy arrays

            # verify that k are the same at every redshift, and proceed with common k
            assert np.all(self.klin == self.klin[0]), "hi_class output different k at different z"
            self.klin = self.klin[0]

        if k is None:
            k, P = self.klin, self.Plin
        else:
            P = CubicSpline(self.klin, self.Plin, axis=1, extrapolate=False)(k) # interpolate to requested k-values
        return k, P

    # calculates nonlinear power spectrum by boosting linear power spectrum with EuclidEmulator2
    def power_spectrum_nonlinear(self):
        cmd = [
            args.ee2, "-d", args.data, "-o", "ee2_boost",
            "-b", str(self.params["ωb0"] / self.params["h"]**2), # Ωb0 = ωb0/h^2
            "-m", str(self.params["ωm0"] / self.params["h"]**2), # Ωm0 = ωm0/h^2
            "-n", str(self.params["ns"]),
            "-H", str(self.params["h"]),
            "-A", str(self.params["As"]),
            "-W", str(-1.0), # TODO: add?
            "-w", str(0.0), # TODO: add?
            "-s", str(0.0), # TODO: add?
        ]
        for z in self.params["z"]:
            cmd += ["-z", str(z)]
        run_command(cmd, log=f"{args.data}/ee2_log.txt")
        data = read_data(f"{args.data}/ee2_boost0.dat") # k/(h/Mpc), Pnonlin/Plin
        kB, B = data[0], data[1:] # k/(h/Mpc)

        k, Plin = self.power_spectrum_linear() # will use these k for output
        k, Plin = k[k<=kB[-1]], Plin[:, k<=kB[-1]] # discard k above boost's k-range
        B = CubicSpline(kB, B, axis=1, extrapolate=False)(k) # interpolate B to output k (EE2's python wrapper interpolates logarithmically, but this seems to have negligible effect)
        B[:, k<kB[0]] = 1 # take linear spectra for k below boost's k-range (overwrites NaNs)
        P = B * Plin # boost from linear to nonlinear power spectrum
        return k, P

    # calculates additional (derived) parameters by reading hiclass' log
    def derived_parameters(self):
        dparams = {}
        if "As" not in self.params:
            dparams["As"] = read_variable(f"{args.data}/class_log.txt", "A_s = ")
        if "σ8" not in self.params:
            dparams["σ8"] = read_variable(f"{args.data}/class_log.txt", "sigma8=")
        return dparams

class BDUniverse(GRUniverse):
    def __init__(self, params):
        GRUniverse.__init__(self, params)

    # extend hiclass input file from GR to BD
    def input_class(self):
        return GRUniverse.input_class(self) + '\n'.join([
            f"gravity_model = brans_dicke", # select BD gravity
            f"Omega_Lambda = 0", # rather include Λ through potential term (first entry in parameters_smg; should be equivalent)
            f"Omega_fld = 0", # no dark energy fluid
            f"Omega_smg = -1", # automatic modified gravity
            f"parameters_smg = NaN, {self.params['ω']}, 1.0, 0.0", # ΩΛ0 (fill with cosmological constant), ω, Φini (arbitrary initial guess), Φ′ini≈0 (fixed)
            f"M_pl_today_smg = {(4+2*self.params['ω'])/(3+2*self.params['ω']) / self.params['G0']}", # overwrites ϕini=1.0 above # see e.g. https://github.com/miguelzuma/hi_class_public/blob/16ae0f6ccfcee513146ec36b690678f34fb687f4/source/input.c#L1796
            f"a_min_stability_test_smg = 1e-6", # BD has early-time instability, so lower tolerance to pass stability checker
        ]) + '\n' # final newline

    # extend derived parameters with ϕini to facilitate BD-to-GR parameter transformation
    def derived_parameters(self):
        return GRUniverse.derived_parameters(self) | {
            "ϕini": read_variable(f"{args.data}/class_log.txt", "phi_ini = "),
        }

    # returns a GR universe with transformed parameters such that PBD/PGR ≈ PBDlin/PGRlin
    def transformed_GR_universe(self):
        BD = self # just for clarity; the BD universe is this class instance
        return GRUniverse({ # GR universe with transformed parameters
            "z":   BD.params["z"],
            "ωm0": BD.params["ωm0"],
            "ωb0": BD.params["ωb0"],
            "h":   BD.params["h"] * np.sqrt(BD.params["ϕini"]), # transformed Hubble parameter!
            "ns":  BD.params["ns"],
            "σ8":  BD.params["σ8"], # same σ8, but different As!
        })

    # calculates nonlinear power spectrum by boosting that from GR
    def power_spectrum_nonlinear(self):
        BD = self # rename just for symmetry with GR variable
        GR = self.transformed_GR_universe()

        # get PGR, PBDlin, PGRlin at the same values of k/(h/Mpc)
        k, PGR    = GR.power_spectrum_nonlinear()
        _, PBDlin = BD.power_spectrum_linear(k)
        _, PGRlin = GR.power_spectrum_linear(k)

        PBD = PBDlin / PGRlin * PGR # we showed PBD/PGR ≈ PBDlin/PGRlin under parameter transformation
        return k, PBD

if __name__ == "__main__":
    params = vars(args)
    params["z"] = sorted(params["z"])

    if args.spectrum == "PBD":
        k, P = BDUniverse(params).power_spectrum_nonlinear()
    elif args.spectrum == "PGR":
        k, P = GRUniverse(params).power_spectrum_nonlinear()
    elif args.spectrum == "PBDlin":
        k, P = BDUniverse(params).power_spectrum_linear()
    elif args.spectrum == "PGRlin":
        k, P = GRUniverse(params).power_spectrum_linear()
    else:
        raise Exception(f"Unrecognized power spectrum {args.spectrum}")

    output = sys.stdout.buffer if args.output == "-" else args.output
    header = f"{'k/(h/Mpc)':<21}" + " ".join(f"P(k,z={z:.3e})/Mpc^3" for z in params["z"])
    np.savetxt(output, np.transpose(np.vstack((k, P))), fmt="%.16e", header=header)
