#!/usr/bin/env python3

import os
import shutil
import subprocess
import numpy as np

from scipy.stats import qmc
from classy import Class

COLAEXEC = os.path.expanduser("~/FML/FML/COLASolver/nbody")

def luastr(var):
    if isinstance(var, str):
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
        self.name = "test"
        self.directory = "sims/sim_" + self.name + "/"

        # jump into simulation directory
        os.makedirs(self.directory, exist_ok=True)
        cwd = os.getcwd()
        os.chdir(self.directory)

        if not self.completed():
            print(f"Simulating \"{self.name}\"")
            self.run_class(params)
            self.run_cola(params)

        # jump out of simulation directory
        os.chdir(cwd)

    def completed(self):
        return os.path.isfile(f"pofk_{self.name}_cb_z0.000.txt")

    def write_data(self, filename, cols, colnames=None):
        if isinstance(cols, dict):
            colnames = cols.keys()
            cols = [cols[colname] for colname in colnames]
            return self.write_data(filename, cols, colnames)

        header = None if colnames is None else " ".join(colnames)
        np.savetxt(filename, np.transpose(cols), header=header)

    def write_file(self, filename, string):
        with open(filename, "w") as file:
            file.write(string)

    def run_class(self, params):
        params_class = {
            "output": "mTk", # needed for Class to compute transfer function
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

        z1, z2 = 10.0, 0.0
        a1, a2 = 1/(z1+1), 1/(z2+1)
        as_ = 10 ** np.linspace(np.log10(a1), np.log10(a2), 100)
        zs  = 1/as_ - 1
        zs  = np.flip(zs) # TODO: from 0 ???
        nsnaps = len(zs)

        # for each redshift snapshot, write transfer function
        transfer_filenames = []
        for i, z in enumerate(zs):
            transfer = self.cosmology.get_transfer(z=z, output_format="camb")
            while len(transfer) < 13: # FML parses exactly 13 columns
                transfer[f"dummy{len(transfer)}"] = np.zeros_like(list(transfer.values())[0])
            transfer_filename = f"transfer_z{z:.3f}.txt"
            self.write_data(transfer_filename, transfer)
            transfer_filenames.append(transfer_filename)

        # write accumulating transfer file
        transferlist  = f". {nsnaps}\n"
        transferlist += "\n".join(f"{transfer_filenames[i]} {zs[i]}" for i in range(0, nsnaps))
        self.write_file("transfer.txt", transferlist)
        
    def run_cola(self, params, z0=10):
        # write cola transferinfo file
        params_cola = {
            "simulation_name": self.name,
            "simulation_boxsize": 350.0,
            "simulation_use_cola": True,

            "cosmology_model": "LCDM",
            "cosmology_h": params["h"],
            "cosmology_Omegab": params["Ωb0"],
            "cosmology_OmegaCDM": params["Ωc0"],
            "cosmology_OmegaLambda": params["ΩΛ0"],
            "cosmology_Neffective": params["Neff"],
            "cosmology_TCMB_kelvin": params["Tγ0"],

            "particle_Npart_1D": 64, # TODO: increase

            "timestep_nsteps": [30],

            "ic_random_seed": 1234,
            "ic_initial_redshift": z0,
            "ic_nmesh" : 64,
            "ic_type_of_input": "transferinfofile",
            "ic_input_filename": "transfer.txt",
            "ic_input_redshift": 0.0, # TODO: ???

            "force_nmesh": 64,

            "output_folder": ".",
            "output_redshifts": [0.0],
        }

        colainfile = "cola_input.lua"
        self.write_file(colainfile, "\n".join(f"{param} = {luastr(val)}\n" for param, val in params_cola.items()))

        colalogfile = "cola.log"
        with open(colalogfile, "w") as logf:
            proc = subprocess.run([COLAEXEC, colainfile], stdout=logf, stderr=logf)
        assert proc.returncode == 0, f"ERROR: see {colalogfile} for details"

#if __name__ == "__main__":
params0 = {
    "h":          0.67,
    "Ωb0":        0.05,
    "Ωc0":        0.267,
    "Ωk0":        0.0,
    "Tγ0":        2.7255,
    #"Omega_ncdm":      0.0012, # TODO: non-cold dark matter???
    "Neff":       3.046,
    "kpivot":    0.05,
    "As":        2.1e-9,
    "ns":        0.965,
    #"w0":         -1.0, 
    #"wa":         0.0,
    #"mu0":        0.0,     # Only for Geff
    #"log10wBD":   3,       # Only for JBD
    #"log10fofr0": -5.0,    # Only for f(R)
    #"kmax_hmpc":  20.0,
    #"use_physical_parameters": False, " True: specify Ωs0*h^2, h is derived; False: specify Ωs0 and h, ΩΛ0 is derived
    #"cosmology_model": "w0waCDM",
    #"gravity_model": "Geff",
    #"omega_b":    0.05    * 0.67**2,
    #"omega_cdm":  0.267   * 0.67**2,
    #"omega_ncdm": 0.0012  * 0.67**2,
    #"omega_k":    0.0     * 0.67**2,
    #"omega_fld":  0.0     * 0.67**2, # Only needed for JBD
}
params0["ΩΛ0"] = 1 - params0["Ωb0"] - params0["Ωc0"] - params0["Ωk0"]
params_varying = {
    "As": (1e-9, 4e-9),
    "Ωc0": (0.15, 0.35),
    "μ0": (-1.0, +1.0),
}

paramspace = ParameterSpace(params_varying)
params = params0
sim = Simulation(params)
