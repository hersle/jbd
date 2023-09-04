import utils

import os
import re
import shutil
import subprocess
import numpy as np
from scipy.interpolate import interp1d, CubicSpline

COLAEXEC = os.path.abspath(os.path.expanduser("./FML/FML/COLASolver/nbody"))
CLASSEXEC = os.path.abspath(os.path.expanduser("./hi_class_public/class"))
SIMDIR = "./sims/"

def params2seeds(params, n=None):
    rng = np.random.default_rng(int(utils.hashdict(params), 16)) # deterministic random number generator from simulation parameters
    seeds = rng.integers(0, 2**31-1, size=n, dtype=int) # output python (not numpy) ints to make compatible with JSON dict hashing
    return int(seeds) if n is None else [int(seed) for seed in seeds]

class Simulation:
    def __init__(self, iparams=None, simdir="./sims/", path=None, verbose=True, run=True):
        if path is not None:
            self.directory = simdir + path + "/"
            iparams = utils.json2dict(self.read_file("parameters.json"))
            constructor = BDSimulation if "lgω" in iparams else GRSimulation
            constructor(iparams=iparams, verbose=verbose, run=run)
            return None

        if "seed" not in iparams:
            iparams["seed"] = params2seeds(iparams) # assign a (random) random seed

        self.iparams = iparams
        self.name = utils.hashdict(self.iparams) # parameters -> hash: identify sim with unique hash of its (independent, including seed) parameter dict
        self.directory = simdir + self.name + "/" # working directory for the simulation to live in
        os.makedirs(self.directory, exist_ok=True)
        self.write_file("parameters.json", utils.dict2json(self.iparams, unicode=True), skip_if_exists=True) # hash -> parameters: store (independent, including seed) parameters to enable reverse lookup

        self.dparams = self.derived_parameters()
        self.params = utils.dictupdate(self.iparams, self.dparams) # all (independent + dependent) parameters
        if verbose:
            print(f"{self.directory}: {self.iparams} -> {self.dparams}") # print independent -> dependent parameters

    # compute and return derived parameters
    def derived_parameters(self):
        dparams = {}

        # derive ωm0, ωb0 or ωc0 from the two others
        if "ωm0" not in self.iparams:
            dparams["ωm0"] = self.iparams["ωb0"] + self.iparams["ωc0"]
        elif "ωb0" not in self.iparams:
            dparams["ωb0"] = self.iparams["ωm0"] - self.iparams["ωc0"]
        elif "ωc0" not in self.iparams:
            dparams["ωc0"] = self.iparams["ωm0"] - self.iparams["ωb0"]
        else:
            raise(Exception("Exactly 2 of (ωb0, ωc0, ωm0) were not specified"))

        # derive σ8 or As from the other
        if "σ8" in self.iparams:
            if not self.file_exists("Ase9_from_s8.txt"):
                σ8_target = self.iparams["σ8"]
                Ase9 = 1.0 # initial guess
                while True:
                    self.params = utils.dictupdate(self.iparams, dparams | {"Ase9": Ase9}, remove=["σ8"])
                    self.run_class() # re-run until we have the correct σ8
                    σ8 = self.read_variable("class.log", "sigma8=")
                    if np.abs(σ8 - σ8_target) < 1e-10:
                        break
                    Ase9 = (σ8_target/σ8)**2 * Ase9 # exploit σ8^2 ∝ As to iterate efficiently (usually requires only one retry)
                self.write_file("Ase9_from_s8.txt", str(self.params["Ase9"])) # cache the result (expensive to compute)
            dparams["Ase9"] = float(self.read_file("Ase9_from_s8.txt"))
        elif "Ase9" in self.iparams:
            self.params = utils.dictupdate(self.iparams, dparams)
            self.run_class()
            dparams["σ8"] = self.read_variable("class.log", "sigma8=")
        else:
            raise(Exception("Exactly one of (Ase9, σ8) were not specified"))

        return dparams

    # whether a file in the simulation directory exists
    def file_exists(self, filename):
        return os.path.isfile(self.directory + filename)

    # check that the output from COLA is consistent with that from CLASS
    def validate_cola(self):
        print("Checking consistency between quantities computed separately by CLASS and COLA/FML:")

        # Read background quantities
        z_class, H_class = self.read_data("class_background.dat", dict=True, cols=("z", "H [1/Mpc]"))
        a_cola, E_cola = self.read_data(f"cosmology_{self.name}.txt", dict=True, cols=("a", "H/H0"))
        a_class  = 1 / (1 + z_class)

        # Compare E = H/H0
        E_class = H_class / H_class[-1] # E = H/H0 (assuming final value is at a=1)
        utils.check_values_are_close(E_class, E_cola, a_class, a_cola, name="(H/H0)", rtol=1e-4)

        # Compare ΩΛ0
        ΩΛ0_class = self.read_variable("class.log", "Lambda = ")
        ΩΛ0_cola  = self.read_variable("cola.log", "OmegaLambda       : ")
        utils.check_values_are_close(ΩΛ0_class, ΩΛ0_cola, name="ΩΛ0", rtol=1e-4)

        # Compare σ8 today
        σ8_class = self.read_variable("class.log", "sigma8=")
        σ8_cola  = self.read_variable("cola.log",  "Sigma\(R = 8 Mpc/h, z = 0.0 \) :\s+")
        utils.check_values_are_close(σ8_class, σ8_cola, name="σ8", rtol=1e-3)

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
    def write_file(self, filename, string, skip_if_exists=False):
        if skip_if_exists and self.file_exists(filename):
            return
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

        assert runcmd.returncode == 0, f"command {runcmd.args} crashed"

    # list of output redshifts constructed equal to those that FML outputs
    def output_redshifts(self):
        return 1 / np.linspace(1/(self.params["zinit"]+1), 1, self.params["Nstep"]+1) - 1

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
            "output": "mPk", # output P(k,z)
            "z_pk": ", ".join(str(z) for z in self.output_redshifts()), # output P(k,z) at same redshifts as FML/COLA, so interpolation behaves consistently
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
    def run_class(self, input_filename="class_input.ini", log="class.log"):
        input = "\n".join(f"{param} = {str(val)}" for param, val in self.params_class().items())
        input_is_unchanged = self.file_exists(input_filename) and self.read_file(input_filename) == input
        output_exists = self.file_exists(f"class_z{self.params['Nstep']+1}_pk.dat")
        complete = input_is_unchanged and output_exists

        if not complete:
            # write input and run class
            self.write_file(input_filename, input)
            self.run_command([CLASSEXEC, input_filename], log=log, verbose=True)

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
    def run_cola(self, np=1, verbose=True, ic_filename="power_spectrum_today.dat", input_filename="cola_input.lua", log="cola.log"):
        input = "\n".join(f"{param} = {utils.luastr(val)}" for param, val in self.params_cola().items())
        input_is_unchanged = self.file_exists(input_filename) and self.read_file(input_filename) == input
        output_exists = self.file_exists(f"pofk_{self.name}_cb_z0.000.txt")
        complete = input_is_unchanged and output_exists

        if not complete:
            k_h, Ph3 = self.power_spectrum(z=0, linear=True, hunits=True) # COLA wants CLASS' linear power spectrum (in h units)
            self.write_data(ic_filename, {"k/(h/Mpc)": k_h, "P/(Mpc/h)^3": Ph3}) # COLA wants "h-units"
            self.write_file(input_filename, input)
            cmd = ["mpirun", "-np", str(np), COLAEXEC, input_filename] if np > 1 else [COLAEXEC, input_filename] # TODO: ssh out to list of machines
            self.run_command(cmd, log=log, verbose=True)
            self.validate_cola()

    def power_spectrum(self, z=0.0, linear=False, hunits=False):
        zs = self.output_redshifts()
        if linear:
            self.run_class()
            filenames = [f"class_z{n+1}_pk.dat" for n in range(0, len(zs))]

            # verify that assumed redshifts are those reported by the files
            zs_from_files = [self.read_variable(filename, "redshift z=") for filename in filenames]
            assert np.all(np.round(zs_from_files, 5) == np.round(zs, 5))
        else:
            self.run_cola(np=16)
            filenames = [f"pofk_{self.name}_cb_z{z:.3f}.txt" for z in zs]

        # CubicSpline() wants increasing z, so sort everything now
        z_filename_pairs = zip(zs, filenames)
        z_filename_pairs = sorted(z_filename_pairs, key=lambda z_filename: z_filename[0])
        zs = [z_filename[0] for z_filename in z_filename_pairs]
        filenames = [z_filename[1] for z_filename in z_filename_pairs]

        k_h = self.read_data(filenames[0], cols=(0,))[0]
        assert np.all(self.read_data(filename, cols=(0,))[0] == k_h for filename in filenames), "P(k,z) files have different k"
        Ph3s = [self.read_data(filename, cols=(1,))[0] for filename in filenames] # indexed like Ph3s[iz][ik]
        Ph3_spline = CubicSpline(zs, Ph3s, axis=0) # spline Ph3 along z (axis 0) for each k (axis 1) # TODO: interpolate in a or z or loga or ...?
        Ph3 = Ph3_spline(z)

        if hunits:
            return k_h, Ph3
        else:
            k = k_h * self.params["h"]    # k / (1/Mpc)
            P = Ph3 / self.params["h"]**3 # P / Mpc^3
            return k, P

class GRSimulation(Simulation):
    def params_cola(self):
        return utils.dictupdate(Simulation.params_cola(self), {
            "cosmology_model": "LCDM",
            "gravity_model": "GR",
        })

class BDSimulation(Simulation):
    def derived_parameters(self):
        dparams = Simulation.derived_parameters(self) # call parent class
        dparams["ϕini"] = self.read_variable("class.log", "phi_ini = ")
        dparams["ϕ0"]   = self.read_variable("class.log", "phi_0 = ")
        dparams["ΩΛ0"]  = self.read_variable("class.log", "Lambda = ") / dparams["ϕ0"] # ρΛ0 / (3*H0^2*ϕ0/8*π)
        dparams["ωΛ0"]  = dparams["ΩΛ0"] * self.iparams["h"]**2 * dparams["ϕ0"]            # ∝ ρΛ0
        return dparams

    def params_class(self):
        ω = 10 ** self.params["lgω"]
        return utils.dictupdate(Simulation.params_class(self), {
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
        return utils.dictupdate(Simulation.params_cola(self), {
            "gravity_model": "JBD",
            "cosmology_model": "JBD",
            "cosmology_JBD_wBD": 10 ** self.params["lgω"],
            "cosmology_JBD_GeffG_today": self.params["G0/G"],
        })

    def validate_cola(self):
        Simulation.validate_cola(self) # do any validation in parent class

        # Read background tables and their scale factors (which we use as the free time variable)
        z_class, H_class, ϕ_class, dϕ_dη_class = self.read_data("class_background.dat", dict=True, cols=("z", "H [1/Mpc]", "phi_smg", "phi'"))
        a_cola, ϕ_cola, dlogϕ_dloga_cola = self.read_data(f"cosmology_{self.name}.txt", dict=True, cols=("a", "phi", "dlogphi/dloga"))
        a_class  = 1 / (1 + z_class)

        # Compare ϕ
        utils.check_values_are_close(ϕ_class, ϕ_cola, a_class, a_cola, name="ϕ", rtol=1e-5)

        # Compare dlogϕ/dloga
        dlogϕ_dloga_class = dϕ_dη_class / ϕ_class / (H_class * a_class) # convert by chain rule
        utils.check_values_are_close(dlogϕ_dloga_class, dlogϕ_dloga_cola, a_class, a_cola, name="dlogϕ/dloga", atol=1e-4)

class SimulationGroup:
    def __init__(self, simtype, iparams, nsims, seeds=None):
        if seeds is None:
            seeds = params2seeds(iparams, nsims)

        self.iparams = iparams
        self.sims = [simtype(iparams | {"seed": seed}) for seed in seeds] # run simulations with all seeds

        self.dparams = self.sims[0].dparams
        for sim in self.sims:
            assert sim.dparams == self.dparams, "simualtions have different derived parameters"

        self.params = utils.dictupdate(self.iparams, self.dparams) # merge

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

    def power_spectrum(self, linear=False, **kwargs):
        k, Ps = self.power_spectra(linear=linear, **kwargs)
        P  = np.mean(Ps, axis=0) # average            over simulations (for each k)
        ΔP = np.std( Ps, axis=0) # standard deviation over simulations (for each k)
        assert not linear or np.all(np.isclose(ΔP, 0.0, rtol=0, atol=1e-10)), "group simulations output different linear power spectra" # linear power spectrum is independent of seeds
        return k, P, ΔP

class SimulationGroupPair:
    def __init__(self, iparams_BD, iparams_BD_to_GR, nsims=1):
        seeds = params2seeds(iparams_BD, nsims) # BD parameters is a superset, so use them to make common seeds for BD and GR

        self.iparams_BD = iparams_BD
        self.sims_BD = SimulationGroup(BDSimulation, iparams_BD, nsims, seeds=seeds)

        self.iparams_GR = iparams_BD_to_GR(iparams_BD, self.sims_BD.params) # θGR = θGR(θBD)
        self.sims_GR = SimulationGroup(GRSimulation, self.iparams_GR, nsims, seeds=seeds)

        self.nsims = nsims

    def power_spectrum_ratio(self, z=0.0, linear=False): # TODO: plot B(k/h) instead?
        kBD, PBDs = self.sims_BD.power_spectra(z=z, linear=linear, hunits=False) # kBD / (1/Mpc), PBD / Mpc^3
        kGR, PGRs = self.sims_GR.power_spectra(z=z, linear=linear, hunits=False) # kGR / (1/Mpc), PGR / Mpc^3
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
        ΔB = np.array([utils.propagate_error([dB_dPBD[ik], dB_dPGR[ik]], [PBDs[ik], PGRs[ik]]) for ik in range(0, len(k))])

        # uncomment to compare matrix error propagation to manual expression (for one k value, to check it is correct)
        # (see formula for f=A/B at https://en.wikipedia.org/wiki/Propagation_of_uncertainty#Example_formulae)
        #σsq = np.cov([PBDs[0], PGRs[0]])
        #ΔB_matrix = ΔB[0]
        #ΔB_manual = B[0] * np.sqrt(σsq[0,0]/PBD[0]**2 + σsq[1,1]/PGR[0]**2 - 2*σsq[0,1]/(PBD[0]*PGR[0]))
        #assert np.isclose(ΔB_matrix, ΔB_manual), "error propagation is wrong"

        return k, B, ΔB
