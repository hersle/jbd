import utils

import os
import re
import glob
import shutil
import subprocess
import numpy as np
from scipy.interpolate import CubicSpline

def params2seeds(params, n=None):
    rng = np.random.default_rng(int(utils.hashdict(params), 16)) # deterministic random number generator from simulation parameters
    seeds = rng.integers(0, 2**31-1, size=n, dtype=int) # output python (not numpy) ints to make compatible with JSON dict hashing
    return int(seeds) if n is None else [int(seed) for seed in seeds]

# parameter transformations
def θGR_identity(θBD, θBD_all):
    return utils.dictupdate(θBD, remove=["ω", "G0"]) # remove BD-specific parameters
def θGR_different_h(θBD, θBD_all):
    θGR = θGR_identity(θBD, θBD_all)
    θGR["h"] = θBD["h"] * np.sqrt(θBD_all["ϕini"]) # ensure similar Hubble evolution (of E=H/H0) during radiation domination
    return θGR

class Simulation: # TODO: makes more sense to name Model, Cosmology or something similar
    SIMDIR = os.path.expandvars(os.path.expandvars("$DATA/jbdsims/"))
    COLAEXEC = os.path.abspath(os.path.expandvars("$HOME/local/FML/FML/COLASolver/nbody"))
    CLASSEXEC = os.path.abspath(os.path.expandvars("$HOME/local/hi_class_public/class"))
    RAMSESEXEC = os.path.abspath(os.path.expandvars("$HOME/local/Ramses/bin/ramses3d"))
    RAMSES2PKEXEC = os.path.abspath(os.path.expandvars("$HOME/local/FML/FML/RamsesUtils/ramses2pk/ramses2pk"))
    EE2EXEC = os.path.abspath(os.path.expandvars("$HOME/local/EuclidEmulator2-pywrapper/ee2.exe"))
    BDPYEXEC = os.path.abspath(os.path.expandvars("$HOME/jbd/bd.py"))

    def __init__(self, iparams):
        if "lgω" in iparams:
            iparams["ω"] = 10 ** iparams.pop("lgω")
        if "seed" not in iparams:
            iparams["seed"] = params2seeds(iparams) # assign a (random) random seed

        self.iparams = iparams
        self.name = utils.hashdict(self.iparams) # parameters -> hash: identify sim with unique hash of its (independent, including seed) parameter dict
        self.directory = self.SIMDIR + self.name + "/" # working directory for the simulation to live in
        self.create_directory("") # create directory for the simulation to live in
        self.write_file("parameters.json", utils.dict2json(self.iparams, unicode=True) + '\n', skip_if_exists=True) # hash -> parameters: store (independent, including seed) parameters to enable reverse lookup

        self.dparams = self.derived_parameters()
        self.params = utils.dictupdate(self.iparams, self.dparams) # all (independent + dependent) parameters

        iparamsstr = ", ".join(f"{param} = {value}" for param, value in self.iparams.items())
        dparamsstr = ", ".join(f"{param} = {value}" for param, value in self.dparams.items())
        print(f"Loaded {self.directory}: {iparamsstr} -> {dparamsstr}") # print independent -> dependent parameters

    @classmethod
    def list(cls):
        for path in os.scandir(cls.SIMDIR):
            iparams = utils.json2dict(utils.read_file(f"{cls.SIMDIR}{path.name}/parameters.json"))
            try:
                cls(iparams)
            except:
                print("ERROR")

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
        if not "As" in self.iparams:
            if not self.file_exists("class/As_from_s8.txt"):
                σ8_target = self.iparams["σ8"]
                As = 1.0e-9 # initial guess
                while True:
                    self.params = utils.dictupdate(self.iparams, dparams | {"As": As}, remove=["σ8"])
                    self.run_class() # re-run until we have the correct σ8
                    σ8 = self.read_variable("class/log.txt", "sigma8=")
                    if np.abs(σ8 - σ8_target) < 1e-10:
                        break
                    As = (σ8_target/σ8)**2 * As # exploit σ8^2 ∝ As to iterate efficiently (usually requires only one retry)
                self.write_file("class/As_from_s8.txt", str(self.params["As"])) # cache the result (expensive to compute)
            dparams["As"] = float(self.read_file("class/As_from_s8.txt"))
        elif not "σ8" in self.iparams:
            self.params = utils.dictupdate(self.iparams, dparams)
            self.run_class()
            dparams["σ8"] = self.read_variable("class/log.txt", "sigma8=")
        else:
            raise(Exception("Exactly one of (As, σ8) were not specified"))

        # derive L or L*h from the other
        if not "L" in self.iparams:
            dparams["L"] = self.iparams["Lh"] / self.iparams["h"]
        elif not "Lh" in self.iparams:
            dparams["Lh"] = self.iparams["L"] * self.iparams["h"]
        else:
            raise(Exception("Exactly one of (L, Lh) were not specified"))

        return dparams

    def create_directory(self, dirname):
        os.makedirs(self.directory + dirname, exist_ok=True)

    def glob(self, query):
        results = glob.glob(self.directory + query)
        results = [os.path.relpath(result, self.directory) for result in results]
        return results

    def file_path(self, filename):
        return self.directory + filename

    # whether a file in the simulation directory exists
    def file_exists(self, filename):
        return os.path.isfile(self.file_path(filename))

    # check that the output from COLA is consistent with that from CLASS
    def validate_cola(self):
        # Read background quantities
        z_class, H_class = self.read_data("class/background.dat", dict=True, cols=("z", "H [1/Mpc]"))
        a_cola, E_cola = self.read_data(f"cola/cosmology_cola.txt", dict=True, cols=("a", "H/H0"))
        a_class  = 1 / (1 + z_class)

        # Compare E = H/H0
        E_class = H_class / H_class[-1] # E = H/H0 (assuming final value is at a=1)
        utils.check_values_are_close(E_class, E_cola, a_class, a_cola, name="(H/H0)", rtol=1e-4)

        # Compare ΩΛ0
        ΩΛ0_class = self.read_variable("class/log.txt", "Lambda = ")
        ΩΛ0_cola  = self.read_variable("cola/log.txt", "OmegaLambda             : ")
        utils.check_values_are_close(ΩΛ0_class, ΩΛ0_cola, name="ΩΛ0", rtol=1e-4)

        # Compare σ8 today
        σ8_class = self.read_variable("class/log.txt", "sigma8=")
        σ8_cola  = self.read_variable("cola/log.txt",  "Sigma\(R = 8 Mpc/h, z = 0.0 \) :\s+")
        utils.check_values_are_close(σ8_class, σ8_cola, name="σ8", rtol=1e-3)

    # save a data file associated with the simulation
    def write_data(self, filename, cols, colnames=None):
        if isinstance(cols, dict):
            colnames = cols.keys()
            cols = [cols[colname] for colname in colnames]
            return self.write_data(filename, cols, colnames)

        header = None if colnames is None else " ".join(colnames)
        np.savetxt(self.file_path(filename), np.transpose(cols), header=header)

    # load a data file associated with the simulation
    def read_data(self, filename, dict=False, cols=None):
        data = np.loadtxt(self.file_path(filename))
        data = np.transpose(data) # index by [icol, irow] (or just [icol] to get a whole column)

        # if requested, read header and generate {header[icol]: data[icol]} dictionary
        if dict:
            with open(self.file_path(filename)) as file:
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
        with open(self.file_path(filename), "w", encoding="utf-8") as file:
            file.write(string)

    # load a file associated with the simulation
    def read_file(self, filename):
        return utils.read_file(self.file_path(filename))

    # read a string like "[prefix][number]" from a file and return number
    # example: if file contains "Omega_Lambda = 1.23", read_variable(filename, "Omega_Lambda = ") returns 1.23
    def read_variable(self, filename, prefix):
        regex = prefix + r"([-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?)"
        matches = re.findall(regex, self.read_file(filename))
        assert len(matches) == 1, f"found {len(matches)} ≠ 1 matches {matches} for regex \"{regex}\" in file {self.file_path(filename)}"
        return float(matches[0][0])

    # run a command in the simulation's directory
    def run_command(self, cmd, np=1, log=None, subdir=""):
        if np > 1:
            cmd = f"mpirun -np {np} {cmd}"
        if log:
            cmd = f"{cmd} | tee {log}"
        cwd = self.directory + subdir
        print(f"Running {cmd} with {np=} in {cwd=}")
        subprocess.run(cmd, shell=True, cwd=cwd, check=True, stdin=subprocess.DEVNULL) # https://stackoverflow.com/a/45988305

    # list of output redshifts constructed equal to those that FML outputs
    def output_redshifts(self):
        return 1 / np.linspace(1/(self.params["zinit"]+1), 1, self.params["Nstep"]+1) - 1

    # dictionary of parameters that should be passed to CLASS
    def input_class(self):
        zs1 = np.array([9899, 7999, 6999, 5999, 4999, 3999, 3606, 2999, 1999, 999, 899, 799, 699, 599, 499, 399, 299, 199, 99, 9]) # early-time zs (require exact hits when querying these)
        zs2 = 1 / np.linspace(1/(3.5+1), 1, 50) - 1 # 50 zs in [3.5, 0] with linear a-spacing (for splining)? # TODO: use class' z_max_pk?
        zs = np.concatenate((zs1, zs2))
        return '\n'.join([
            # cosmological parameters
            f"h = {self.params['h']}",
            f"Omega_b = {self.params['ωb0'] / self.params['h']**2}",
            f"Omega_cdm = {self.params['ωc0'] / self.params['h']**2}",
            f"Omega_k = {self.params['ωk0'] / self.params['h']**2}",
            f"T_cmb = {self.params['Tγ0']}",
            f"N_eff = {self.params['Neff']}",
            f"A_s = {self.params['As']}",
            f"n_s = {self.params['ns']}",
            f"k_pivot = {self.params['kpivot']}",
            f"YHe = 0.25",

            # output control
            f"output = mPk", # output matter power spectrum P(k,z)
            f"non linear = halofit", # also estimate non-linear P(k,z) from halo modelling
            f"z_pk = {', '.join(str(np.round(z, 5)) for z in zs)}",
            f"write background = yes",
            f"write primordial = yes",
            f"root = ./",
            f"P_k_max_h/Mpc = 100.0", # output linear power spectrum to fill my plots

            # log verbosity (increase integers to make more talkative)
            f"input_verbose = 10",
            f"background_verbose = 10",
            f"thermodynamics_verbose = 2",
            f"perturbations_verbose = 2",
            f"spectra_verbose = 2",
            f"output_verbose = 2",
            f"", # final newline
        ])

    # run CLASS and return today's matter power spectrum
    def run_class(self):
        self.create_directory("class/")
        input = self.input_class()
        input_is_unchanged = self.file_exists("class/input.ini") and self.read_file("class/input.ini") == input
        output_exists = self.file_exists(f"class/z1_pk.dat")
        complete = input_is_unchanged and output_exists

        if not complete:
            # write input and run class
            self.write_file("class/input.ini", input)
            self.run_command(f"{self.CLASSEXEC} input.ini", subdir="class/", log="log.txt")

    # dictionary of parameters that should be passed to COLA
    def input_cola(self):
        return '\n'.join([ # common parameters (for any derived simulation)
            f'simulation_name = "cola"',
            f'simulation_boxsize = {self.params["Lh"]}',
            f'simulation_use_cola = true',
            f'simulation_use_scaledependent_cola = false', # TODO: only relevant with massive neutrinos?

            f'cosmology_h = {self.params["h"]}',
            f'cosmology_Omegab = {self.params["ωb0"] / self.params["h"]**2}',
            f'cosmology_OmegaCDM = {self.params["ωc0"] / self.params["h"]**2}',
            f'cosmology_OmegaK = {self.params["ωk0"] / self.params["h"]**2}',
            f'cosmology_Neffective = {self.params["Neff"]}',
            f'cosmology_TCMB_kelvin = {self.params["Tγ0"]}',
            f'cosmology_As = {self.params["As"]}',
            f'cosmology_ns = {self.params["ns"]}',
            f'cosmology_kpivot_mpc = {self.params["kpivot"]}',
            f'cosmology_OmegaMNu = 0.0',

            f'particle_Npart_1D = {self.params["Npart"]}',

            f'timestep_nsteps = {{{self.params["Nstep"]}}}',

            f'ic_random_field_type = "gaussian"',
            f'ic_random_seed = {self.params["seed"]}',
            f'ic_fix_amplitude = true', # use P(k) when generating Gaussian random field # TODO: (?)
            f'ic_use_gravity_model_GR = false', # don't use GR for backscaling P(k) in MG runs; instead be consistent with gravity model
            f'ic_initial_redshift = {self.params["zinit"]}',
            f'ic_nmesh = {self.params["Ncell"]}',
            f'ic_type_of_input = "powerspectrum"', # transferinfofile only relevant with massive neutrinos?
            f'ic_input_filename = "pofk_ic.dat"',
            f'ic_input_redshift = 0.0', # TODO: feed initial power spectrum directly instead of backscaling? Hans said someone incorporated this into his code?

            f'force_nmesh = {self.params["Ncell"]}',

            f'output_folder = "."',
            f'output_redshifts = {{{self.params["zinit"]}, 0.0}}', # list(self.output_redshifts()), # dump initial and final particles
            f'output_particles = true',

            f'pofk = false', # rather use underway computation (ignoring all pofk_... parameters)
            f'pofk_nmesh = {self.params["Ncell"]}',
            f'pofk_interlacing = true',
            f'pofk_subtract_shotnoise = true',
            f'pofk_density_assignment_method = "CIC"',
            f'', # final newline
        ])

    # run COLA simulation from back-scaling today's matter power spectrum (from CLASS)
    def run_cola(self, np=1, verbose=True):
        self.create_directory("cola")
        input = self.input_cola()
        input_is_unchanged = self.file_exists("cola/input.lua") and self.read_file("cola/input.lua") == input
        output_exists = self.file_exists(f"cola/pofk_cola_cb_z0.000.txt")
        complete = input_is_unchanged and output_exists

        if not complete:
            k_h, Ph3 = self.power_spectrum(z=0, source="class", hunits=True) # COLA wants CLASS' linear power spectrum (in h units)
            self.write_data("cola/pofk_ic.dat", {"k/(h/Mpc)": k_h, "P/(Mpc/h)^3": Ph3}) # COLA wants "h-units"
            self.write_file("cola/input.lua", input)
            self.run_command(f"{self.COLAEXEC} input.lua", subdir="cola/", np=np, log="log.txt") # TODO: ssh out to list of machines?
            self.validate_cola()

    def input_ramses(self, nproc=1):
        Ncell1D = self.params["Ncell"]
        Npart1D = self.params["Npart"]
        Nparts = Npart1D**3
        Ncells = Ncell1D**3
        levelmin = int(np.round(np.log2(Ncell1D))) # e.g. 10 if Ncell1D = 1024
        levelmax = levelmin + 10                   # e.g. 20 if Ncell1D = 1024
        assert 2**levelmin == Ncell1D, "cell count is not a power of 2"

        snaps = re.findall(r"Main step=\s*(\d+)", self.read_file("ramses/log.txt")) if self.file_exists("ramses/log.txt") else []
        lastsnap = snaps[-1] if len(snaps) >= 1 else "0"

        return '\n'.join([
            f"&RUN_PARAMS",
            f"cosmo=.true.", # enable cosmological run
            f"pic=.true.", # enable particle-in-cell solver
            f"poisson=.true.", # enable Poisson solver
            f"nrestart={lastsnap}", # start simulation from last available snapshot (or from scratch) # TODO: make sure number of CPUs is the same?
            f"nremap=8", # coarse time steps between each load balancing of AMR grid (important for parallelization)
            f"verbose=.false.", # verbosity
            #f"nsubcycle=" # TODO: control fine time stepping?
            f"/",
            f"&OUTPUT_PARAMS",
            f"foutput=1", # one output per coarse time step
            f"/",
            f"&INIT_PARAMS",
            f"filetype='gadget'",
            f"initfile(1)='../cola/snapshot_cola_z{self.params['zinit']:.3f}/gadget_z{self.params['zinit']:.3f}'", # start from COLA's initial particles
            f"/",
            f"&AMR_PARAMS",
            f"levelmin={levelmin}", # min number of refinement levels (if Ncell1D = 2^n, it should be n)
            f"levelmax={levelmax}", # max number of refinement levels (if very high, it will automatically stop refining) # TODO: revise?
            f"npartmax={3*Nparts//2//nproc+1}", # need +1 to avoid crash with 1 process (maybe ramses allocates one dummy particle or something?) # TODO: optimal value? maybe double what would be needed if particles were shared equally across CPUs?
            f"ngridmax={4*Ncells//2//nproc+1}", # TODO: optimal value?
            f"nexpand=1", # number of mesh expansions # TODO: ???
            #f"boxlen={self.params['Lh']}", # WARNING: don't set this; something is fucked with RAMSES' units when boxlen != 1.0
            f"/",
            f"&REFINE_PARAMS",
            f"m_refine={','.join([str(8)] * (levelmax-levelmin))}", # refine cells with >= 8 particles (on average into 8 cells with 1 paticle each)
            f"/",
            f"", # final newline
        ])

    def run_ramses(self, np=1):
        self.create_directory("ramses/")
        output_exists = self.file_exists("ramses/log.txt") and self.read_file("ramses/log.txt").find("Run completed") != -1
        complete = output_exists

        if not complete:
            input = self.input_ramses(nproc=np)
            self.run_cola(np=np) # use COLA's particles as initial conditions
            self.write_file("ramses/input.nml", input)
            self.run_command(f"{self.RAMSESEXEC} input.nml", subdir="ramses/", np=np, log="log.txt")

    def power_spectrum(self, z=0.0, source="class", hunits=True, subshot=False):
        if source == "class":
            self.run_class()
            zs, filenames, n = [], [], 1
            while True:
                filename = f"class/z{n}_pk.dat"
                if self.file_exists(filename):
                    zf = self.read_variable(filename, "redshift z=")
                    if (z <= 3.5 and zf <= 3.5) or (z > 3.5 and zf == z): # require exact hit for high z
                        zs.append(zf)
                        filenames.append(filename)
                    if zf == 0.0:
                        break # last file (later ones must be from old runs)
                else:
                    break
                n += 1
        elif source == "halofit":
            self.run_class()
            filenames = [f"class/z{n+1}_pk_nl.dat" for n in range(0, len(zs))]
        elif source == "cola":
            self.run_cola(np=32)
            filenames = self.glob("cola/pofk_cola_cb_z*.txt")
            zs = [float(filename[filename.find("z")+1:filename.rfind(".")]) for filename in filenames]
        elif source == "ramses":
            self.run_ramses(np=32)
            zs, filenames = [], [] # ramses outputs its own redshifts
            snapnum = 1
            while True:
                level = int(np.log2(self.params["Npart"])) # coarsest AMR grid level
                level += 2 # enhance P(k) computation by 2 levels (e.g. N = 256 -> 256*2^2 = 1024)
                snapdir = f"ramses/output_{snapnum:05d}"
                info_filename = f"{snapdir}/info_{snapnum:05d}.txt"
                pofk_filename = f"{snapdir}/pofk_fml_level{level}.dat"
                if self.file_exists(info_filename):
                    # compute P(k) if not already done
                    if not self.file_exists(pofk_filename):
                        self.run_command(f"{self.RAMSES2PKEXEC} --verbose --level={level} --subtract-shotnoise --density-assignment=CIC {snapdir}", np=32)
                        self.run_command(f"mv \"{snapdir}/pofk_fml.dat\" \"{pofk_filename}\"") # e.g. pofk_fml.dat -> pofk_fml_level10.dat
                        assert self.file_exists(pofk_filename)
                    a = self.read_variable(info_filename, "aexp        =  ") # == 1 / (z+1)
                    zs.append(1/a - 1) # be careful to not override z!
                    filenames.append(pofk_filename)
                else:
                    break # no more snapshots
                snapnum += 1
        elif source == "primordial":
            zs = [np.inf]
            filenames = ["class/primordial_Pk.dat"]
        elif source == "scaleindependent":
            data = self.read_data("cola/gravitymodel_cola_k1.0.txt", dict=True)
            a, D = data["a"], data["D1(a,k)"]
            D_D0 = CubicSpline(np.log(a), D/D[0])(np.log(1 / (z+1)))
            k, P = self.power_spectrum(z, source="primordial", hunits=hunits, subshot=subshot)
            return k, P * D_D0**2
        else:
            raise Exception(f"unknown power spectrum source \"{source}\"")

        # CubicSpline() wants increasing z, so sort everything now
        z_filename_pairs = zip(zs, filenames)
        z_filename_pairs = sorted(z_filename_pairs, key=lambda z_filename: z_filename[0])
        zs = [z_filename[0] for z_filename in z_filename_pairs]
        filenames = [z_filename[1] for z_filename in z_filename_pairs]

        k_h = self.read_data(filenames[0], cols=(0,))[0]
        assert np.all(self.read_data(filename, cols=(0,))[0] == k_h for filename in filenames), "P(k,z) files have different k"
        Ph3s = [self.read_data(filename, cols=(1,))[0] for filename in filenames] # indexed like Ph3s[iz][ik]

        if len(z_filename_pairs) == 1: # require exact hit for high z
            Ph3 = Ph3s[0]
        else:
            Ph3 = CubicSpline(zs, Ph3s, axis=0, extrapolate=False)(z) # spline Ph3 along z (axis 0) for each k (axis 1) # TODO: interpolate in a or z or loga or ...?

        # analytically computed primordial power spectrum (can also do calculation myself)
        if source == "primordial":
            k_h = 10 ** np.linspace(-5, +2, 100)
            Δ = (k_h * self.params["h"] / self.params["kpivot"]) ** (self.params["ns"]-1) # dimensionless primordial power spectrum
            Ph3 = 2 * np.pi**2 / k_h**3 * self.params["As"] * Δ # dimensionful primordial power spectrum

        if not subshot and source in ("cola", "ramses"):
            Ph3 += (self.params["Lh"]/self.params["Npart"])**3 # add shot noise back in

        if hunits:
            return k_h, Ph3
        else:
            k = k_h * self.params["h"]    # k / (1/Mpc)
            P = Ph3 / self.params["h"]**3 # P / Mpc^3
            return k, P

class GRSimulation(Simulation):
    SIMDIR = Simulation.SIMDIR + "GR/"
    RAMSESEXEC = os.path.abspath(os.path.expandvars("$HOME/local/Ramses/bin/ramses3dGR"))

    def __init__(self, iparams):
        Simulation.__init__(self, iparams)

    def input_cola(self):
        return Simulation.input_cola(self) + '\n'.join([
            'cosmology_model = "LCDM"',
            'gravity_model = GR',
            '' # final newline
        ])

    def run_ee2(self, z=0.0, outfile_stem="BNL"):
        h = self.params["h"]
        Ωb0 = self.params["ωb0"] / h**2
        Ωm0 = self.params["ωm0"] / h**2
        ns = self.params["ns"]
        As = self.params["As"]

        self.create_directory("ee2")
        self.run_command(f"{self.EE2EXEC} -b {Ωb0} -m {Ωm0} -n {ns} -H {h} -A {As} -z {z} -W -1.0 -w 0.0 -s 0.0 -d . -o {outfile_stem}", subdir="ee2/", log="log.txt")
        outfile = f"ee2/{outfile_stem}0.dat"
        assert self.file_exists(outfile)
        return outfile

    def power_spectrum(self, **kwargs):
        # TODO: prevent running of duplicate sims with different seeds !!!
        k, P = None, None
        if kwargs["source"] == "ee2":
            # 1) get B = P / Plin from EE2
            outfile = self.run_ee2(z=kwargs["z"])
            k_h, B = self.read_data(outfile) # k/(h/Mpc), P(k)/Plin(k)
            k = k_h * self.params["h"] # k/(1/Mpc)

            # 2) get Plin from class
            klin, Plin = self.power_spectrum(**kwargs | {"source": "class", "hunits": False}) # k/(1/Mpc), P/Mpc^3

            # 3a) get P by multiplying Plin * B = P with splined B
            P = CubicSpline(k, B, axis=1, extrapolate=False)(klin) * Plin # careful with units!
            k, P = klin[np.isfinite(P)], P[np.isfinite(P)] # remove NaNs because EE2 has a smaller k-range than class

            # 3b) get P by multiplying Plin * B = P with splined Plin
            #P = B * CubicSpline(klin, Plin, extrapolate=False)(k)

        elif kwargs["source"] == "script":
            self.create_directory("script")
            self.run_command(f"{self.BDPYEXEC} -z {kwargs['z']} -w 0 -G 1 -H {self.params['h']} -m {self.params['ωm0']} -b {self.params['ωb0']} -n {self.params['ns']} -A {self.params['As']} --hiclass {self.CLASSEXEC} --ee2 {self.EE2EXEC} PGR > P.dat", subdir="script/")
            k, P = self.read_data("script/P.dat")
            k = k * self.params["h"]

        if k is not None and P is not None:
            if kwargs["hunits"]:
                return k / self.params["h"], P * self.params["h"]**3
            else:
                return k, P

        return Simulation.power_spectrum(self, **kwargs)

class BDSimulation(Simulation):
    SIMDIR = Simulation.SIMDIR + "BD/"
    RAMSESEXEC = os.path.abspath(os.path.expandvars("$HOME/local/Ramses/bin/ramses3dBD"))

    def __init__(self, iparams):
        Simulation.__init__(self, iparams)

    def derived_parameters(self):
        dparams = Simulation.derived_parameters(self) # call parent class
        dparams["ϕini"] = self.read_variable("class/log.txt", "phi_ini = ")
        dparams["ϕ0"]   = self.read_variable("class/log.txt", "phi_0 = ")
        dparams["ΩΛ0"]  = self.read_variable("class/log.txt", "Lambda = ") / dparams["ϕ0"] # ρΛ0 / (3*H0^2*ϕ0/8*π)
        dparams["ωΛ0"]  = dparams["ΩΛ0"] * self.iparams["h"]**2 * dparams["ϕ0"]            # ∝ ρΛ0
        return dparams

    def input_class(self):
        return Simulation.input_class(self) + '\n'.join([
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

    def input_cola(self):
        return Simulation.input_cola(self) + '\n'.join([
            f'gravity_model = "JBD"',
            f'cosmology_model = "JBD"',
            f'cosmology_JBD_wBD = {self.params["ω"]}',
            f'cosmology_JBD_GeffG_today = {self.params["G0"]}',
            f'cosmology_JBD_density_parameter_definition = "hi-class"',
            f''# final newline
        ])

    def validate_cola(self):
        Simulation.validate_cola(self) # do any validation in parent class

        # Read background tables and their scale factors (which we use as the free time variable)
        z_class, H_class, ϕ_class, dϕ_dη_class = self.read_data("class/background.dat", dict=True, cols=("z", "H [1/Mpc]", "phi_smg", "phi'"))
        a_cola, ϕ_cola, dlogϕ_dloga_cola = self.read_data(f"cola/cosmology_cola.txt", dict=True, cols=("a", "phi", "dlogphi/dloga"))
        a_cola, ϕ_cola, dlogϕ_dloga_cola = a_cola[a_cola<=1], ϕ_cola[a_cola<=1], dlogϕ_dloga_cola[a_cola<=1]
        a_class  = 1 / (1 + z_class)

        # Compare ϕ
        utils.check_values_are_close(ϕ_class, ϕ_cola, a_class, a_cola, name="ϕ", rtol=1e-5)

        # Compare dlogϕ/dloga
        dlogϕ_dloga_class = dϕ_dη_class / ϕ_class / (H_class * a_class) # convert by chain rule
        utils.check_values_are_close(dlogϕ_dloga_class, dlogϕ_dloga_cola, a_class, a_cola, name="dlogϕ/dloga", atol=1e-4)

    def input_ramses(self, **kwargs):
        lines = Simulation.input_ramses(self, **kwargs).split('\n')

        # generate file with columns log(a), Geff(a)/G0, E(a) = H(a)/H0
        bg = self.read_data("class/background.dat", dict=True)
        z = bg["z"]
        assert z[-1] == 0.0, "last row of class/background.dat is not today (z=0)"
        inds = z <= self.params["zinit"] # only need background with z <= zinit
        inds[np.argmax(inds)-1] = True # include z before zinit, too
        for col in bg:
            bg[col] = bg[col][inds]
        loga = np.log(1 / (bg["z"] + 1)) # log = ln
        G_G0 = (4+2*self.params["ω"]) / (3+2*self.params["ω"]) / bg["phi_smg"] # TODO: divide by G or G0 (relevant if G0 != G)
        H_H0 = bg["H [1/Mpc]"] / bg["H [1/Mpc]"][-1]
        self.write_data("ramses/BD.dat", [loga, G_G0, H_H0], colnames=["ln(a)", "G(a)/G0", "H(a)/H0"])
        qtylines = self.read_file("ramses/BD.dat").split('\n')
        qtylines.insert(1, str(len(loga))) # first line should have number of rows
        self.write_file("ramses/BD.dat", '\n'.join(qtylines))

        # add extra line in &AMR_PARAMS section with path to BD quantities
        line = f"filename_geff_hubble_data='BD.dat'"
        lines.insert(lines.index("&AMR_PARAMS")+1, line)
        lines.append("") # final newline
        return '\n'.join(lines)

    def power_spectrum(self, **kwargs):
        k, P = None, None

        if kwargs["source"] == "ee2":
            # 1) get B = PBD/PGR ≈ PBDlin/PGRlin from BD+GR simulation pair using class
            sims = SimulationGroupPair(self.iparams, θGR_different_h)
            k_h, B, _ = sims.power_spectrum_ratio(**kwargs | {"source": "class", "hunits": True}) # k/h, [PBD(k/h,z)*hBD^3] / [PGR(k/h,z)*hGR^3]

            # 2) get PGR from EE2
            kGR_h, PGRh3 = sims.sims_GR[0].power_spectrum(**kwargs | {"source": "ee2", "hunits": True}) # kGR/(h/Mpc), PGR/(Mpc/h)^3
            kBD_h = k_h

            # 3a) get PBD by multiplying PGR*B = PBD with splined B
            PBDh3 = PGRh3 * CubicSpline(k_h, B, extrapolate=False)(kGR_h)
            k_h = kGR_h

            P = PBDh3 / self.params["h"]**3
            k = k_h * self.params["h"]

            # 3b) get PBD by multiplying PGR*B = PBD with splined PGR
            #PBDh3 = CubicSpline(kGR_h, PGRh3, extrapolate=False)(k_h) * B
            #k_h, PBDh3 = k_h[np.isfinite(PBDh3)], PBDh3[np.isfinite(PBDh3)] # remove NaNs due to different k-ranges

        elif kwargs["source"] == "script":
            self.create_directory("script")
            self.run_command(f"{self.BDPYEXEC} -z {kwargs['z']} -w {self.params['ω']} -G {self.params['G0']} -H {self.params['h']} -m {self.params['ωm0']} -b {self.params['ωb0']} -n {self.params['ns']} -A {self.params['As']} --hiclass {self.CLASSEXEC} --ee2 {self.EE2EXEC} PBD > P.dat", subdir="script/")
            k, P = self.read_data("script/P.dat")
            k = k * self.params["h"]

        if k is not None and P is not None:
            if kwargs["hunits"]:
                return k / self.params["h"], P * self.params["h"]**3
            else:
                return k, P

        return Simulation.power_spectrum(self, **kwargs)

class SimulationGroup:
    def __init__(self, simtype, iparams, nsims, seeds=None):
        if "lgω" in iparams:
            iparams["ω"] = 10 ** iparams.pop("lgω")
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
            assert len(ks) == 0 or np.all(np.isclose(k, ks[0])), "group simulations output different k"
            ks.append(k)
            Ps.append(P)

        k = ks[0] # common wavenumbers for all simulations (by assertion)
        Ps = np.array(Ps) # 2D numpy array P[isim, ik]
        return k, Ps

    def power_spectrum(self, **kwargs):
        k, Ps = self.power_spectra(**kwargs)
        P  = np.mean(Ps, axis=0) # average            over simulations (for each k)
        ΔP = np.std( Ps, axis=0) # standard deviation over simulations (for each k)
        return k, P, ΔP

class SimulationGroupPair:
    def __init__(self, iparams_BD, iparams_BD_to_GR, nsims=1):
        if "lgω" in iparams_BD:
            iparams_BD["ω"] = 10 ** iparams_BD.pop("lgω")

        seeds = params2seeds(iparams_BD, nsims) # BD parameters is a superset, so use them to make common seeds for BD and GR

        self.iparams_BD = iparams_BD
        self.sims_BD = SimulationGroup(BDSimulation, iparams_BD, nsims, seeds=seeds)

        self.iparams_GR = iparams_BD_to_GR(iparams_BD, self.sims_BD.params) # θGR = θGR(θBD)
        self.sims_GR = SimulationGroup(GRSimulation, self.iparams_GR, nsims, seeds=seeds)

        self.nsims = nsims

    def power_spectrum_ratio(self, z=0.0, source="class", hunits=False, divide="", subshot=False):
        kBD, PBDs = self.sims_BD.power_spectra(z=z, source=source, hunits=hunits, subshot=subshot) # kBD / (hBD/Mpc), PBD / (Mpc/hBD)^3
        kGR, PGRs = self.sims_GR.power_spectra(z=z, source=source, hunits=hunits, subshot=subshot) # kGR / (hGR/Mpc), PGR / (Mpc/hGR)^3

        # Verify that COLA/RAMSES simulations output comparable k-values (k*L should be equal)
        kLBD = kBD * (self.sims_BD.params["Lh" if hunits else "L"])
        kLGR = kGR * (self.sims_GR.params["Lh" if hunits else "L"])
        assert source in ("class", "primordial", "ee2", "script") or np.all(np.isclose(kLBD, kLGR)), "weird k-values"

        # get reference wavenumbers and interpolate P to those values
        kmin = np.maximum(np.min(kBD), np.min(kGR))
        kmax = np.minimum(np.max(kBD), np.max(kGR))
        k = np.unique(np.concatenate((kBD[np.logical_and(kBD >= kmin, kBD <= kmax)], kGR[np.logical_and(kGR >= kmin, kGR <= kmax)]))) # all k-values in range spanned by both BD and GR
        PGRs = CubicSpline(kGR, PGRs, axis=1, extrapolate=False)(k) # interpolate PGR(k/hGR) to PGR(k/h)
        PBDs = CubicSpline(kBD, PBDs, axis=1, extrapolate=False)(k) # interpolate PBD(k/hBD) to PBD(k/h)

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
        dB_dPBD =      1 / PGR    # dB/dPBD evaluated at means
        dB_dPGR = -PBD / PGR**2 # dB/dPGR evaluated at means
        ΔB = np.array([utils.propagate_error([dB_dPBD[ik], dB_dPGR[ik]], [PBDs[ik], PGRs[ik]]) for ik in range(0, len(k))])

        # uncomment to compare matrix error propagation to manual expression (for one k value, to check it is correct)
        # (see formula for f=A/B at https://en.wikipedia.org/wiki/Propagation_of_uncertainty#Example_formulae)
        #σsq = np.cov([PBDs[0], PGRs[0]])
        #ΔB_matrix = ΔB[0]
        #ΔB_manual = B[0] * np.sqrt(σsq[0,0]/PBD[0]**2 + σsq[1,1]/PGR[0]**2 - 2*σsq[0,1]/(PBD[0]*PGR[0]))
        #assert np.isclose(ΔB_matrix, ΔB_manual), "error propagation is wrong"

        if divide: # TODO: handle ΔB?
            kdiv, Bdiv, _ = self.power_spectrum_ratio(source=divide, z=z, hunits=hunits, divide="", subshot=subshot) # don't divide
            Bdiv = CubicSpline(kdiv, Bdiv, axis=1, extrapolate=False)(k) # interpolate to main source's k
            return k, B/Bdiv, ΔB/Bdiv
        else:
            return k, B, ΔB

SIMTYPES = [BDSimulation, GRSimulation]
