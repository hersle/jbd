import utils

import os
import re
import shutil
import subprocess
import numpy as np
from scipy.interpolate import CubicSpline

def params2seeds(params, n=None):
    rng = np.random.default_rng(int(utils.hashdict(params), 16)) # deterministic random number generator from simulation parameters
    seeds = rng.integers(0, 2**31-1, size=n, dtype=int) # output python (not numpy) ints to make compatible with JSON dict hashing
    return int(seeds) if n is None else [int(seed) for seed in seeds]

class Simulation: # TODO: makes more sense to name Model, Cosmology or something similar
    SIMDIR = os.path.expandvars(os.path.expandvars("$DATA/jbdsims/"))
    COLAEXEC = os.path.abspath(os.path.expandvars("$HOME/local/FML/FML/COLASolver/nbody"))
    CLASSEXEC = os.path.abspath(os.path.expandvars("$HOME/local/hi_class_public/class"))
    RAMSESEXEC = os.path.abspath(os.path.expandvars("$HOME/local/Ramses/bin/ramses3d"))
    RAMSES2PKEXEC = os.path.abspath(os.path.expandvars("$HOME/local/FML/FML/RamsesUtils/ramses2pk/ramses2pk"))

    def __init__(self, iparams):
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
            cls(iparams)

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
            if not self.file_exists("class/Ase9_from_s8.txt"):
                σ8_target = self.iparams["σ8"]
                Ase9 = 1.0 # initial guess
                while True:
                    self.params = utils.dictupdate(self.iparams, dparams | {"Ase9": Ase9}, remove=["σ8"])
                    self.run_class() # re-run until we have the correct σ8
                    σ8 = self.read_variable("class/log.txt", "sigma8=")
                    if np.abs(σ8 - σ8_target) < 1e-10:
                        break
                    Ase9 = (σ8_target/σ8)**2 * Ase9 # exploit σ8^2 ∝ As to iterate efficiently (usually requires only one retry)
                self.write_file("class/Ase9_from_s8.txt", str(self.params["Ase9"])) # cache the result (expensive to compute)
            dparams["Ase9"] = float(self.read_file("class/Ase9_from_s8.txt"))
        elif "Ase9" in self.iparams:
            self.params = utils.dictupdate(self.iparams, dparams)
            self.run_class()
            dparams["σ8"] = self.read_variable("class/log.txt", "sigma8=")
        else:
            raise(Exception("Exactly one of (Ase9, σ8) were not specified"))

        # derive L or L*h from the other
        if "Lh" in self.iparams:
            dparams["L"] = self.iparams["Lh"] / self.iparams["h"]
        elif "L" in self.iparams:
            dparams["Lh"] = self.iparams["L"] * self.iparams["h"]
        else:
            raise(Exception("Exactly one of (L, Lh) were not specified"))

        return dparams

    def create_directory(self, dirname):
        os.makedirs(self.directory + dirname, exist_ok=True)

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
        subprocess.run(cmd, shell=True, cwd=self.directory+subdir, check=True, stdin=subprocess.DEVNULL) # https://stackoverflow.com/a/45988305

    # list of output redshifts constructed equal to those that FML outputs
    def output_redshifts(self):
        return 1 / np.linspace(1/(self.params["zinit"]+1), 1, self.params["Nstep"]+1) - 1

    # dictionary of parameters that should be passed to CLASS
    def input_class(self):
        return '\n'.join([
            # cosmological parameters
            f"h = {self.params['h']}",
            f"Omega_b = {self.params['ωb0'] / self.params['h']**2}",
            f"Omega_cdm = {self.params['ωc0'] / self.params['h']**2}",
            f"Omega_k = {self.params['ωk0'] / self.params['h']**2}",
            f"T_cmb = {self.params['Tγ0']}",
            f"N_eff = {self.params['Neff']}",
            f"A_s = {self.params['Ase9'] / 1e9}",
            f"n_s = {self.params['ns']}",
            f"k_pivot = {self.params['kpivot']}",
            f"YHe = 0.25",

            # output control
            f"output = mPk", # output matter power spectrum P(k,z)
            f"non linear = halofit", # also estimate non-linear P(k,z) from halo modelling
            f"z_pk = {', '.join(str(z) for z in self.output_redshifts())}", # output P(k,z) at same redshifts as FML/COLA, so interpolation behaves consistently
            f"write background = yes",
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
        output_exists = self.file_exists(f"class/z{self.params['Nstep']+1}_pk.dat")
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
            f'cosmology_As = {self.params["Ase9"] / 1e9}',
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
            k_h, Ph3 = self.power_spectrum(z=0, source="linear-class", hunits=True) # COLA wants CLASS' linear power spectrum (in h units)
            self.write_data("cola/pofk_ic.dat", {"k/(h/Mpc)": k_h, "P/(Mpc/h)^3": Ph3}) # COLA wants "h-units"
            self.write_file("cola/input.lua", input)
            self.run_command(f"{self.COLAEXEC} input.lua", subdir="cola/", np=np, log="log.txt") # TODO: ssh out to list of machines?
            self.validate_cola()

    def input_ramses(self, nproc=1):
        Npart1D = self.params["Npart"]
        Nparts = Npart1D**3
        levelmin = int(np.round(np.log2(Npart1D))) # e.g. 10 if Npart1D = 1024
        levelmax = levelmin + 10                   # e.g. 20 if Npart1D = 1024
        assert 2**levelmin == Npart1D, "particle count is not a power of 2"

        return '\n'.join([
            f"&RUN_PARAMS",
            f"cosmo=.true.", # enable cosmological run
            f"pic=.true.", # enable particle-in-cell solver
            f"poisson=.true.", # enable Poisson solver
            f"nrestart=0", # start simulation from scratch # TODO: could read most recent existing output_xxxxx directory (and ncpu)
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
            f"levelmin={levelmin}", # min number of refinement levels (if Npart1D = 2^n, it should be n)
            f"levelmax={levelmax}", # max number of refinement levels (if very high, it will automatically stop refining) # TODO: revise?
            f"npartmax={Nparts+1}", # need +1 to avoid crash with 1 process (maybe ramses allocates one dummy particle or something?) # TODO: optimal value? maybe double what would be needed if particles were shared equally across CPUs?
            f"ngridmax={Nparts+1}", # TODO: optimal value?
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
        input = self.input_ramses()
        input_is_unchanged = self.file_exists("ramses/input.nml") and self.read_file("ramses/input.nml") == input
        output_exists = self.file_exists("ramses/log.txt") and self.read_file("ramses/log.txt").find("Run completed") != -1
        complete = input_is_unchanged and output_exists

        if not complete:
            self.run_cola(np=np) # use COLA's particles as initial conditions
            self.write_file("ramses/input.nml", input)
            self.run_command(f"{self.RAMSESEXEC} input.nml", subdir="ramses/", np=np, log="log.txt")

    def power_spectrum(self, z=0.0, source="linear-class", hunits=True):
        zs = self.output_redshifts()
        if source == "linear-class":
            self.run_class()
            filenames = [f"class/z{n+1}_pk.dat" for n in range(0, len(zs))]
        elif source == "nonlinear-class":
            self.run_class()
            filenames = [f"class/z{n+1}_pk_nl.dat" for n in range(0, len(zs))]
        elif source == "nonlinear-cola":
            self.run_cola(np=16)
            filenames = [f"cola/pofk_cola_cb_z{z:.3f}.txt" for z in zs]
        elif source == "nonlinear-ramses":
            self.run_ramses(np=16)
            zs, filenames = [], [] # ramses outputs its own redshifts
            snapnum = 1
            while True:
                snapdir = f"ramses/output_{snapnum:05d}"
                info_filename = f"{snapdir}/info_{snapnum:05d}.txt"
                pofk_filename = f"{snapdir}/pofk_fml.dat"
                if self.file_exists(info_filename):
                    # compute P(k) if not already done
                    if not self.file_exists(pofk_filename):
                        self.run_command(f"{self.RAMSES2PKEXEC} --verbose --density-assignment=CIC {snapdir}", np=16)
                        assert self.file_exists(pofk_filename)
                    a = self.read_variable(info_filename, "aexp        =  ") # == 1 / (z+1)
                    zs.append(1/a - 1) # be careful to not override z!
                    filenames.append(pofk_filename)
                else:
                    break # no more snapshots
                snapnum += 1
        else:
            raise Exception(f"unknown power spectrum source \"{source}\"")

        if source in ("linear-class", "nonlinear-class"):
            # verify that assumed redshifts are those reported by the files
            zs_from_files = [self.read_variable(filename, "redshift z=") for filename in filenames]
            assert np.all(np.round(zs_from_files, 5) == np.round(zs, 5))

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
        ω = 10 ** self.params["lgω"]
        return Simulation.input_class(self) + '\n'.join([
            f"gravity_model = brans_dicke", # select BD gravity
            f"Omega_Lambda = 0", # rather include Λ through potential term (first entry in parameters_smg; should be equivalent)
            f"Omega_fld = 0", # no dark energy fluid
            f"Omega_smg = -1", # automatic modified gravity
            f"parameters_smg = NaN, {ω}, 1.0, 0.0", # ΩΛ0 (fill with cosmological constant), ω, Φini (arbitrary initial guess), Φ′ini≈0 (fixed)
            f"M_pl_today_smg = {(4+2*ω)/(3+2*ω) / self.params['G0/G']}", # see https://github.com/HAWinther/hi_class_pub_devel/blob/3160be0e0482ac2284c20b8878d9a81efdf09f2a/gravity_smg/gravity_models_smg.c#L462
            f"a_min_stability_test_smg = 1e-6", # BD has early-time instability, so lower tolerance to pass stability checker
            f"output_background_smg = 2", # >= 2 needed to output phi to background table (https://github.com/miguelzuma/hi_class_public/blob/16ae0f6ccfcee513146ec36b690678f34fb687f4/source/background.c#L3031)
            f"" # final newline
        ])

    def input_cola(self):
        return Simulation.input_cola(self) + '\n'.join([
            f'gravity_model = "JBD"',
            f'cosmology_model = "JBD"',
            f'cosmology_JBD_wBD = {10 ** self.params["lgω"]}',
            f'cosmology_JBD_GeffG_today = {self.params["G0/G"]}',
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

    def input_ramses(self):
        lines = Simulation.input_ramses(self).split('\n')

        # generate file with columns log(a), Geff(a)/G0, E(a) = H(a)/H0
        bg = self.read_data("class/background.dat", dict=True)
        assert bg["z"][-1] == 0.0, "last row of class/background.dat is not today (z=0)"
        ω = 10**self.params["lgω"]
        loga = np.log(1 / (bg["z"] + 1)) # log = ln
        G_G0 = (4+2*ω) / (3+2*ω) / bg["phi_smg"] # TODO: divide by G or G0 (relevant if G0 != G)
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

    def power_spectrum(self, **kwargs):
        k, Ps = self.power_spectra(**kwargs)
        P  = np.mean(Ps, axis=0) # average            over simulations (for each k)
        ΔP = np.std( Ps, axis=0) # standard deviation over simulations (for each k)
        return k, P, ΔP

class SimulationGroupPair:
    def __init__(self, iparams_BD, iparams_BD_to_GR, nsims=1):
        seeds = params2seeds(iparams_BD, nsims) # BD parameters is a superset, so use them to make common seeds for BD and GR

        self.iparams_BD = iparams_BD
        self.sims_BD = SimulationGroup(BDSimulation, iparams_BD, nsims, seeds=seeds)

        self.iparams_GR = iparams_BD_to_GR(iparams_BD, self.sims_BD.params) # θGR = θGR(θBD)
        self.sims_GR = SimulationGroup(GRSimulation, self.iparams_GR, nsims, seeds=seeds)

        self.nsims = nsims

    def power_spectrum_ratio(self, z=0.0, source="linear-class", hunits=False):
        kBD, PBDs = self.sims_BD.power_spectra(z=z, source=source, hunits=hunits) # kBD / (hBD/Mpc), PBD / (Mpc/hBD)^3
        kGR, PGRs = self.sims_GR.power_spectra(z=z, source=source, hunits=hunits) # kGR / (hGR/Mpc), PGR / (Mpc/hGR)^3

        # Verify that COLA/RAMSES simulations output comparable k-values (k*L should be equal)
        kLBD = kBD * (self.sims_BD.params["Lh" if hunits else "L"])
        kLGR = kGR * (self.sims_GR.params["Lh" if hunits else "L"])
        assert source.endswith("class") or np.all(np.isclose(kLBD, kLGR)), "weird k-values"

        # get reference wavenumbers and interpolate P to those values
        k = (kBD + kGR) / 2 # simulations have kBD == kGR == k
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

        return k, B, ΔB
