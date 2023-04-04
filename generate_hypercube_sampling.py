#!/usr/bin/env python3

import numpy as np
import json
import copy
from smt.sampling_methods import LHS
import matplotlib.pyplot as plt

#
# Very simple example of making hypercube samples
# Define fiducial parameters
# Give ranges for hypercube samples (if total_samples=1 and lowerlimit==upperlimit for parameters then you just do the fiducial cosmology)
# Give outputfolder and path to cola executable (this latter is just to be able to generate BASH files for running the sims after)
#

# Total number of samples to generate
total_samples = 8
prefix = "Geff"
dictfile = "latin_hypercube_samples_"+prefix+"_"+str(total_samples)+".json"
make_hostfiles = True
test_run = False # Don't make dict, but plot samples instead...

# Choose parameters to vary and the prior
parameters_to_vary = {
  'A_s': [1e-9, 4e-9],
  'Omega_cdm': [0.15, 0.35],
  'mu0': [-1.0, 1.0],
}
parameters_to_vary_arr = []
for key in parameters_to_vary:
  parameters_to_vary_arr.append(key)

# Set the fiducial cosmoloy
run_param_fiducial = {
  'label':        "FiducialCosmology",
  'outputfolder': "./output",
  'colaexe':      "/mn/stornext/u3/hansw/Dennis/PipelineTest/FML/FML/COLASolver/nbody",

  # COLA parameters
  'boxsize':    350.0,
  'Npart':      640,
  'Nmesh':      640,
  'Ntimesteps': 30,
  'Seed':       1234567,
  'zini':       30.0,
  'input_spectra_from_lcdm': "true",
  
  # Cosmological parameters
  'cosmo_param': {
    # With physical parameters we specify Omega_i h^2 and h is a derived quantity
    # Otherwise we specify Omega_i and h and Omega_Lambda is a derived quantity
    'use_physical_parameters': False,
    'cosmology_model': 'w0waCDM',
    'gravity_model': 'Geff',
    'h':          0.67,
    'Omega_b':    0.05,
    'Omega_cdm':  0.267,
    'Omega_ncdm': 0.0012,
    'Omega_k':    0.0,
    'omega_b':    0.05    * 0.67**2,
    'omega_cdm':  0.267   * 0.67**2,
    'omega_ncdm': 0.0012  * 0.67**2,
    'omega_k':    0.0     * 0.67**2,
    'omega_fld':  0.0     * 0.67**2, # Only needed for JBD
    'w0':         -1.0, 
    'wa':         0.0,
    'Neff':       3.046,
    'k_pivot':    0.05,
    'A_s':        2.1e-9,
    'n_s':        0.965,
    'T_cmb':      2.7255,
    'mu0':        0.0,     # Only for Geff
    'log10wBD':   3,       # Only for JBD
    'log10fofr0': -5.0,    # Only for f(R)
    'kmax_hmpc':  20.0,
  },
 
  # Parameters used for generating a bash-file for running the sim
  'ncpu':     32, 
  'nthreads': 4,
  'npernode': 32,
  'hostfile': "/mn/stornext/u3/hansw/TestRunsPedro/hostfile.txt",
}

# Generate nodelist
if make_hostfiles:
  # List of nodes we have availiable
  hostfiles = ["euclid25.uio.no", "euclid26.uio.no", "euclid27.uio.no", "euclid28.uio.no", "euclid29.uio.no", "euclid30.uio.no", "euclid31.uio.no", "euclid32.uio.no"]

  count = 0
  for node in hostfiles:
    hostfile = run_param_fiducial['outputfolder'] + "/hostfile" + str(count) + ".txt"
    with open(hostfile,"w") as f:
      f.write(node)
    hostfiles[count] = hostfile
    count += 1

#========================================================================
#========================================================================

# Generate all samples
ranges = []
for key in parameters_to_vary:
  ranges.append(parameters_to_vary[key])
ranges = np.array(ranges)
sampling = LHS(xlimits=ranges)
all_samples = sampling(total_samples)

# Generate the dictionaries
sims = {}
count = 0
for sample in all_samples:
  if test_run:
    print("===========================")
    print("New parameter sample:")
  run_param = copy.deepcopy(run_param_fiducial)
  for i, param in enumerate(parameters_to_vary):
    run_param["cosmo_param"][param] = sample[i]
    if test_run:
      print("Setting ", param, " to value ", sample[i])
  label = prefix+str(count)
  run_param['label'] = label
  
  # For assigning different hostfiles to different nodes
  if make_hostfiles:
    node_number = count % len(hostfiles)
    run_param['hostfile'] = hostfiles[node_number]
    run_param['node_number'] = node_number

  sims[str(count)] = copy.deepcopy(run_param)
  count += 1

if test_run:
  nparam = len(parameters_to_vary_arr)
  for i in range(nparam):
    for j in range(i+1,nparam):
      plt.plot(all_samples[:,i], all_samples[:,j], "o")
      plt.xlabel(parameters_to_vary_arr[i])
      plt.ylabel(parameters_to_vary_arr[j])
      plt.show()
  exit(1)

# Save to file
with open(dictfile, "w") as f:
  data = json.dumps(sims)
  f.write(data)

