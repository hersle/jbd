#!/bin/sh

# background
#./main.py --transform-h --divide primordial --params σ8 ωm0 ωb0 --evolution

# test different parametrizations
#./main.py --power class cola --h-units --subtract-shotnoise --divide primordial --transform-h --params σ8 ωm0 ωb0 z=0,1.5,3.0 --B-lims 0.9 1.3
#./main.py --power class cola --h-units --subtract-shotnoise --divide primordial --transform-h --params As ωm0 ωb0 z=0,1.5,3.0 --B-lims 0.9 1.3
#./main.py --power class cola --h-units --subtract-shotnoise --divide primordial               --params σ8 ωm0 ωb0 z=0,1.5,3.0 --B-lims 0.9 1.3
#./main.py --power class cola --h-units --subtract-shotnoise --divide primordial               --params As ωm0 ωb0 z=0,1.5,3.0 --B-lims 0.9 1.3

# linear evolution
#./main.py --power class scaleindependent --h-units --transform-h --subtract-shotnoise --divide primordial --B-lims 1.0 1.2 --params σ8 ωm0 ωb0 z=0,9,99,999,9899 --power-stem plots/linevo

# vary computational parameters
#./main.py --power class cola --h-units --transform-h --subtract-shotnoise --divide primordial --B-lims 1.1 1.2 --params σ8 ωm0 ωb0 Npart=256,512,1024
#./main.py --power class cola --h-units --transform-h --subtract-shotnoise --divide primordial --B-lims 1.1 1.2 --params σ8 ωm0 ωb0 Ncell=256,512,1024
#./main.py --power class cola --h-units --transform-h --subtract-shotnoise --divide primordial --B-lims 1.1 1.2 --params σ8 ωm0 ωb0 Nstep=30,300,2000
#./main.py --power class cola --h-units --transform-h --subtract-shotnoise --divide primordial --B-lims 1.1 1.2 --params σ8 ωm0 ωb0 zinit=10,30,50
#./main.py --power class cola --h-units --transform-h --subtract-shotnoise --divide primordial --B-lims 1.1 1.2 --params σ8 ωm0 ωb0 Lh=128,384,640

# vary physical parameters
#./main.py --power class cola --h-units --transform-h --subtract-shotnoise --divide primordial --B-lims 1.0 1.2 --params σ8 ωm0 ωb0 lgω=2.0,2.5,3.0,4.0
#./main.py --power class cola --h-units --transform-h --subtract-shotnoise --divide primordial --B-lims 1.0 1.2 --params σ8 ωm0 ωb0 G0=0.95,1.0,1.05
#./main.py --power class cola --h-units --transform-h --subtract-shotnoise --divide primordial --B-lims 1.0 1.2 --params σ8 ωb0 ωm0=0.10,0.15,0.20
#./main.py --power class cola --h-units --transform-h --subtract-shotnoise --divide primordial --B-lims 1.0 1.2 --params σ8 ωm0 ωb0=0.01,0.02,0.03
#./main.py --power class cola --h-units --transform-h --subtract-shotnoise --divide primordial --B-lims 1.0 1.2 --params ωm0 ωb0 σ8=0.75,0.80,0.85
#./main.py --power class cola --h-units --transform-h --subtract-shotnoise --divide primordial --B-lims 1.0 1.2 --params σ8 ωm0 ωb0 ns=0.92,0.96,1.0
#./main.py --power class cola --h-units --transform-h --subtract-shotnoise --divide primordial --B-lims 1.0 1.2 --params σ8 ωm0 ωb0 h=0.65,0.70,0.73

# paramter space sampling of ω and ωm0
#./main.py --params lgω=2.0,4.0 σ8 ωb0 z=0 ωm0=0.1,0.2 --samples 100 --parameter-space --power class cola --divide class      --h-units --subtract-shotnoise --transform-h --B-lims 0.9 1.1 --power-stem plots/sample_z0_divclass
#./main.py --params lgω=2.0,4.0 σ8 ωb0 z=0 ωm0=0.1,0.2 --samples 100 --parameter-space --power class cola --divide primordial --h-units --subtract-shotnoise --transform-h --B-lims 1.0 1.2 --power-stem plots/sample_z0_divprim
#./main.py --params lgω=2.0,4.0 σ8 ωb0 z=3 ωm0=0.1,0.2 --samples 100 --parameter-space --power class cola --divide class      --h-units --subtract-shotnoise --transform-h --B-lims 0.9 1.1 --power-stem plots/sample_z3_divclass
#./main.py --params lgω=2.0,4.0 σ8 ωb0 z=3 ωm0=0.1,0.2 --samples 100 --parameter-space --power class cola --divide primordial --h-units --subtract-shotnoise --transform-h --B-lims 1.0 1.2 --power-stem plots/sample_z3_divprim

# EuclidEmulator2
./main.py --power class ee2 --h-units --transform-h --params σ8=0.81 ωm0 ωb0 z=0,1.5,3,9 --subtract-shotnoise --divide primordial --B-lims 0.8 1.2 --power-stem plots/ee2
