#!/bin/sh

# background
#./main.py --transform-h --divide primordial --params σ8 ωm0 ωb0 --evolution

# test different parametrizations
#./main.py --power class cola --h-units --subtract-shotnoise --transform-h --params σ8 ωm0 ωb0 z=0,1.5,3.0 --B-lims 0.9 1.3
#./main.py --power class cola --h-units --subtract-shotnoise --transform-h --params As ωm0 ωb0 z=0,1.5,3.0 --B-lims 0.9 1.3
#./main.py --power class cola --h-units --subtract-shotnoise --params σ8 ωm0 ωb0 z=0,1.5,3.0 --B-lims 0.9 1.3
#./main.py --power class cola --h-units --subtract-shotnoise --params As ωm0 ωb0 z=0,1.5,3.0 --B-lims 0.9 1.3

# linear evolution
#./main.py --power class scaleindependent --h-units --transform-h --subtract-shotnoise --divide primordial --B-lims 1.0 1.2 --params σ8 ωm0 ωb0 z=0,1.5,9,99,999,9899 --power-stem plots/linevo --figsize 5.0 2.2

# ramses comparison
#./main.py --power class cola ramses --h-units --transform-h --subtract-shotnoise --B-lims 0.9 1.1 --figsize 3.0 1.5 --params σ8 ωm0 ωb0 z=3,1.5,0 --power-stem plots/ramses

# vary computational parameters
#./main.py --power class cola --h-units --transform-h --subtract-shotnoise --B-lims 0.9 1.1 --figsize 3.0 1.5 --params σ8 ωm0 ωb0 Npart=256,512,1024 --power-stem plots/Npart
#./main.py --power class cola --h-units --transform-h --subtract-shotnoise --B-lims 0.9 1.1 --figsize 3.0 1.5 --params σ8 ωm0 ωb0 Ncell=256,512,1024 --power-stem plots/Ncell
#./main.py --power class cola --h-units --transform-h --subtract-shotnoise --B-lims 0.9 1.1 --figsize 3.0 1.5 --params σ8 ωm0 ωb0 Nstep=30,300,2000 --power-stem plots/Nstep
#./main.py --power class cola --h-units --transform-h --subtract-shotnoise --B-lims 0.9 1.1 --figsize 3.0 1.5 --params σ8 ωm0 ωb0 zinit=10,30,50 --power-stem plots/zinit
#./main.py --power class cola --h-units --transform-h --subtract-shotnoise --B-lims 0.9 1.1 --figsize 3.0 1.5 --params σ8 ωm0 ωb0 Lh=128,384,640 --power-stem plots/Lh

# vary physical parameters
#./main.py --power class cola --h-units --transform-h --subtract-shotnoise --B-lims 0.9 1.1 --figsize 3.0 1.5 --params σ8 ωm0 ωb0 lgω=1.0,2.0,3.0 --power-stem plots/ω
#./main.py --power class cola --h-units --transform-h --subtract-shotnoise --B-lims 0.9 1.1 --figsize 3.0 1.5 --params σ8 ωm0 ωb0 G0=0.95,1.0,1.05 --power-stem plots/G
#./main.py --power class cola --h-units --transform-h --subtract-shotnoise --B-lims 0.9 1.1 --figsize 3.0 1.5 --params σ8 ωb0 ωm0=0.10,0.15,0.20 --power-stem plots/ωm0
#./main.py --power class cola --h-units --transform-h --subtract-shotnoise --B-lims 0.9 1.1 --figsize 3.0 1.5 --params σ8 ωm0 ωb0=0.01,0.02,0.03 --power-stem plots/ωb0
#./main.py --power class cola --h-units --transform-h --subtract-shotnoise --B-lims 0.9 1.1 --figsize 3.0 1.5 --params ωm0 ωb0 σ8=0.75,0.80,0.85 --power-stem plots/σ8
#./main.py --power class cola --h-units --transform-h --subtract-shotnoise --B-lims 0.9 1.1 --figsize 3.0 1.5 --params σ8 ωm0 ωb0 ns=0.92,0.96,1.0 --power-stem plots/ns
#./main.py --power class cola --h-units --transform-h --subtract-shotnoise --B-lims 0.9 1.1 --figsize 3.0 1.5 --params σ8 ωm0 ωb0 h=0.65,0.70,0.73 --power-stem plots/h
#./main.py --power class cola --h-units --transform-h --subtract-shotnoise --divide class --B-lims 0.99 1.01 --figsize 3.0 1.5 --one-by-one --params z=0.0,1.5,3.0 h=0.70,0.65,0.73 ns=1.0,0.96,0.92 G0=1.0,0.95,1.05 σ8=0.80,0.75,0.85 ωm0=0.15,0.10,0.20 ωb0=0.02,0.01,0.03 lgω=2.0,3.0 --power-stem plots/all

# parameter space sampling of ω and ωm0
#./main.py --params lgω=2.0,4.0 σ8 ωb0 z=0 ωm0=0.1,0.2 --samples 100 --parameter-space --power class cola --divide class      --h-units --subtract-shotnoise --transform-h --B-lims 0.99 1.01 --power-stem plots/sample_z0_divclass --figsize 3.0 1.2
#./main.py --params lgω=2.0,4.0 σ8 ωb0 z=0 ωm0=0.1,0.2 --samples 100 --parameter-space --power class cola --divide primordial --h-units --subtract-shotnoise --transform-h --B-lims 1.0 1.2 --power-stem plots/sample_z0_divprim
#./main.py --params lgω=2.0,4.0 σ8 ωb0 z=3 ωm0=0.1,0.2 --samples 100 --parameter-space --power class cola --divide class      --h-units --subtract-shotnoise --transform-h --B-lims 0.99 1.01 --power-stem plots/sample_z3_divclass --figsize 3.0 1.2
#./main.py --params lgω=2.0,4.0 σ8 ωb0 z=3 ωm0=0.1,0.2 --samples 100 --parameter-space --power class cola --divide primordial --h-units --subtract-shotnoise --transform-h --B-lims 1.0 1.2 --power-stem plots/sample_z3_divprim

# EuclidEmulator2
#./main.py --power class ee2 --h-units --transform-h --params σ8=0.81 ωm0 ωb0 z=0,1.5,3,9 --subtract-shotnoise --divide primordial --B-lims 0.8 1.2 --power-stem plots/ee2

# test final script
#./main.py --power class ee2 script --h-units --transform-h --params σ8=0.81 ωm0 ωb0 z=0,1.5,3 --subtract-shotnoise --divide primordial --B-lims 0.8 1.2 --power-stem plots/script
