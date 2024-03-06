#!/bin/sh

# background
#./main.py --transform-h --divide primordial --params σ8 ωm0 ωb0 --evolution

# test different parametrizations
#./main.py --power class cola --h-units --subtract-shotnoise --transform-h --params σ8 ωm0 ωb0 z=0,1.5,3.0 --B-lims 0.9 1.3 --legend-location "upper right" "upper left" --Blabel "B / (h_\mathrm{BD}/h_\mathrm{GR})^{-3}" --power-stem plots/s8_h2phi
#./main.py --power class cola --h-units --subtract-shotnoise --transform-h --params As ωm0 ωb0 z=0,1.5,3.0 --B-lims 0.9 1.3 --legend-location "upper right" "upper left" --Blabel "B / (h_\mathrm{BD}/h_\mathrm{GR})^{-3}" --power-stem plots/As_h2phi
#./main.py --power class cola --h-units --subtract-shotnoise               --params σ8 ωm0 ωb0 z=0,1.5,3.0 --B-lims 0.9 1.3 --legend-location "upper right" "upper left" --Blabel "B / (h_\mathrm{BD}/h_\mathrm{GR})^{-3}" --power-stem plots/s8_h
#./main.py --power class cola --h-units --subtract-shotnoise               --params As ωm0 ωb0 z=0,1.5,3.0 --B-lims 0.9 1.3 --legend-location "upper right" "upper left" --Blabel "B / (h_\mathrm{BD}/h_\mathrm{GR})^{-3}" --power-stem plots/As_h

# linear evolution
#./main.py --power class scaleindependent --h-units --transform-h --subtract-shotnoise --divide primordial --B-lims 1.0 1.20 --params σ8 ωm0 ωb0 z=0,1.5,9,99,999 --power-stem plots/linevo --figsize 3.5 2.8 --legend-location "upper right" "upper left"

# ramses comparison
#./main.py --power class cola ramses --h-units --transform-h --subtract-shotnoise --B-lims 0.98 1.02 --figsize 3.5 2.2 --params σ8 ωm0 ωb0 z=3,1.5,0 --power-stem plots/ramses --legend-location "upper right" "lower left"
#./main.py --power cola --divide ramses --h-units --transform-h --subtract-shotnoise --B-lims 0.98 1.02 --figsize 3.5 2.2 --params σ8 ωm0 ωb0 z=3,1.5,0 --power-stem plots/ramses --legend-location "upper right" "" --Blabel "B_\\textsc{cola} / B_\\textsc{ramses}"

# vary computational parameters
#./main.py --power class cola --h-units --transform-h --subtract-shotnoise --B-lims 0.98 1.02 --figsize 3.5 1.5 --params σ8 ωm0 ωb0 Npart=256,512,1024 --power-stem plots/Npart --Blabel "B / (h_\mathrm{BD}/h_\mathrm{GR})^{-3}"  --legend-location "upper right" "upper left"
#./main.py --power class cola --h-units --transform-h --subtract-shotnoise --B-lims 0.98 1.02 --figsize 3.5 1.5 --params σ8 ωm0 ωb0 Ncell=256,512,1024 --power-stem plots/Ncell --Blabel "B / (h_\mathrm{BD}/h_\mathrm{GR})^{-3}"  --legend-location "upper right" "upper left"
#./main.py --power class cola --h-units --transform-h --subtract-shotnoise --B-lims 0.98 1.02 --figsize 3.5 1.5 --params σ8 ωm0 ωb0 Nstep=30,300,2000 --power-stem plots/Nstep --Blabel "B / (h_\mathrm{BD}/h_\mathrm{GR})^{-3}"  --legend-location "upper right" "upper left"
#./main.py --power class cola --h-units --transform-h --subtract-shotnoise --B-lims 0.98 1.02 --figsize 3.5 1.5 --params σ8 ωm0 ωb0 zinit=10,30,50 --power-stem plots/zinit --Blabel "B / (h_\mathrm{BD}/h_\mathrm{GR})^{-3}"  --legend-location "upper right" "upper left"
#./main.py --power class cola --h-units --transform-h --subtract-shotnoise --B-lims 0.98 1.02 --figsize 3.5 1.5 --params σ8 ωm0 ωb0 Lh=128,384,640 --power-stem plots/Lh --Blabel "B / (h_\mathrm{BD}/h_\mathrm{GR})^{-3}"  --legend-location "upper right" "upper left"

# vary physical parameters
#./main.py --power class cola --h-units --transform-h --subtract-shotnoise --B-lims 0.98 1.02 --figsize 3.5 1.5 --params σ8 ωm0 ωb0 ω=100.0,500.0,1000.0 --Blabel "B / (h_\mathrm{BD}/h_\mathrm{GR})^{-3}" --power-stem plots/w --legend-location "upper right" "upper left"
#./main.py --power class cola --h-units --transform-h --subtract-shotnoise --B-lims 0.98 1.02 --figsize 3.5 1.5 --params σ8 ωm0 ωb0 G0=0.95,1.0,1.05 --Blabel "B / (h_\mathrm{BD}/h_\mathrm{GR})^{-3}" --power-stem plots/G --legend-location "upper right" "upper left"
#./main.py --power class cola --h-units --transform-h --subtract-shotnoise --B-lims 0.98 1.02 --figsize 3.5 1.5 --params σ8 ωb0 ωm0=0.10,0.15,0.20 --Blabel "B / (h_\mathrm{BD}/h_\mathrm{GR})^{-3}" --power-stem plots/wm0 --legend-location "upper right" "upper left"
#./main.py --power class cola --h-units --transform-h --subtract-shotnoise --B-lims 0.98 1.02 --figsize 3.5 1.5 --params σ8 ωm0 ωb0=0.01,0.02,0.03 --Blabel "B / (h_\mathrm{BD}/h_\mathrm{GR})^{-3}" --power-stem plots/wb0 --legend-location "upper right" "upper left"
#./main.py --power class cola --h-units --transform-h --subtract-shotnoise --B-lims 0.98 1.02 --figsize 3.5 1.5 --params ωm0 ωb0 σ8=0.75,0.80,0.85 --Blabel "B / (h_\mathrm{BD}/h_\mathrm{GR})^{-3}" --power-stem plots/s8 --legend-location "upper right" "upper left"
#./main.py --power class cola --h-units --transform-h --subtract-shotnoise --B-lims 0.98 1.02 --figsize 3.5 1.5 --params σ8 ωm0 ωb0 ns=0.92,0.96,1.0 --Blabel "B / (h_\mathrm{BD}/h_\mathrm{GR})^{-3}" --power-stem plots/ns --legend-location "upper right" "upper left"
#./main.py --power class cola --h-units --transform-h --subtract-shotnoise --B-lims 0.98 1.02 --figsize 3.5 1.5 --params σ8 ωm0 ωb0 h=0.65,0.70,0.73 --Blabel "B / (h_\mathrm{BD}/h_\mathrm{GR})^{-3}" --power-stem plots/h --legend-location "upper right" "upper left"
#./main.py --power       cola --h-units --transform-h --subtract-shotnoise --divide class --B-lims 0.98 1.02 --figsize 3.5 1.5 --one-by-one --Blabel "B_\mathrm{COLA} / B_\mathrm{linear}" --params z=0.0,1.5,3.0 h=0.70,0.65,0.73 ns=1.0,0.96,0.92 G0=1.0,0.95,1.05 σ8=0.80,0.75,0.85 ωm0=0.15,0.10,0.20 ωb0=0.02,0.01,0.03 ω=100.0,500.0,1000.0 --power-stem plots/all --legend-location "upper left" ""

# parameter space sampling of ω and ωm0
#./main.py --params lgω=2.0,4.0 σ8 ωb0 z=0 ωm0=0.1,0.2 --samples 100 --parameter-space --power class cola --divide class      --h-units --subtract-shotnoise --transform-h --B-lims 0.99 1.01 --power-stem plots/sample_z0_divclass --figsize 3.0 1.2
#./main.py --params lgω=2.0,4.0 σ8 ωb0 z=0 ωm0=0.1,0.2 --samples 100 --parameter-space --power class cola --divide primordial --h-units --subtract-shotnoise --transform-h --B-lims 1.0 1.2 --power-stem plots/sample_z0_divprim
#./main.py --params lgω=2.0,4.0 σ8 ωb0 z=3 ωm0=0.1,0.2 --samples 100 --parameter-space --power class cola --divide class      --h-units --subtract-shotnoise --transform-h --B-lims 0.99 1.01 --power-stem plots/sample_z3_divclass --figsize 3.0 1.2
#./main.py --params lgω=2.0,4.0 σ8 ωb0 z=3 ωm0=0.1,0.2 --samples 100 --parameter-space --power class cola --divide primordial --h-units --subtract-shotnoise --transform-h --B-lims 1.0 1.2 --power-stem plots/sample_z3_divprim

# EuclidEmulator2
#./main.py --power class ee2 --h-units --transform-h --params σ8=0.81 ωm0 ωb0 z=0,1.5,3,9 --subtract-shotnoise --divide primordial --B-lims 0.8 1.2 --power-stem plots/ee2

# test final script
#./main.py --power class ee2 script --h-units --transform-h --params σ8=0.81 ωm0 ωb0 z=0,1.5,3 --subtract-shotnoise --divide primordial --B-lims 0.8 1.2 --power-stem plots/script

# test hmcode
#./main.py --power hmcode              --h-units --transform-h --params ω=1000 σ8=0.82 ωm0 ωb0 z=0,1.5,3 --subtract-shotnoise --B-lims 0.9 1.1 --power-stem plots/hmcode1 # --legend-location "upper right" "upper left"
#./main.py --power script hmcode class --h-units --transform-h --params ω=1000 σ8=0.82 ωm0 ωb0 z=0,1.5,3 --subtract-shotnoise --B-lims 0.9 1.1 --power-stem plots/hmcode2 --Blabel "B / (h_\mathrm{BD}/h_\mathrm{GR})^{-3}" --legend-location "upper right" "lower left"
#./main.py --power script hmcode class --h-units --transform-h --params ω=1e8 σ8=0.82 ωm0 ωb0 z=0,1.5,3 --subtract-shotnoise --B-lims 0.9 1.1 --power-stem plots/hmcode2_100000000 --legend-location "upper right" "lower left" --Blabel "B / (h_\mathrm{BD}/h_\mathrm{GR})^{-3}" --legend-location "upper right" "lower left"
