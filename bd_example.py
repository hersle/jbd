#!/usr/bin/env python3

from subprocess import run
import matplotlib.pyplot as plt
import numpy as np

zs = [0, 1, 2, 3]
for spectrum in ["PGR", "PGRlin", "PBD", "PBDlin"]:
    run(f"./bd.py -w 100 -G 1 -m 0.15 -b 0.02 -H 0.7 -n 1 -z {' '.join(str(z) for z in zs)} -A 2.0e-9 --hiclass ~/local/hi_class_public/class --ee2 ~/local/EuclidEmulator2-pywrapper/ee2.exe {spectrum} > bd_example_data.dat", shell=True)
    data = np.transpose(np.loadtxt("bd_example_data.dat"))
    k, P = data[0], data[1:]
    for i, z in enumerate(zs):
        plt.plot(np.log10(k), np.log10(P[i]), label=f"{spectrum}(z={z:.2f})")
plt.xlabel(r"$\lg[k/(1/\mathrm{Mpc})]$")
plt.ylabel(r"$\lg[P/\mathrm{Mpc}^3]$")
plt.legend()
plt.savefig("bd_example_plot.png")
print("Saved bd_example_plot.png")
