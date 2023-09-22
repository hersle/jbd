# Brans-Dicke (BD) vs. General Relativity (GR)

## Requirements

* `python3` with `numpy`, `scipy` and `matplotlib`
* [`hi_class`](https://github.com/miguelzuma/hi_class_public/) for $P(k)$ from linear perturbations
* [`FML/COLASolver`](https://github.com/HAWinther/FML/tree/master/FML/COLASolver) for $P(k)$ from COLA $N$-body simulations
* [`ramses`](https://arxiv.org/abs/astro-ph/0111367) with *and* without [`JordanBransDicke`](https://github.com/HAWinther/JordanBransDicke) patch, and [`FML/RamsesUtils/ramses2pk`](https://github.com/hersle/FML/tree/master/FML/RamsesUtils/ramses2pk), for $P(k)$ from AMR $N$-body simulations

Compile them and set the executable paths in `sim.py`.

## Usage

```sh
main.py --help # show instructions
main.py --list-params # list available parameters
```

### Operation

The program runs Class, FML/COLA and Ramses when required.
The data is stored like this:

* `$DATA/jbdsims/` is the main data directory.
* `$DATA/jbdsims/{BD,GR}/` contains the results of each model.
* `$DATA/jbdsims/{BD,GR}/NAME` contains the results of universe NAME, which is a hash of its parameters and initial conditions seed.
* `$DATA/jbdsims/{BD,GR}/NAME/parameters.json` contains the parameters and initial conditions seed of a universe.
* `$DATA/jbdsims/{BD,GR}/NAME/{class,cola,ramses}` contains the data and logs from Class, FML/COLA and Ramses.

To get an overview of the data directory, do `tree -L 3 $DATA/jbdsims`, for example.

To regenerate results for a universe, delete its directory and re-run the program.

### Examples

Plot $P(k)$ and $B(k)$ from Class, COLA and Ramses for varying redshifts with fixed fiducial $\sigma_8$, $\omega_{b0}$, $\omega_{m0}$ and $h$:

```sh
main.py --params σ8 ωb0 ωm0 z=0,1,2,3,4,5 --power class cola ramses
```

As above, but use $h_\mathrm{GR} = h_\mathrm{BD} \sqrt{\phi_\mathrm{ini}}$ *and* plot power spectrum and boost with $P(k/h) \cdot h^3$ instead of $P(k)$:

```sh
main.py --params σ8 ωb0 ωm0 z=0,1,2,3,4,5 --transform-h --h-units --power class cola ramses
```

As above, but plot evolution of selected background and perturbation quantities (from Class) instead:

```sh
main.py --params σ8 ωb0 ωm0 --transform-h --evolution
```
