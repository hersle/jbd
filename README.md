# Non-linear Brans-Dicke power spectrum predictor

The script `bd.py` predicts the non-linear Brans-Dicke power spectrum.

## Requirements

* [hi_class](https://github.com/miguelzuma/hi_class_public/) (tested with version [16ae0f6](https://github.com/miguelzuma/hi_class_public/tree/16ae0f6ccfcee513146ec36b690678f34fb687f4))
* [EuclidEmulator2](https://github.com/miknab/EuclidEmulator2/) (may need to be compiled with absolute `PATH_TO_EE2_DATA_FILE`)
* `numpy` and `scipy`

## Usage

See `./bd.py -h` for full instructions.
For example, 

```
./bd.py -w 100 -G 1 -m 0.15 -b 0.02 -H 0.7 -A 2.0e-9 -n 1 -z 0 1 2 3 --hiclass path/to/hiclass/executable --ee2 path/to/EuclidEmulator2/executable PBD > output.dat
```

predicts $P_\mathrm{BD}(k,z)$ with
$\omega = 100$, $G_0 = 1$, $\omega_{m0} = 0.15$, $\omega_{b0} = 0.02$, $h = 0.7$, $A_s = 2 \cdot 10^{-9}$ and $n_s = 1$
at redshifts $z=\{0,1,2,3\}$ using the given `hi_class` and `EuclidEmulator2` executables,
and writes the results to `output.dat`.
