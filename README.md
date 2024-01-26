# Non-linear Brans-Dicke power spectrum predictor

The script `bd.py` predicts the non-linear Brans-Dicke power spectrum.

## Requirements

* [hi_class](https://github.com/miguelzuma/hi_class_public/) (tested with version [16ae0f6](https://github.com/miguelzuma/hi_class_public/tree/16ae0f6ccfcee513146ec36b690678f34fb687f4))
* [EuclidEmulator2](https://github.com/miknab/EuclidEmulator2/) (may need to be compiled with absolute `PATH_TO_EE2_DATA_FILE`)

## Usage

See `./bd.py -h` for instructions.
For example, to predict
$P_\mathrm{BD}(k,z=0\,|\,\omega=100, G_0=1, \omega_{m0} = 0.15, \omega_{b0} = 0.02, h = 0.7, A_s = 2, n_s = 1 \cdot 10^{-9}$
and output the results to `output.dat`, run:

``
./bd.py -w 100 -G 1 -m 0.15 -b 0.02 -H 0.7 -n 1 -z 0 -A 2.0e-9 --hiclass path/to/hiclass/executable --ee2 path/to/EuclidEmulator2/executable PBD > output.dat
``
