# Non-linear Brans-Dicke power spectrum predictor

The script `bd.py` predicts the non-linear Brans-Dicke power spectrum.

## Requirements

* [hi_class](https://github.com/miguelzuma/hi_class_public/) (tested with version [16ae0f6](https://github.com/miguelzuma/hi_class_public/tree/16ae0f6ccfcee513146ec36b690678f34fb687f4))
* [EuclidEmulator2](https://github.com/miknab/EuclidEmulator2/) (may need to be compiled with absolute `PATH_TO_EE2_DATA_FILE`)

## Usage

See `./bd.py -h` for instructions. For example, to 

``
./bd.py -w 100 -G 1 -m 0.15 -b 0.02 -H 0.7 -n 1 -z 0 -A 2.0e-9 --hiclass ~/local/hi_class_public/class --ee2 ~/local/EuclidEmulator2-pywrapper/ee2.exe PBD > output.dat
``
