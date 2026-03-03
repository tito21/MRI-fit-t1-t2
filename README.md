
# $T_1$ and $T_2$ MRI fitting script

This script can perform $T_1$ or $T_2$ fitting to a series of MR images. All relevant sequence parameters are detected
automatically from the DICOM header (tested only in Siemens enhanced DICOMs). The resulting maps are saved in a `.npz`
file and as plots. A mask is automatically applied by Otsu thresholding the brightest of the input images.

For $T_2$ it assumes a Spin Echo (SE) experiment. The signal equation is

$$
S(TE) = S_0 e^{-TE / T_2}
$$

For fitting first an ordinary least squares (OLS) fit is performed on the log of the signal followed by a non-linear
least squares fit.

For the $T_1$ an inversion recovery (IR) experiment is assumed. The TR is assumed to be longer than $5T_1$, but the
inversion pulse is assumed to be imperfect. The signal equation is then:

$$
S(TI) = |S_0 (1 - (1 - k) e^{-TI / T_1})|
$$

The fitting uses a bound constrained non-linear least squares fit. The starting point for fitting is from the
null-crossing method given by the formula:

$$
T_1^0 = \frac{TI_{\text{NULL}}}{\log(2)}
$$

With $TI_{\text{NULL}}$ the inversion time with the lowest signal (assumed to be the inversion time when the signal
crosses zero). The magnetization ($S_0$) is initialized to the value from the longest inversion pulse and the inversion
efficiency ($k$) to -1 (assumed perfect inversion).

## Usage

If you have (uv)[https://docs.astral.sh/uv/getting-started/installation/] installed and using a UNIX system (Linux,
macos) just run the script: `./measure_t1_t2.py ARGS`.

For other platforms or if you don't use uv, the script has these dependencies:

```
requires-python = ">=3.12"
dependencies = [
   "matplotlib>=3.10.6",
   "numpy>=2.3.3",
   "pydicom>=3.0.1",
   "scikit-image>=0.25.2",
   "scipy>=1.16.2",
   "tqdm>=4.67.1"
]
```

### Command line arguments

```
$ ./measure_t1_t2.py -h
usage: measure_t1_t2.py [-h] [--mode {T1,T2}] [--output OUTPUT] [--range RANGE RANGE] [--debug] folder

Measure T1 and T2 from DICOM MRI images

positional arguments:
  folder               Folder containing DICOM images

options:
  -h, --help           show this help message and exit
  --mode {T1,T2}       Relaxation constant to measure
  --output OUTPUT      Output path for the numpy arrays
  --range RANGE RANGE  Display range for the relaxation map
  --debug              Show intermediate results for debugging
```

After doing the fitting it will print to the console the average and standard deviation of the relaxation map inside the
selected mask.