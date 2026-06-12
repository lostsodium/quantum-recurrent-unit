# Data Sources

This directory contains processed datasets and dataset-loading utilities
used across the experiments in `experiments/01-03` and
`experiments/04_hardware_validation/`.

```
data/
├── README.md                  # this file
├── mnist_pixels_3-5_8x8.pkl    # processed MNIST "3 vs 5" data (8x8)
├── MNIST_3_5_dataset.py        # MNIST dataset loader
├── WDBC_dataset.py             # WDBC dataset loader
└── provenance/
    └── generate_mnist.ipynb    # original generation script (see below)
```

---

## Oscillation Prediction (Section 3.2)

The damped oscillation time series is generated synthetically. See the
data-generation cell in `experiments/01_oscillation/`. No external data
files are required.

---

## WDBC Classification (Section 3.3)

The Wisconsin Diagnostic Breast Cancer (WDBC) dataset is loaded directly
via scikit-learn:

```python
from sklearn.datasets import load_breast_cancer
```

`WDBC_dataset.py` provides a thin wrapper that:
- loads the dataset via the above call,
- applies min-max normalization to each feature (`normalize2D`),
- splits the data into class-balanced (benign/malignant) subsets
  according to user-specified ratios.

This loader is used by `experiments/02_wdbc/` and the WDBC notebooks
under `experiments/04_hardware_validation/`.

---

## MNIST Digit Recognition (Section 3.4)

Binary classification of handwritten digits "3" vs "5", downsampled to
8x8 pixels.

### Provenance

The 8x8 "3 vs 5" pixel data originates from the data-generation pipeline
accompanying:

> J. Bowles, S. Ahmed, M. Schuld, "Better than classical? The subtle art
> of benchmarking quantum machine learning models," arXiv:2403.07059
> (2024). [Reference 25 in the paper.]
>
> Repository: https://github.com/XanaduAI/qml-benchmarks
> (Licensed under the Apache License, Version 2.0)

`provenance/generate_mnist.ipynb` is a copy of the relevant script from
that repository (license header preserved). It downloads the original
MNIST dataset (via `torchvision`/`keras`) and applies the `"cg"`
(coarse-grained) preprocessing: digits 3 and 5 are selected, each image
is resized to the target resolution (here, 8x8), flattened, and
standardized (`StandardScaler`). This produces
`mnist_pixels_3-5_8x8_{train,test}.csv`.

### Processing into `mnist_pixels_3-5_8x8.pkl`

The train/test CSVs produced above were combined into a single pickle
file, `mnist_pixels_3-5_8x8.pkl`, containing a `(train_dataset,
test_dataset)` tuple of `(image, label)` pairs, where each image is an
8x8 array and labels are `0` (digit "3") or `1` (digit "5"). The file
contains 11,552 training samples and 1,902 test samples.

`MNIST_3_5_dataset.py` loads this pickle and applies an additional
global min-max normalization (across both train and test sets) to map
pixel values to `[0, 1]`, which is the input range expected by the QRU
data-encoding circuit. This loader is used by `experiments/03_mnist/`
and the MNIST notebooks under `experiments/04_hardware_validation/`.

### Note on intermediate files

The `provenance/generate_mnist.ipynb` script also produces several other
resolutions (4x4, 16x16, 32x32) and the raw MNIST files, none of which
are used by this project and are therefore not included in this
repository. Only the final 8x8 "3 vs 5" pickle (`mnist_pixels_3-5_8x8.pkl`)
is provided, as it is the data actually consumed by the QRU experiments.
If you wish to regenerate it from scratch, run
`provenance/generate_mnist.ipynb` (requires `torch`, `torchvision`, and
`keras` — not listed in the main `requirements.txt` since they are only
needed for this regeneration step).
