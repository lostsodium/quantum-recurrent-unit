# Data Sources

This directory does not contain raw data files. Instead, this README
describes where the datasets used in each experiment come from and
how to obtain them.

## Oscillation Prediction (Section 3.2)

The damped oscillation time series is generated synthetically. See the
data-generation cell in `experiments/01_oscillation/`.

## WDBC Classification (Section 3.3)

Wisconsin Diagnostic Breast Cancer (WDBC) dataset, available from the
UCI Machine Learning Repository:

https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic

It is also directly accessible via `sklearn.datasets.load_breast_cancer()`.

## MNIST Digit Recognition (Section 3.4)

Binary classification of digits "3" vs "5", downsampled to 8x8.

<!-- TODO: Add the specific source / preprocessing reference used for
     the 8x8 MNIST subset (originally adapted from another repository).
     Replace this note with the correct citation/link before final
     submission. -->

---

*This file is a placeholder and will be updated with complete dataset
provenance details before final submission.*
