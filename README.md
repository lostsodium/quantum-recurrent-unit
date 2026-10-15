# Quantum Recurrent Unit (QRU)

Code for the paper:

**Quantum Recurrent Unit: An Effective and Parameter-Efficient Quantum Neural Network Architecture for NISQ Devices**  
Tzong-Daw Wu, Hsi-Sheng Goan  
arXiv: [2601.18164](https://arxiv.org/abs/2601.18164)

> **Note:** In earlier development versions, this architecture was referred to as SQGRU (Simple Quantum GRU). File and module names such as `SQGRU_j8.py` refer to QRU.

---

## Overview

QRU is a quantum recurrent neural network architecture that implements information selection natively in the quantum domain via controlled-SWAP (C-SWAP) operations, combined with a measurement results feedforward scheme for NISQ compatibility. Key properties:

- Constant per-step circuit depth and constant parameter count regardless of sequence length
- 46.2%–99.7% parameter reduction compared to classical counterparts, under statevector simulation
- Validated on three tasks: oscillation prediction, WDBC classification, and MNIST digit recognition
- Preliminary hardware experiments on IBM Quantum processors (ibm_marrakesh, ibm_boston, ibm_phoenix) with several error mitigation methods

---

## Repository Structure

```
quantum-recurrent-unit/
├── README.md
├── requirements.txt
├── experiments/
│   ├── 01_oscillation/                          # Section 3.2 (Tables 1-3, Figures 4-6)
│   ├── 02_wdbc/                                 # Section 3.3 (Tables 4-5); see 02_wdbc_README.md
│   │   ├── QRU/
│   │   └── (classical counterparts trained in this study)
│   ├── 03_mnist/                                # Section 3.4 (Tables 6-7)
│   │   └── cnn_param_count/                     # parameter count of the CNN benchmark (51,841)
│   ├── 04_hardware_validation/                  # Section 3.5
│   │   ├── ni_training/
│   │   │   ├── wdbc/                            # NI training (Table 8)
│   │   │   ├── mnist/                           # NI training (Table 9)
│   │   │   └── mnist_sigma0.05_qesem_params/    # NI-trained parameters used for QESEM inference
│   │   ├── error_mitigation_comparison/         # Table 10, Figure 7 (figure7_error_mitigation_v2.ipynb); OTC calibration (Section S5)
│   │   ├── pec_phoenix/                         # PEC and other methods on ibm_phoenix (Section S8, Tables S8-S9)
│   │   └── qesem_sequential/                    # Figures 8-9
│   └── supplementary/
│       ├── S4_1_dual_basis/                     # Z+X vs Z-only comparison (Table S1)
│       │   ├── v1_single_basis/                 # Z-only notebook
│       │   └── v2_dual_basis/README.md          # points to S4_2 standard_qru (Z+X)
│       ├── S4_2_c-swap_ablation/                # C-SWAP ablation (Table S2, Figure S10)
│       │   ├── standard_qru/                    # (a) full QRU with C-SWAP
│       │   ├── without_c-swap/                  # (b) C-SWAP and ancilla qubits removed
│       │   └── cnot_replacement/                # (c) C-SWAP replaced by CNOT
│       ├── S4_3_feature_ordering/               # WDBC feature permutations (Table S3)
│       └── S7_QLSTMQGRU Comparison/             # QLSTM/QGRU comparison (Table S7)
└── data/
    ├── README.md                                # dataset sources and provenance
    ├── mnist_pixels_3-5_8x8.pkl                 # processed MNIST "3 vs 5" data (8x8)
    ├── MNIST_3_5_dataset.py                     # MNIST dataset loader
    ├── WDBC_dataset.py                          # WDBC dataset loader
    └── provenance/
        └── generate_mnist.ipynb                 # original generation script (Apache 2.0)
```

---

## Experiments

### Experiment 1: Oscillation Prediction (`experiments/01_oscillation/`)
- **Task**: Joint prediction of simple harmonic and damped oscillation time series (temporal memory evaluation)
- **QRU config**: 5 hidden qubits, 4 variational layers of U2 gates; 72 trainable parameters
- **Evaluation**: 50 independent runs with random initialization; outliers identified with Tukey's fences and reported separately
- **Classical counterparts**: GRU and LSTM over a range of sizes; GRU needs 307 parameters (all runs) or 197 parameters (without outliers) for a comparable test MSE

### Experiment 2: WDBC Classification (`experiments/02_wdbc/`)
- **Task**: Binary classification on the Wisconsin Diagnostic Breast Cancer dataset (569 samples, 30 features, fed as a sequence of length 30)
- **QRU config**: 4 hidden qubits, 3 variational layers of alternating Rx–Ry–Rx gates; 35 trainable parameters
- **Evaluation**: Leave-one-out cross-validation (LOOCV); Min-Max feature scaling
- **Result**: 96.13% test accuracy
- **Classical counterparts trained in this study** (same pipeline and LOOCV protocol): MLPs with 2 and 14 hidden neurons (65 and 449 parameters; 96.84% and 97.36%) and an MLP with the WEKA default architecture (530 parameters; 97.01%). See `experiments/02_wdbc/02_wdbc_README.md`.

### Experiment 3: MNIST Digit Recognition (`experiments/03_mnist/`)
- **Task**: Binary classification of handwritten digits "3" vs "5" (8×8 images, one row per time step)
- **QRU config**: 4 hidden qubits, 4 variational layers of Rot gates; 132 trainable parameters
- **Evaluation**: Stratified 7-fold cross-validation
- **Result**: 98.05% test accuracy (CNN benchmark of Bowles et al.: 51,841 parameters, 96.42%)

### Experiment 4: Hardware Validation (`experiments/04_hardware_validation/`)
- **Noise-aware training**: Observable-Targeted Calibration (OTC) noise injection on WDBC and MNIST (Tables 8-9)
- **Error mitigation comparison** (single-step MNIST circuit, Table 10 and Figure 7): readout mitigation, gate twirling, and ZNE (Qiskit Runtime) and QESEM (Qedma) on ibm_marrakesh; PEC (Qiskit Runtime) on ibm_phoenix, measured in 15 independent runs
- **ibm_phoenix** (`pec_phoenix/`): PEC runs (Table S8) and readout mitigation, gate twirling, and ZNE applied to the same transpiled circuit (Table S9)
- **Sequential inference** (Figures 8-9): complete 8-step MNIST inference with QESEM on ibm_marrakesh and ibm_boston

> **Note:** Running the hardware notebooks requires an IBM Quantum account; QESEM additionally requires access to the Qedma QESEM function in the Qiskit Functions Catalog. Experiments 1–3 run without quantum hardware.

---

## Requirements

This repository uses two separate environments:

- **Simulation environment** (PennyLane + JAX): required for `experiments/01-03` and `experiments/supplementary/`
- **Hardware environment** (Qiskit + qiskit-ibm-runtime): required only for `experiments/04_hardware_validation/`

See `requirements.txt` for the specific package versions used to verify these notebooks.

---

## Reproducibility Notes

The notebooks in this repository reflect the actual research workflow and may contain exploratory code, alternative configurations that were not used in the final results, and intermediate outputs. They are provided primarily for transparency and as a reference for the methods and parameters used to obtain the results reported in the paper.

A few practical notes:

- **File paths**: Some notebooks use hardcoded local paths (e.g. for loading parameter files or saving checkpoints). These may need to be adjusted to match your local directory structure.
- **Package versions**: `requirements.txt` lists the package versions used to verify these notebooks at the time of writing. Earlier results (particularly for the oscillation and WDBC experiments) were originally obtained over an extended development period and may have used slightly different versions; minor numerical differences may occur with different package versions.
- **Stochastic results**: Experiments involving random initialization (e.g. the oscillation prediction task, Section 3.2) report statistics over multiple independent runs. Individual runs may differ from the reported mean/std due to random seeding.
- **Hardware experiments**: Results on real hardware depend on the device calibration at the time of execution and are not expected to be reproduced exactly. Account-specific identifiers (CRN, instance names, job IDs) have been replaced with placeholders (`<YOUR_CRN_INSTANCE>`, `<YOUR_INSTANCE_NAME>`, `<YOUR_JOB_ID>`).

If you run into issues reproducing a specific result, feel free to open an issue on this repository.

---

## Citation

If you use this code, please cite:

```bibtex
@article{wu2026qru,
  title={Quantum Recurrent Unit: An Effective and Parameter-Efficient Quantum Neural Network Architecture for NISQ Devices},
  author={Wu, Tzong-Daw and Goan, Hsi-Sheng},
  journal={arXiv preprint arXiv:2601.18164},
  year={2026}
}
```

---

## License

To be determined upon paper acceptance.
