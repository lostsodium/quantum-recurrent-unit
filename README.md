# Quantum Recurrent Unit (QRU)

Code for the paper:

**Quantum Recurrent Unit: An Effective and Parameter-Efficient Quantum Neural Network Architecture for NISQ Devices**  
Tzong-Daw Wu, Hsi-Sheng Goan  
arXiv: [2601.18164](https://arxiv.org/abs/2601.18164)

> **Note:** In earlier development versions, this architecture was referred to as SQGRU (Simple Quantum GRU).

---

## Overview

QRU is a quantum recurrent neural network architecture that implements information selection natively in the quantum domain via controlled-SWAP (C-SWAP) operations, combined with a measurement results feedforward scheme for NISQ compatibility. Key properties:

- Constant per-step circuit depth and constant parameter count regardless of sequence length
- 63.5%–99.5% parameter reduction compared to classical counterparts
- Validated on three tasks: oscillation prediction, WDBC classification, and MNIST digit recognition
- Preliminary hardware validation on IBM Quantum (ibm_marrakesh) with QESEM error mitigation

---

## Repository Structure

```
quantum-recurrent-unit/
├── README.md
├── requirements.txt
├── experiments/
│   ├── 01_oscillation/                          # Section 3.2
│   ├── 02_wdbc/                                 # Section 3.3
│   ├── 03_mnist/                                # Section 3.4
│   ├── 04_hardware_validation/                  # Section 3.5
│   │   ├── ni_training/
│   │   │   ├── wdbc/                            # NI training (Tables 8-9)
│   │   │   ├── mnist/                           # NI training (Tables 8-9)
│   │   │   └── mnist_sigma0.05_qesem_params/    # NI training for QESEM params
│   │   ├── error_mitigation_comparison/         # Figure 7 / Table 10
│   │   └── qesem_sequential/                    # Figure 8, 9
│   └── supplementary/
│       ├── S4_1_dual_basis/                     # Z+X vs Z-only comparison
│       │   ├── v1_single_basis/                 # actual notebook (Z-only)
│       │   └── v2_dual_basis/
│       │       └── README.md                    # points to S4_2 standard_qru (Z+X)
│       ├── S4_2_c-swap_ablation/                # C-SWAP ablation (Table S2 / Fig S9)
│       │   └── v2_dual_basis/
│       │       ├── standard_qru/                # (a) full QRU with C-SWAP
│       │       ├── without_c-swap/              # (b) C-SWAP + ancilla removed
│       │       └── cnot_replacement/            # (c) C-SWAP -> CNOT
│       └── S4_3_feature_ordering/               # WDBC feature permutation (Table S3)
└── data/
    ├── README.md                                # dataset sources & provenance
    ├── mnist_pixels_3-5_8x8.pkl                 # processed MNIST "3 vs 5" data (8x8)
    ├── MNIST_3_5_dataset.py                     # MNIST dataset loader
    ├── WDBC_dataset.py                          # WDBC dataset loader
    └── provenance/
        └── generate_mnist.ipynb                 # original generation script (Apache 2.0)
```

---

## Experiments

### Experiment 1: Oscillation Prediction (`experiments/01_oscillation/`)
- **Task**: Predict damped oscillatory time series (temporal memory evaluation)
- **QRU config**: 5 hidden qubits, 4-layer Rot-based VQC; 72 trainable parameters
- **Evaluation**: 50 independent runs with random initialization; Tukey's fences outlier criterion
- **Classical counterpart**: GRU (307 parameters full results; 197 parameters w/o outlier runs)

### Experiment 2: WDBC Classification (`experiments/02_wdbc/`)
- **Task**: Binary classification on Wisconsin Diagnostic Breast Cancer dataset (569 samples, 30 features)
- **QRU config**: 5 hidden qubits, 4-layer Rot-based VQC; 35 trainable parameters
- **Evaluation**: Leave-one-out cross-validation (LOOCV); Min-Max feature scaling
- **Result**: 96.13% test accuracy (classical counterpart ANN: 167 parameters)

### Experiment 3: MNIST Digit Recognition (`experiments/03_mnist/`)
- **Task**: Binary classification of handwritten digits "3" vs "5" (8×8 downsampled images)
- **QRU config**: 4 hidden qubits, 4-layer Rot-based VQC; 132 trainable parameters
- **Evaluation**: Stratified 7-fold cross-validation
- **Result**: 98.05% test accuracy (classical counterpart CNN: 27,265 parameters)

### Experiment 4: Hardware Validation (`experiments/04_hardware_validation/`)
- **Task**: Noise-aware training and real-hardware evaluation on IBM Quantum
- **NI training**: Observable-Targeted Calibration (OTC) noise injection on WDBC and MNIST
- **Hardware**: ibm_marrakesh via IBM Quantum (NTU Premium Plan)
- **Error mitigation**: QESEM (Qedma) — recovers statevector-level observable estimates to within 0.3% deviation

> **Note:** Running the hardware validation notebooks requires IBM Quantum access. Experiments 1–3 can be run without any quantum hardware.

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
- **Hardware experiments**: Notebooks in `experiments/04_hardware_validation/` require an IBM Quantum account and access to the Qedma QESEM function via the Qiskit Functions Catalog. Account-specific identifiers (CRN, instance names) have been replaced with placeholders (`<YOUR_CRN_INSTANCE>`, `<YOUR_INSTANCE_NAME>`).

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
