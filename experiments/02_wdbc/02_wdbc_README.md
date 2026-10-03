# 02_wdbc — WDBC classification (Section 3.3)

Binary classification of the Wisconsin Diagnostic Breast Cancer (WDBC) dataset
(569 samples, 30 features) with Leave-One-Out Cross-Validation (LOOCV).
The 30 features are Min-Max scaled to [0, 1] and fed to each sequential model
as a length-30 sequence, one feature per step.

## Contents

```
02_wdbc/
├── QRU/                          # QRU, 35 trainable parameters (Table 4)
│   ├── I1_H8_O2 (J8 3 layers rx_ry 100x LOOCV) seed_0.py / .ipynb   # training script
│   ├── ..._results.csv / .xlsx            # per-fold LOOCV results
│   ├── ..._wdbc_loss_history.csv          # training loss per fold (Supplementary Figure S4)
│   ├── Result.ipynb                       # accuracy, confusion matrix, derived metrics
│   ├── Loss.ipynb                         # loss-curve figure
│   ├── SQGRU_j8.py, VQC_j8.py             # QRU circuit (SQGRU = earlier name of QRU)
│   ├── TRAIN_v4_debug_3.py                # training loop (shared by all models below)
│   └── WDBC_dataset.py                    # data loading and Min-Max scaling
├── Classical MLP & logistic/     # classical counterparts trained in this study (Table 5)
│   ├── wdbc_classical_loocv.py
│   └── WDBC_classical_summary (MLP & logistic).csv
└── Classical GRU/                # classical counterparts trained in this study (Table 5)
    ├── wdbc_classical_loocv.py
    └── WDBC_classical_summary (GRU).csv
```

## Classical counterparts

All classical models use the same pipeline as QRU: the same data loading and
Min-Max scaling (`WDBC_dataset.py`), the same LOOCV loop over all 569 samples,
the same training loop (`TRAIN_v4_debug_3.py`: Adam, initial learning rate 0.01,
batch size 20, same learning-rate schedule and early-stopping rule), and binary
cross-entropy loss. Only the model is replaced.

| Model key | Model | Input | Trainable parameters |
|---|---|---|---|
| `logistic` | Logistic regression | all 30 features at once | 31 |
| `mlp1` | MLP, 1 hidden unit (tanh) | all 30 features at once | 33 |
| `mlp2` | MLP, 2 hidden units (tanh) | all 30 features at once | 65 |
| `gru2` | GRU, hidden size 2 | one feature per step | 27 |
| `gru3` | GRU, hidden size 3 | one feature per step | 49 |

The GRU cell uses one bias vector per gate: 3 × (h + h² + h) + (h + 1) parameters.

Run one model per call (optional second argument: seed, default 0):

```bash
python wdbc_classical_loocv.py logistic
python wdbc_classical_loocv.py gru2
```

Each run appends one line to `WDBC_classical_summary.csv`:
model, seed, trainable parameters, LOOCV accuracy, TP, TN, FP, FN, run time.
Malignant samples are the positive class (TP = correctly classified malignant).

## Per-fold training checkpoints

The intermediate training checkpoints (.pkl) of all 569 folds of each model are
not stored in this repository. They are archived as zip files in Release v1.0.0:
https://github.com/lostsodium/quantum-recurrent-unit/releases/tag/v1.0.0

| File | Model |
|---|---|
| Checkpoint_QRU.zip | QRU |
| Checkpoint_logistic.zip | Logistic regression |
| Checkpoint_mlp1.zip | MLP, 1 hidden unit |
| Checkpoint_mlp2.zip | MLP, 2 hidden units |
| Checkpoint_gru2.zip | GRU, hidden size 2 |
| Checkpoint_gru3.zip | GRU, hidden size 3 |
