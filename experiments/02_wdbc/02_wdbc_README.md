# 02_wdbc — WDBC classification (Section 3.3)

Binary classification of the Wisconsin Diagnostic Breast Cancer (WDBC) dataset
(569 samples, 30 features) with Leave-One-Out Cross-Validation (LOOCV).
The 30 features are Min-Max scaled to [0, 1]. QRU reads them as a length-30
sequence, one feature per step; the MLPs take all 30 features at once.

## Contents

```
02_wdbc/
├── QRU/                          # QRU, 35 trainable parameters (Tables 4-5)
│   ├── I1_H8_O2 (J8 3 layers rx_ry 100x LOOCV) seed_0.py / .ipynb   # training script
│   ├── ..._results.csv / .xlsx            # per-fold LOOCV results
│   ├── ..._wdbc_loss_history.csv          # training loss per fold (Supplementary Figure S4)
│   ├── Result.ipynb                       # accuracy, confusion matrix, derived metrics
│   ├── Loss.ipynb                         # loss-curve figure
│   ├── SQGRU_j8.py, VQC_j8.py             # QRU circuit (SQGRU = earlier name of QRU)
│   ├── TRAIN_v4_debug_3.py                # training loop (shared with the MLPs below)
│   └── WDBC_dataset.py                    # data loading and Min-Max scaling
└── Classical MLP/                # classical counterparts trained in this study (Table 5)
    ├── wdbc_mlp_loocv.py                  # MLPs with 2 and 14 hidden neurons
    ├── wdbc_mlp_weka_loocv.py             # MLP with the WEKA default architecture
    ├── WDBC_mlp_results.csv               # LOOCV results of the three MLPs
    ├── TRAIN_v4_debug_3.py                # same file as in QRU/
    └── WDBC_dataset.py                    # same file as in QRU/
```

## Classical counterparts (Table 5)

All MLPs use the same pipeline as QRU: the same data loading and Min-Max
scaling (`WDBC_dataset.py`), the same LOOCV loop over all 569 samples, the same
training loop (`TRAIN_v4_debug_3.py`: Adam, initial learning rate 0.01, batch
size 20, same learning-rate schedule and early-stopping rule), and binary
cross-entropy loss. Only the model is replaced. Each model was run once (seed 0).

| Model key | Model | Trainable parameters | LOOCV accuracy |
|---|---|---|---|
| `mlp_2` | 30 → 2 sigmoid hidden → 1 linear output (architecture of Guo & Nandi [40]) | 65 | 96.84% (551/569) |
| `mlp_14` | 30 → 14 sigmoid hidden → 1 linear output (architecture of Guo & Nandi [40]) | 449 | 97.36% (554/569) |
| `mlp_weka16` | 30 → 16 sigmoid hidden → 2 sigmoid outputs (WEKA default [44], used in [43]) | 530 | 97.01% (552/569) |

- `mlp_2` and `mlp_14`: 32h + 1 parameters for h hidden neurons; sigmoid BCE on
  the single output. `wdbc_mlp_loocv.py` also defines `mlp_3` and `mlp_4`, which
  are not used in the paper.
- `mlp_weka16`: hidden size (30 attributes + 2 classes)/2 = 16, one output per
  class; sigmoid BCE on each output with one-hot targets; the predicted class is
  the larger of the two outputs.

Run:

```bash
python wdbc_mlp_loocv.py mlp_2        # optional second argument: seed (default 0)
python wdbc_mlp_loocv.py mlp_14
python wdbc_mlp_weka_loocv.py         # optional argument: seed (default 0)
```

`wdbc_mlp_loocv.py` appends one line per run to `WDBC_classical_summary.csv`, and
`wdbc_mlp_weka_loocv.py` to `WDBC_mlp_summary.csv`. `WDBC_mlp_results.csv`
combines the three runs reported in the paper, with the columns:
model, seed, trainable parameters, LOOCV accuracy, TP, TN, FP, FN, run time (min).
Malignant samples are the positive class (TP = correctly classified malignant).
Sensitivity, specificity, precision, and F1 in Table 5 follow from TP, TN, FP, FN.
