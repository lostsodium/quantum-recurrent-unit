#!/usr/bin/env python
# coding: utf-8
# =============================================================================
# WDBC: MLP matching the WEKA MultilayerPerceptron default used in [43]
# (Salama et al.), trained under the SAME LOOCV protocol as QRU.
#
# Architecture (WEKA default for a nominal class):
#   30 inputs -> 16 sigmoid hidden neurons -> 2 sigmoid outputs (one per class)
#   hidden size = (30 attributes + 2 classes) / 2 = 16
#   parameters  = 30*16 + 16 + 16*2 + 2 = 530
#
# Reuses the QRU pipeline unchanged:
#   - WDBC_dataset(SEED)          same data loading / Min-Max scaling / order
#   - TRAIN_v4_debug_3.TRAIN      same optimizer, lr 0.01, batch 20,
#                                 early stopping (every 10 epochs, <1% x 3, train loss)
#   - BCE loss                    one sigmoid BCE per output, one-hot targets
#   - prediction                  class of the larger output (as in WEKA)
#
# Usage:
#   python wdbc_mlp_weka_loocv.py          # seed 0 (same as QRU)
#   python wdbc_mlp_weka_loocv.py 0        # optional seed argument
#
# Per-fold files use the prefix "WDBC_mlp_weka16_s<seed>_###"; the summary is
# appended to WDBC_mlp_summary.csv in the same format as WDBC_classical_summary.csv.
# =============================================================================
import sys, csv, time
import jax
jax.config.update("jax_enable_x64", True)
jax.config.update('jax_platform_name', 'cpu')
import jax.numpy as jnp
from jax.nn.initializers import glorot_normal
import optax

from WDBC_dataset import WDBC_dataset
from TRAIN_v4_debug_3 import TRAIN

MODEL = 'mlp_weka16'
H = 16
SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 0
N_STEPS, BATCH_SIZE = 1000000, 20
N_FEAT = 30

# -----------------------------------------------------------------------------
# Model: inputs have shape (batch, 30, 1); output (batch, 2) pre-activations
# ordered [malignant (label 0), benign (label 1)].
# -----------------------------------------------------------------------------
def init_fun(rng, input_shape):
    k1, k2 = jax.random.split(rng)
    return (-1,), (glorot_normal()(k1, (N_FEAT, H)), jnp.zeros(H),
                   glorot_normal()(k2, (H, 2)), jnp.zeros(2))

def apply(params, x, **kw):
    W1, b1, W2, b2 = params
    z = jax.nn.sigmoid(x.reshape(x.shape[0], -1) @ W1 + b1)
    return z @ W2 + b2

n_params = sum(p.size for p in jax.tree_util.tree_leaves(
    init_fun(jax.random.PRNGKey(0), (-1, N_FEAT, 1))[1]))
print(f"Model: {MODEL} | trainable parameters: {n_params} | seed: {SEED}")
assert n_params == 530, n_params

# -----------------------------------------------------------------------------
# Loss / accuracy
# -----------------------------------------------------------------------------
@jax.jit
def loss_fn(params, inputs, targets):
    t = jnp.asarray(targets, dtype=float)
    onehot = jnp.stack([1 - t, t], axis=1)
    return jnp.mean(optax.sigmoid_binary_cross_entropy(apply(params, inputs), onehot))

def result_fn(params, dataset):
    # Predicted class = larger output. sigmoid(z_benign - z_malignant) > 0.5
    # iff the benign output wins, so the |r - t| < 0.5 rule of the QRU script
    # applies unchanged.
    inp, tar = zip(*dataset)
    z = apply(params, jnp.array(inp))
    result = jax.nn.sigmoid(z[:, 1] - z[:, 0])
    return sum(abs(r - t) < 0.5 for r, t in zip(result, tar)) / len(result)

# -----------------------------------------------------------------------------
# LOOCV loop: identical to the other classical scripts
# -----------------------------------------------------------------------------
all_dataset = WDBC_dataset(SEED).all_dataset
key = jax.random.PRNGKey(SEED)
prefix = f"WDBC_{MODEL}_s{SEED}"
tp = tn = fp = fn = 0
t0 = time.time()

for i in range(len(all_dataset)):
    test_dataset  = [all_dataset[i]]
    train_dataset = all_dataset[:i] + all_dataset[i+1:]
    key, _ = jax.random.split(key)

    train = TRAIN(key, init_fun, loss_fn, train_dataset + test_dataset, test_dataset,
                  result_fn, f"{prefix}_{i:03}")
    train.N_STEPS = N_STEPS
    train.BATCH_SIZE = BATCH_SIZE
    train.NUM_SEEDs = 1
    train.STD_DEV = 0.0
    train.REC_INTE = 10
    train.VARI_FRE = 'epoch'
    train.ini_learning_rate = 0.01
    train.TRAIN_VALID_TEST = jnp.array([len(train_dataset), 1])
    train.ES_THRES = 1e-2
    train.ES_LEN = 3
    train.ES_MODE = 'loss'
    train.ES_DATASET = 'train'
    train.train()

    correct = train.acc_results[0][1] > 0.5
    # sklearn load_breast_cancer: target 0 = malignant, 1 = benign.
    # Positive class = malignant.
    malignant = int(round(float(all_dataset[i][1]))) == 0
    tp += correct and malignant
    tn += correct and not malignant
    fn += (not correct) and malignant
    fp += (not correct) and not malignant
    n = i + 1
    print(f"[{MODEL}] fold {n}/{len(all_dataset)}  running acc = {(tp+tn)/n:.4f}  "
          f"({(time.time()-t0)/60:.1f} min)")

acc = (tp + tn) / len(all_dataset)
print(f"\n=== {MODEL} ({n_params} params, seed {SEED}) ===")
print(f"LOOCV accuracy: {acc*100:.2f}%  ({tp+tn}/{len(all_dataset)})")
print(f"TP={tp} TN={tn} FP={fp} FN={fn}")

with open('WDBC_mlp_summary.csv', 'a', newline='') as f:
    csv.writer(f).writerow([MODEL, SEED, n_params, f"{acc:.6f}", tp, tn, fp, fn,
                            f"{(time.time()-t0)/60:.1f}"])
