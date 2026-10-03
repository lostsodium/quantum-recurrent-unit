#!/usr/bin/env python
# coding: utf-8
# =============================================================================
# In-house classical baselines on WDBC under the SAME LOOCV protocol as QRU
# (Reviewer 2, Major Comment 1)
#
# Reuses the QRU pipeline unchanged:
#   - WDBC_dataset(SEED)          same data loading / Min-Max scaling / order
#   - TRAIN_v4_debug_3.TRAIN      same optimizer, lr 0.01, batch 20,
#                                 early stopping (every 10 epochs, <1% x 3, train loss)
#   - sigmoid BCE on logits       same loss
#   - result_fn                   same accuracy rule (|sigmoid - y| < 0.5)
# Only the model (init_fun / apply_fun) is replaced.
#
# Usage (one model per run; can run several terminals in parallel):
#   python wdbc_classical_loocv.py logistic
#   python wdbc_classical_loocv.py mlp1
#   python wdbc_classical_loocv.py gru2
#   python wdbc_classical_loocv.py gru3
# Optional second argument = seed (default 0, same as QRU).
#
# Per-fold files written by TRAIN use a NEW prefix ("WDBC_classical_<model>_s<seed>_###"),
# so nothing from the QRU runs is overwritten. A summary is appended to
# WDBC_classical_summary.csv.
# =============================================================================
import sys, csv, time
import jax
jax.config.update("jax_enable_x64", True)
jax.config.update('jax_platform_name', 'cpu')
import jax.numpy as jnp
from jax import lax
from jax.nn.initializers import glorot_normal, orthogonal, zeros
import optax

from WDBC_dataset import WDBC_dataset
from TRAIN_v4_debug_3 import TRAIN

MODEL = sys.argv[1] if len(sys.argv) > 1 else 'logistic'
SEED  = int(sys.argv[2]) if len(sys.argv) > 2 else 0
N_STEPS, BATCH_SIZE = 1000000, 20
N_FEAT = 30                      # WDBC features, fed as a length-30 sequence (as in QRU)

# -----------------------------------------------------------------------------
# Models (stax-style init_fun / apply_fun). Inputs have shape (batch, 30, 1).
# init_fun ignores input_shape so it works with whatever TRAIN passes.
# -----------------------------------------------------------------------------
def Logistic():
    """Linear model on the 30 features: 30 + 1 = 31 parameters."""
    def init_fun(rng, input_shape):
        return (-1,), (glorot_normal()(rng, (N_FEAT, 1)), jnp.zeros(1))
    def apply_fun(params, x, **kw):
        W, b = params
        return (x.reshape(x.shape[0], -1) @ W + b)[:, 0]
    return init_fun, apply_fun

def MLP(h):
    """30 -> h (tanh) -> 1: 31h + h + 1 parameters (h=1 -> 33)."""
    def init_fun(rng, input_shape):
        k1, k2 = jax.random.split(rng)
        return (-1,), (glorot_normal()(k1, (N_FEAT, h)), jnp.zeros(h),
                       glorot_normal()(k2, (h, 1)), jnp.zeros(1))
    def apply_fun(params, x, **kw):
        W1, b1, W2, b2 = params
        z = jnp.tanh(x.reshape(x.shape[0], -1) @ W1 + b1)
        return (z @ W2 + b2)[:, 0]
    return init_fun, apply_fun

def GRU(h):
    """Reads the 30 features one per step (input dim 1), like QRU.
    Params: 3 gates x (1*h + h*h + h) + readout (h + 1).  h=2 -> 27, h=3 -> 49."""
    def init_fun(rng, input_shape):
        ks = jax.random.split(rng, 7)
        gate = lambda kw, ku: (glorot_normal()(kw, (1, h)), orthogonal()(ku, (h, h)), jnp.zeros(h))
        return (-1,), (gate(ks[0], ks[1]), gate(ks[2], ks[3]), gate(ks[4], ks[5]),
                       (glorot_normal()(ks[6], (h, 1)), jnp.zeros(1)))
    def apply_fun(params, x, **kw):
        (Wz, Uz, bz), (Wr, Ur, br), (Wn, Un, bn), (Wo, bo) = params
        def step(hid, xt):                                  # xt: (batch, 1)
            z = jax.nn.sigmoid(xt @ Wz + hid @ Uz + bz)
            r = jax.nn.sigmoid(xt @ Wr + hid @ Ur + br)
            n = jnp.tanh(xt @ Wn + (r * hid) @ Un + bn)
            return (1 - z) * n + z * hid, None
        xs = jnp.swapaxes(x.reshape(x.shape[0], N_FEAT, 1), 0, 1)   # (30, batch, 1)
        hT, _ = lax.scan(step, jnp.zeros((x.shape[0], h)), xs)
        return (hT @ Wo + bo)[:, 0]
    return init_fun, apply_fun

MODELS = {'logistic': Logistic(), 'mlp1': MLP(1), 'mlp2': MLP(2),
          'gru2': GRU(2), 'gru3': GRU(3)}
init_fun, apply = MODELS[MODEL]

n_params = sum(p.size for p in jax.tree_util.tree_leaves(
    init_fun(jax.random.PRNGKey(0), (-1, N_FEAT, 1))[1]))
print(f"Model: {MODEL} | trainable parameters: {n_params} | seed: {SEED}")

# -----------------------------------------------------------------------------
# Loss / accuracy: identical to the QRU script
# -----------------------------------------------------------------------------
@jax.jit
def loss_fn(params, inputs, targets):
    return jnp.mean(optax.sigmoid_binary_cross_entropy(apply(params, inputs), targets))

def result_fn(params, dataset):
    inp, tar = zip(*dataset)
    result = jax.nn.sigmoid(apply(params, jnp.array(inp)))
    return sum(abs(r - t) < 0.5 for r, t in zip(result, tar)) / len(result)

# -----------------------------------------------------------------------------
# LOOCV loop: identical to the QRU script
# -----------------------------------------------------------------------------
all_dataset = WDBC_dataset(SEED).all_dataset
key = jax.random.PRNGKey(SEED)
prefix = f"WDBC_classical_{MODEL}_s{SEED}"
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

    # acc_results[0][1] = accuracy of the lowest-training-loss params on the
    # held-out sample (same quantity the QRU script accumulates).
    correct = train.acc_results[0][1] > 0.5
    # sklearn load_breast_cancer: target 0 = malignant, 1 = benign.
    # Positive class = malignant, as in the paper's Table 4 (TP=200 of 212 malignant).
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

with open('WDBC_classical_summary.csv', 'a', newline='') as f:
    csv.writer(f).writerow([MODEL, SEED, n_params, f"{acc:.6f}", tp, tn, fp, fn,
                            f"{(time.time()-t0)/60:.1f}"])
