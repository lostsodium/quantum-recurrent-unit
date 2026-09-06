#!/usr/bin/env python
# coding: utf-8
#
# run_QGRU_MNIST.py
# ---------------------------------------------------------------------------
# QGRU MNIST experiment as a single callable function, parameterized by
# (n_qubits, n_layers). Each call:
#   - computes the parameter count
#   - draws the gate VQC circuit (text + saved .png), named by config
#   - trains with a checkpoint name that encodes (n_qubits, n_layers), so
#     different configs never collide or overwrite each other
#   - appends one row to a results CSV (append mode -- safe to call many
#     times with different configs without losing earlier results)
#
# Usage (interactively, e.g. in a notebook):
#   from run_QGRU_MNIST import run_qgru_experiment
#   run_qgru_experiment(n_qubits=5, n_layers=4)
#   run_qgru_experiment(n_qubits=4, n_layers=5, N_STEPS=2000)
#
# Usage (batch, from the command line):
#   python3 run_QGRU_MNIST.py 5 4
#   python3 run_QGRU_MNIST.py 4 5 --n_steps 2000
# ---------------------------------------------------------------------------

import csv
import os
import time

import jax
import jax.numpy as jnp
import pennylane as qml
from sklearn.model_selection import StratifiedKFold

from MNIST_3_5_dataset import MNIST_3_5_dataset
from QGRU_lit_j8 import qgru_lit, qgru_lit_param_count, QGRUGateCirc
from TRAIN_v4_debug_4 import TRAIN

IN_DIM, READOUT_DIM = 8, 2
DEFAULT_HIDDEN_DIM = 8  # matches QRU's own hidden dimension for MNIST
SEED = 2
RESULTS_CSV = 'results_QGRU_MNIST.csv'
CIRCUIT_DIR = 'circuits'

# ---------------------------------------------------------------------------
# Data is loaded once (module import time) and reused across every call --
# building the split is the same regardless of model config.
# ---------------------------------------------------------------------------
_MNIST = MNIST_3_5_dataset(SEED)
_X = _MNIST.dataset
_y = [_[1] for _ in _X]

_outer_kf = StratifiedKFold(n_splits=7, shuffle=True, random_state=SEED)
for _train_index, _test_index in _outer_kf.split(_X, _y):
    _X_train_outer = [_X[i] for i in _train_index]
    _X_test_outer = [_X[i] for i in _test_index]
    _y_train_outer = [_y[i] for i in _train_index]
    _inner_kf = StratifiedKFold(n_splits=6, shuffle=True, random_state=SEED)
    for _inner_train_idx, _inner_val_idx in _inner_kf.split(_X_train_outer, _y_train_outer):
        X_train_inner = [_X_train_outer[i] for i in _inner_train_idx]
        X_val_inner = [_X_train_outer[i] for i in _inner_val_idx]
        break
    break
X_test_outer = _X_test_outer


def draw_circuit(n_qubits, n_layers, save_dir=CIRCUIT_DIR):
    """Draw one gate VQC (reset/update/candidate share this structure).
    Note: the gate VQC circuit structure only depends on (n_qubits,
    n_layers), not hidden_dim, so the diagram is the same regardless of
    which hidden_dim is used."""
    os.makedirs(save_dir, exist_ok=True)
    circ = QGRUGateCirc(n_qubits, n_layers)
    dev = qml.device('default.qubit', wires=n_qubits)
    qnode = qml.QNode(circ, dev)

    dummy_features = jnp.zeros(n_qubits)
    dummy_weights = jnp.zeros(circ.num_weights)

    text_diagram = qml.draw(qnode)(dummy_features, dummy_weights)

    png_path = os.path.join(save_dir, f'QGRU_q{n_qubits}_l{n_layers}.png')
    fig, ax = qml.draw_mpl(qnode)(dummy_features, dummy_weights)
    fig.savefig(png_path, dpi=100, bbox_inches='tight')
    import matplotlib.pyplot as plt
    plt.close(fig)

    return text_diagram, png_path


def run_qgru_experiment(n_qubits, n_layers, hidden_dim=DEFAULT_HIDDEN_DIM,
                         N_STEPS=500, NUM_SEEDS=1,
                         draw=True, verbose=True, results_csv=RESULTS_CSV):
    """Run one QGRU MNIST experiment for a given (n_qubits, n_layers),
    with hidden_dim defaulting to 8 (matching QRU's own hidden dimension).
    Uses readout_mode='slice' (hidden_dim >= readout_dim=2, output is the
    first 2 components of the final hidden state -- no extra readout
    parameters, mirroring QRU's own readout convention).

    Returns a dict with n_params, test_acc, elapsed_seconds,
    checkpoint_name, circuit_png (if draw=True).
    """
    config_tag = f'h{hidden_dim}_q{n_qubits}_l{n_layers}'
    checkpoint_name = f'QGRU_MNIST_{config_tag}'

    n_params = qgru_lit_param_count(IN_DIM, hidden_dim, n_qubits, n_layers,
                                     readout_dim=READOUT_DIM, readout_mode='slice')

    circuit_png = None
    if draw:
        text_diagram, circuit_png = draw_circuit(n_qubits, n_layers)
        if verbose:
            print(f"--- circuit ({config_tag}) ---")
            print(text_diagram)
            print(f"(saved to {circuit_png})")

    if verbose:
        print(f"[{config_tag}] parameter count: {n_params}  "
              f"(QRU comparison point: 131)")

    init_fun, apply_fun = qgru_lit(IN_DIM, hidden_dim, n_qubits, n_layers,
                                    readout_dim=READOUT_DIM, readout_mode='slice')

    @jax.jit
    def loss_fn(params, inputs, targets):
        logits = apply_fun(params, inputs)
        target = jnp.where(targets[:, None] == 0,
                            jnp.array([1.0, -1.0]), jnp.array([-1.0, 1.0]))
        return jnp.mean((logits - target) ** 2)

    def result_fn(params, dataset):
        inp, tar = zip(*dataset)
        inp = jnp.array(inp)
        logits = apply_fun(params, inp)
        s = sum(1 for r, t in zip(logits, tar) if jnp.argmax(r) == t)
        return s / len(logits)

    key = jax.random.PRNGKey(SEED)
    trainer = TRAIN(key, init_fun, loss_fn, X_train_inner + X_val_inner,
                     X_test_outer, result_fn, save_name=checkpoint_name)
    trainer.N_STEPS = N_STEPS
    trainer.BATCH_SIZE = 50
    trainer.NUM_SEEDs = NUM_SEEDS
    trainer.STD_DEV = 0
    trainer.REC_INTE = 10
    trainer.VARI_FRE = 'epoch'
    trainer.ini_learning_rate = 0.005
    trainer.TRAIN_VALID_TEST = jnp.array([len(X_train_inner), len(X_val_inner)])
    trainer.ES_THRES = None
    trainer.ES_LEN = 10
    trainer.ES_MODE = 'loss'
    trainer.ES_DATASET = 'valid'

    t0 = time.time()
    trainer.train()
    elapsed = time.time() - t0

    test_acc = trainer.acc_results[0][1] if trainer.acc_results else None

    if verbose:
        print(f"[{config_tag}] test_acc={test_acc}  time={elapsed:.1f}s  "
              f"checkpoint='{checkpoint_name}_current.pkl'")

    file_exists = os.path.isfile(results_csv)
    with open(results_csv, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['hidden_dim', 'n_qubits', 'n_layers', 'n_params', 'test_acc',
                              'seconds', 'checkpoint_name', 'circuit_png'])
        writer.writerow([hidden_dim, n_qubits, n_layers, n_params, test_acc,
                          f"{elapsed:.1f}", checkpoint_name, circuit_png])

    return {
        'hidden_dim': hidden_dim, 'n_qubits': n_qubits, 'n_layers': n_layers,
        'n_params': n_params, 'test_acc': test_acc, 'elapsed_seconds': elapsed,
        'checkpoint_name': checkpoint_name, 'circuit_png': circuit_png,
    }


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('n_qubits', type=int)
    parser.add_argument('n_layers', type=int)
    parser.add_argument('--hidden_dim', type=int, default=DEFAULT_HIDDEN_DIM)
    parser.add_argument('--n_steps', type=int, default=500)
    parser.add_argument('--num_seeds', type=int, default=1)
    args = parser.parse_args()

    run_qgru_experiment(args.n_qubits, args.n_layers, hidden_dim=args.hidden_dim,
                         N_STEPS=args.n_steps, NUM_SEEDS=args.num_seeds)
