#!/usr/bin/env python
# coding: utf-8
#
# run_QLSTM_chen_MNIST.py
# ---------------------------------------------------------------------------
# QLSTM (Chen, Yang & Yin, 2022 design) MNIST experiment as a single
# callable function -- rebuilt to match the run_QGRU_MNIST.py /
# run_QLSTM_ces_MNIST.py pattern, using the CURRENT, literature-audit-fixed
# QLSTM_chen_lit_j8.py (entangle-then-rotate ansatz order). Any earlier
# result obtained before that fix (e.g. the hidden_dim=2/n_layers_gate=1 ->
# 53.07% result from the standalone train_MNIST_QLSTM_lit.py script) used
# the WRONG gate order and should not be treated as evidence about this
# architecture's trainability -- rerun with this script instead.
#
# Three free parameters: hidden_dim, n_layers_gate (depth of the 4
# full-scale, dconc-qubit gate VQCs), and n_layers_readout (depth of the 2
# smaller, hidden_dim-qubit readout VQCs -- defaults to n_layers_gate if
# omitted). There's no free n_qubits knob: gate-VQC qubit count is fixed by
# the architecture (n_qubits = hidden_dim + in_dim, one feature per qubit,
# no FC compression).
#
# Usage (interactively, e.g. in a notebook):
#   from run_QLSTM_chen_MNIST import run_qlstm_chen_experiment
#   run_qlstm_chen_experiment(hidden_dim=2, n_layers_gate=1)
#   run_qlstm_chen_experiment(hidden_dim=2, n_layers_gate=1, n_layers_readout=3)
#
# Usage (batch, from the command line):
#   python3 run_QLSTM_chen_MNIST.py 2 1 --n_steps 500
#   python3 run_QLSTM_chen_MNIST.py 2 1 --n_layers_readout 3 --n_steps 500
# ---------------------------------------------------------------------------

import csv
import os
import time

import jax
import jax.numpy as jnp
import pennylane as qml
from sklearn.model_selection import StratifiedKFold

from MNIST_3_5_dataset import MNIST_3_5_dataset
from QLSTM_chen_lit_j8 import qlstm_chen_lit, qlstm_chen_lit_param_count, ChenVQCCirc
from TRAIN_v4_debug_4 import TRAIN

IN_DIM, READOUT_DIM = 8, 2
SEED = 2
RESULTS_CSV = 'results_QLSTM_chen_MNIST.csv'
CIRCUIT_DIR = 'circuits'

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


def draw_circuit(hidden_dim, n_layers_gate, n_layers_readout, save_dir=CIRCUIT_DIR):
    """Draw one gate VQC (forget/input/candidate/output share this
    structure; the two readout VQCs are smaller, hidden_dim qubits, and
    may use a different depth -- see draw_readout_circuit)."""
    os.makedirs(save_dir, exist_ok=True)
    dconc = hidden_dim + IN_DIM
    circ = ChenVQCCirc(input_dim=dconc, n_layers=n_layers_gate, out_dim=hidden_dim)
    dev = qml.device('default.qubit', wires=circ.n_qubits)
    qnode = qml.QNode(circ, dev)

    dummy_features = jnp.zeros(dconc)
    dummy_weights = jnp.zeros(circ.num_weights)

    text_diagram = qml.draw(qnode)(dummy_features, dummy_weights)

    png_path = os.path.join(save_dir, f'QLSTM_chen_h{hidden_dim}_lg{n_layers_gate}.png')
    fig, ax = qml.draw_mpl(qnode)(dummy_features, dummy_weights)
    fig.savefig(png_path, dpi=100, bbox_inches='tight')
    import matplotlib.pyplot as plt
    plt.close(fig)

    return text_diagram, png_path


def draw_readout_circuit(hidden_dim, n_layers_readout, readout_dim=READOUT_DIM, save_dir=CIRCUIT_DIR):
    """Draw the final-output readout VQC (hidden_dim qubits)."""
    os.makedirs(save_dir, exist_ok=True)
    circ = ChenVQCCirc(input_dim=hidden_dim, n_layers=n_layers_readout, out_dim=readout_dim)
    dev = qml.device('default.qubit', wires=circ.n_qubits)
    qnode = qml.QNode(circ, dev)

    dummy_features = jnp.zeros(hidden_dim)
    dummy_weights = jnp.zeros(circ.num_weights)

    text_diagram = qml.draw(qnode)(dummy_features, dummy_weights)

    png_path = os.path.join(save_dir, f'QLSTM_chen_readout_h{hidden_dim}_lr{n_layers_readout}.png')
    fig, ax = qml.draw_mpl(qnode)(dummy_features, dummy_weights)
    fig.savefig(png_path, dpi=100, bbox_inches='tight')
    import matplotlib.pyplot as plt
    plt.close(fig)

    return text_diagram, png_path


def run_qlstm_chen_experiment(hidden_dim, n_layers_gate, n_layers_readout=None,
                          N_STEPS=500, NUM_SEEDS=1,
                          draw=True, verbose=True, results_csv=RESULTS_CSV):
    """Run one QLSTM (Chen et al. design) MNIST experiment for a given
    (hidden_dim, n_layers_gate, n_layers_readout), using the current
    literature-audit-fixed QLSTM_lit_j8.py (entangle-then-rotate ansatz
    order). n_layers_readout defaults to n_layers_gate if not given.

    n_layers_gate sets the depth of the 4 full-scale (dconc-qubit) gate
    VQCs; n_layers_readout sets the depth of the 2 smaller (hidden_dim-
    qubit) readout VQCs. These are exposed separately because the two VQC
    "shapes" differ substantially in qubit count and thus in simulation
    cost -- see QLSTM_lit_j8.py's module docstring.

    Returns a dict with n_params, test_acc, elapsed_seconds,
    checkpoint_name, circuit_png (if draw=True).

    Note: gate-VQC qubit count = hidden_dim + 8 (no FC compression), so
    this design is substantially more expensive to simulate than the
    Ceschini-style models for the same nominal "size" -- see
    QLSTM_lit_j8.py's module docstring for the full cost/parameter-growth
    discussion before choosing n_layers_gate > 1."""
    if n_layers_readout is None:
        n_layers_readout = n_layers_gate

    config_tag = f'h{hidden_dim}_lg{n_layers_gate}_lr{n_layers_readout}'
    checkpoint_name = f'QLSTM_chen_MNIST_{config_tag}'
    gate_qubits = hidden_dim + IN_DIM

    n_params = qlstm_chen_lit_param_count(IN_DIM, hidden_dim, READOUT_DIM,
                                      n_layers_gate, n_layers_readout)

    circuit_png = None
    if draw:
        text_diagram, circuit_png = draw_circuit(hidden_dim, n_layers_gate, n_layers_readout)
        if verbose:
            print(f"--- gate circuit ({config_tag}, gate_qubits={gate_qubits}) ---")
            print(text_diagram)
            print(f"(saved to {circuit_png})")
        _, readout_png = draw_readout_circuit(hidden_dim, n_layers_readout)
        if verbose:
            print(f"(readout circuit saved to {readout_png})")

    if verbose:
        print(f"[{config_tag}] parameter count: {n_params}  "
              f"(QRU comparison point: 131)")

    init_fun, apply_fun = qlstm_chen_lit(IN_DIM, hidden_dim, READOUT_DIM,
                                     n_layers_gate, n_layers_readout)

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
            writer.writerow(['hidden_dim', 'n_layers_gate', 'n_layers_readout', 'gate_qubits',
                              'n_params', 'test_acc', 'seconds', 'checkpoint_name', 'circuit_png'])
        writer.writerow([hidden_dim, n_layers_gate, n_layers_readout, gate_qubits, n_params,
                          test_acc, f"{elapsed:.1f}", checkpoint_name, circuit_png])

    return {
        'hidden_dim': hidden_dim, 'n_layers_gate': n_layers_gate,
        'n_layers_readout': n_layers_readout, 'gate_qubits': gate_qubits,
        'n_params': n_params, 'test_acc': test_acc, 'elapsed_seconds': elapsed,
        'checkpoint_name': checkpoint_name, 'circuit_png': circuit_png,
    }


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('hidden_dim', type=int)
    parser.add_argument('n_layers_gate', type=int)
    parser.add_argument('--n_layers_readout', type=int, default=None)
    parser.add_argument('--n_steps', type=int, default=500)
    parser.add_argument('--num_seeds', type=int, default=1)
    args = parser.parse_args()

    run_qlstm_chen_experiment(args.hidden_dim, args.n_layers_gate,
                          n_layers_readout=args.n_layers_readout,
                          N_STEPS=args.n_steps, NUM_SEEDS=args.num_seeds)
