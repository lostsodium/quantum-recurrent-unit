#!/usr/bin/env python
# coding: utf-8
#
# QGRU_lit_j8.py
# ---------------------------------------------------------------------------
# Literature-faithful Quantum GRU (Ceschini et al., 2024) implementation,
# built for the supplementary QLSTM/QGRU comparison requested by EPJ QT
# Reviewers R2 and R5.
#
# IMPORTANT: this is NOT the QRU architecture (that lives in SQGRU_j8.py,
# despite its confusingly similar legacy name -- SQGRU_j8.py is QRU's own
# C-SWAP-based circuit under an old naming convention).
#
# This module implements the *classical-gating* QGRU design: three VQCs
# (reset / update / candidate) act as feature extractors, sandwiched between
# two SHARED classical FC layers (FC_in, FC_out), exactly as described in
# Ceschini et al., "A variational approach to quantum gated recurrent units"
# (J. Phys. Commun. 8, 085004, 2024), Section 4.1 and Eq. (8) family.
#
# Parameter count follows their Eq. (given n=qubits, l=ansatz layers,
# d_hid=hidden dim, d_in=input dim per step):
#     num_params = n*(3*l + 2*d_hid + d_in + 1) + d_hid
#
# Interface mirrors qgru() in SQGRU_j8.py: exposes init_fun/apply_fun so it
# plugs directly into the existing TRAIN class without any changes there.
# ---------------------------------------------------------------------------

import pennylane as qml

from jax import lax
import jax
import jax.numpy as jnp

from functools import partial
import math

from lit_vqc_common import ansatz_layer, num_req_params


# ---------------------------------------------------------------------------
# Single VQC "gate" circuit: n_qubits angle-encoded features -> n_qubits
# PauliZ expectation values, after n_layers of a variational ansatz.
# Uses lit_vqc_common.py (decoupled from VQC_j8.py's fixed gate order --
# see that module's docstring) with explicit 'rotate_then_entangle' order
# and single-parameter Rx rotations, matching Ceschini et al.'s description
# exactly: "a layer of Rx gates for data encoding, an ansatz of parametrized
# Rx gates with circular CNOT entanglement" (Sec 4.1; encoding gate detailed
# in their Fig. 3, ansatz order in their Sec 3 / Fig. 4).
# ---------------------------------------------------------------------------

class QGRUGateCirc:
    def __init__(self, n_qubits, n_layers, vari_gate='rx'):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.vari_gate = vari_gate
        self.n_weights_per_layer = num_req_params(n_qubits, vari_gate)
        self.num_weights = n_layers * self.n_weights_per_layer

    def __call__(self, features, weights):
        # features: (n_qubits,) already-bounded values (e.g. via tanh), used
        # as RX encoding angles -- matches Ceschini et al.'s data encoding.
        for q in range(self.n_qubits):
            qml.RX(features[q], wires=q)
        qml.Barrier(wires=range(self.n_qubits))

        wi = 0
        for _ in range(self.n_layers):
            wj = wi + self.n_weights_per_layer
            ansatz_layer(weights[wi:wj], list(range(self.n_qubits)),
                         gate=self.vari_gate, order='rotate_then_entangle')
            wi = wj

        return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]


# ---------------------------------------------------------------------------
# QGRU cell: classical FC_in/FC_out (shared across the 3 gates, per
# Ceschini et al.) wrapping three independent VQCs (reset/update/candidate).
# ---------------------------------------------------------------------------

class QGRUCellLit:
    def __init__(self, input_dim, hidden_dim, n_qubits, n_layers, vari_gate='rx'):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.dconc = hidden_dim + input_dim

        self.gate_circ = QGRUGateCirc(n_qubits, n_layers, vari_gate)
        self.n_vqc = self.gate_circ.num_weights  # per gate

        # shared classical layers (weight + bias)
        self.n_fcin_w = self.dconc * n_qubits
        self.n_fcin_b = n_qubits
        self.n_fcout_w = n_qubits * hidden_dim
        self.n_fcout_b = hidden_dim

        self.n_fcin = self.n_fcin_w + self.n_fcin_b
        self.n_fcout = self.n_fcout_w + self.n_fcout_b

        # total trainable parameters: n(3l + 2*d_hid + d_in + 1) + d_hid
        self.num_weights = self.n_fcin + self.n_fcout + 3 * self.n_vqc

        dev = qml.device("default.qubit", wires=n_qubits)
        self.qnode = qml.QNode(self.gate_circ, dev)

    # ---- parameter unpacking -------------------------------------------------
    def _unpack(self, params):
        wi = 0
        fcin_w = params[wi:wi + self.n_fcin_w].reshape(self.dconc, self.n_qubits)
        wi += self.n_fcin_w
        fcin_b = params[wi:wi + self.n_fcin_b]
        wi += self.n_fcin_b
        fcout_w = params[wi:wi + self.n_fcout_w].reshape(self.n_qubits, self.hidden_dim)
        wi += self.n_fcout_w
        fcout_b = params[wi:wi + self.n_fcout_b]
        wi += self.n_fcout_b
        w_reset = params[wi:wi + self.n_vqc]
        wi += self.n_vqc
        w_update = params[wi:wi + self.n_vqc]
        wi += self.n_vqc
        w_cand = params[wi:wi + self.n_vqc]
        wi += self.n_vqc
        return fcin_w, fcin_b, fcout_w, fcout_b, w_reset, w_update, w_cand

    def _gate(self, concat_vec, fcin_w, fcin_b, fcout_w, fcout_b, vqc_w):
        z = jnp.tanh(concat_vec @ fcin_w + fcin_b) * jnp.pi  # bound encoding angles to [-pi, pi]
        q_out = self.qnode(z, vqc_w)
        q_out = jnp.array(q_out)
        return q_out @ fcout_w + fcout_b

    def __call__(self, x_t, h_prev, params):
        fcin_w, fcin_b, fcout_w, fcout_b, w_reset, w_update, w_cand = self._unpack(params)

        concat1 = jnp.concatenate([h_prev, x_t])
        r = jax.nn.sigmoid(self._gate(concat1, fcin_w, fcin_b, fcout_w, fcout_b, w_reset))
        z = jax.nn.sigmoid(self._gate(concat1, fcin_w, fcin_b, fcout_w, fcout_b, w_update))

        concat2 = jnp.concatenate([r * h_prev, x_t])
        h_tilde = jnp.tanh(self._gate(concat2, fcin_w, fcin_b, fcout_w, fcout_b, w_cand))

        h_new = (1 - z) * h_prev + z * h_tilde
        return h_new


# ---------------------------------------------------------------------------
# Sequence-level wrapper, matching qgru()'s init_fun/apply_fun signature so
# it drops directly into the existing TRAIN class.
# ---------------------------------------------------------------------------

def qgru_lit(in_dim, hidden_dim, n_qubits, n_layers, vari_gate='rx',
             readout_dim=1, out_fun=None, readout_mode='raw'):
    """
    Literature-faithful QGRU, for use as a supplementary comparison baseline
    against QRU (Ceschini et al., 2024 design).

    readout_mode:
      'raw' (default): requires hidden_dim == readout_dim. Output = h_final
        directly, no extra trainable parameters. Matches QRU's Section 3.5
        (Noise Injection) protocol exactly: no scaling parameter, raw
        bounded circuit/hidden-state output used directly, paired with MSE
        loss against {-1,+1}-mapped labels.
      'slice': requires hidden_dim >= readout_dim. Output = h_final[:readout_dim]
        -- the first readout_dim components of the hidden state, no extra
        trainable parameters. This mirrors QRU's OWN readout convention
        more closely than 'raw' does: QRU's hidden state is 8-dimensional
        (4 qubits x 2 bases) for MNIST, but only the first two qubits'
        expectation values are used as output -- hidden_dim > readout_dim
        there too. Use this when you want hidden_dim free to be larger
        than readout_dim (e.g. to address a "hidden dimension too small"
        concern) while still adding zero extra readout parameters.
      'scalar': requires hidden_dim == readout_dim. Output = h_final * s,
        a single trainable shared scalar, no mixing across output
        dimensions, no bias. Matches QRU's Section 3.4.1 (main text)
        protocol: raw bounded qubit expectation values scaled by one
        shared trainable parameter s, before softmax/cross-entropy.
        Adds exactly 1 parameter.
      'linear': a full trainable Linear(hidden_dim -> readout_dim) + bias.
        Matches Ceschini et al.'s OWN paper convention directly: "all the
        networks had a final FC layer to compile the output of the
        recurrent layer to a suitable output dimension" (Sec 5). Frees
        hidden_dim completely, at the cost of hidden_dim*readout_dim+readout_dim
        extra parameters -- more expressive than QRU's own readout.

    Returns (init_fun, apply_fun), matching qgru() in SQGRU_j8.py.
    """
    if readout_mode in ('raw', 'scalar') and hidden_dim != readout_dim:
        raise ValueError(
            f"readout_mode='{readout_mode}' requires hidden_dim == readout_dim "
            f"(got hidden_dim={hidden_dim}, readout_dim={readout_dim}).")
    if readout_mode == 'slice' and hidden_dim < readout_dim:
        raise ValueError(
            f"readout_mode='slice' requires hidden_dim >= readout_dim "
            f"(got hidden_dim={hidden_dim}, readout_dim={readout_dim}).")

    cell = QGRUCellLit(in_dim, hidden_dim, n_qubits, n_layers, vari_gate)
    if readout_mode in ('raw', 'slice'):
        n_readout = 0
    elif readout_mode == 'scalar':
        n_readout = 1
    else:
        n_readout = hidden_dim * readout_dim + readout_dim  # weight + bias

    def init_fun(rng, input_shape):
        # input_shape: (batch, seq_len, in_dim)
        rng1, rng2 = jax.random.split(rng)
        n_cell = cell.num_weights
        cell_params = jax.random.uniform(rng1, (n_cell,), minval=-jnp.pi, maxval=jnp.pi)
        if readout_mode in ('raw', 'slice'):
            readout_params = jnp.zeros((0,))
        elif readout_mode == 'scalar':
            readout_params = jnp.ones((1,))  # s initialized to 1.0, like an identity scaling
        else:
            readout_params = jax.random.normal(rng2, (n_readout,)) * 0.1
        params = jnp.concatenate([cell_params, readout_params])

        output_shape = (input_shape[0], readout_dim)
        if out_fun is not None:
            out_temp = jnp.ones(output_shape)
            out_temp = out_fun(out_temp)
            output_shape = out_temp.shape

        return output_shape, params

    def apply_fun(params, Xs, **kwargs):
        n_cell = cell.num_weights
        cell_params = params[:n_cell]
        readout_params = params[n_cell:]

        if readout_mode == 'scalar':
            s = readout_params[0]
        elif readout_mode == 'linear':
            ro_w = readout_params[:hidden_dim * readout_dim].reshape(hidden_dim, readout_dim)
            ro_b = readout_params[hidden_dim * readout_dim:]

        def run_one(x_seq):
            h0 = jnp.zeros(hidden_dim)

            def step(h, x_t):
                h_new = cell(x_t, h, cell_params)
                return h_new, h_new

            h_final, _ = lax.scan(step, h0, x_seq)
            if readout_mode == 'raw':
                out = h_final
            elif readout_mode == 'slice':
                out = h_final[:readout_dim]
            elif readout_mode == 'scalar':
                out = h_final * s
            else:
                out = h_final @ ro_w + ro_b
            if out_fun is not None:
                out = out_fun(out[None, :])[0]
            return out

        return jax.vmap(run_one)(Xs)

    return init_fun, apply_fun


# ---------------------------------------------------------------------------
# Convenience: report parameter count without instantiating a circuit,
# for quickly scanning (n_qubits, n_layers) combinations to approximate a
# target parameter count (e.g. QRU's 35 params on WDBC) analytically,
# instead of running a full parameter sweep of trainings.
# ---------------------------------------------------------------------------

def qgru_lit_param_count(in_dim, hidden_dim, n_qubits, n_layers, readout_dim=1,
                          vari_gate='rx', readout_mode='raw'):
    from lit_vqc_common import num_req_params
    per_layer = num_req_params(n_qubits, vari_gate)
    n_vqc = n_layers * per_layer
    dconc = hidden_dim + in_dim
    n_fcin = dconc * n_qubits + n_qubits
    n_fcout = n_qubits * hidden_dim + hidden_dim
    if readout_mode in ('raw', 'slice'):
        n_readout = 0
    elif readout_mode == 'scalar':
        n_readout = 1
    else:
        n_readout = hidden_dim * readout_dim + readout_dim
    return n_fcin + n_fcout + 3 * n_vqc + n_readout


if __name__ == '__main__':
    # quick self-test: forward pass + parameter-count sanity check
    IN_DIM, HID_DIM, N_Q, N_L = 1, 1, 4, 2
    init_fun, apply_fun = qgru_lit(IN_DIM, HID_DIM, N_Q, N_L, readout_dim=1)

    key = jax.random.PRNGKey(0)
    seq_len = 30  # matches WDBC's 30-feature sequential encoding
    batch = 3
    dummy_X = jax.random.uniform(key, (batch, seq_len, IN_DIM))
    out_shape, params = init_fun(key, dummy_X.shape)

    print("output_shape:", out_shape)
    print("total num params:", params.shape[0])

    analytic = qgru_lit_param_count(IN_DIM, HID_DIM, N_Q, N_L, readout_dim=1)
    print("analytic count:", analytic)
    assert params.shape[0] == analytic, "parameter count mismatch!"

    out = apply_fun(params, dummy_X)
    print("forward pass output shape:", out.shape)
    assert out.shape == (batch, 1)

    print("\nOK -- forward pass and parameter accounting both check out.")
