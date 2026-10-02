#!/usr/bin/env python
# coding: utf-8
#
# QLSTM_lit_j8.py
# ---------------------------------------------------------------------------
# Literature-faithful QLSTM (Chen, Yang & Yin, 2022, "Quantum Long
# Short-Term Memory", ICASSP), for the supplementary QLSTM/QGRU comparison
# requested by EPJ QT Reviewers R2 and R5.
#
# This is a DIFFERENT design philosophy from QGRU_lit_j8.py (Ceschini et al.
# reformulation): here each VQC directly encodes its input vector one
# feature per qubit (H -> Ry(arctan(x)) -> Rz(arctan(x^2)), following the
# paper's own Fig. 2), with NO shared classical FC_in/FC_out layers. There
# are 6 independent VQCs (forget/input/candidate/output gates + a hidden
# readout + an output readout), following the paper's Eq. (1a)-(1g):
#
#     ft = sigmoid(VQC1(vt))
#     it = sigmoid(VQC2(vt))
#     C~t = tanh(VQC3(vt))
#     ct = ft*ct-1 + it*C~t
#     ot = sigmoid(VQC4(vt))
#     ht = VQC5(ot*tanh(ct))
#     yt = VQC6(ot*tanh(ct))
#
# where vt = concat([xt, ht-1]).
#
# Parameter count (uniform n_qubits per VQC type, uniform depth l,
# 3-parameter rotation gate 'rot'):
#     gate VQCs (x4):    n_qubits = dconc = hidden_dim + in_dim (direct
#                         1-feature-per-qubit encoding of vt)
#     readout VQCs (x2): n_qubits = hidden_dim (direct encoding of
#                         ot*tanh(ct))
#     total = 6 * l * 3 * qubits_used, i.e. no classical parameters at all
#     -- this is the key structural contrast with QGRU_lit_j8.py's large
#     FC_in/FC_out matrices.
#
# Interface mirrors qgru_lit() in QGRU_lit_j8.py: exposes init_fun/apply_fun
# so it plugs directly into the existing TRAIN class.
# ---------------------------------------------------------------------------

import pennylane as qml

from jax import lax
import jax
import jax.numpy as jnp

from functools import partial
import math

from lit_vqc_common import ansatz_layer, num_req_params


# ---------------------------------------------------------------------------
# Direct-encoding VQC: one feature per qubit (H, Ry(arctan(x)), Rz(arctan(x^2))
# -- fixed, non-trainable encoding, exactly Fig. 2 of Chen et al., 2022).
# Reverted from the 2-features-per-qubit variant: that compression left
# "blank" (unencoded) qubits in the readout VQCs whenever out_dim exceeded
# the compressed encoding qubit count, which is a structural side effect of
# packing 2 features per qubit, not a deliberate ancilla design choice. The
# original 1-feature-per-qubit encoding avoids this for vqc_hidden_out
# (whose input_dim always equals its out_dim = hidden_dim), at the cost of
# a much larger qubit count for the gate VQCs (n_qubits = dconc directly,
# no compression) -- see the module-level cost discussion below.
#
# Ansatz order: 'entangle_then_rotate', via lit_vqc_common.py (decoupled
# from VQC_j8.py's temp_Variation, which is hardcoded rotate-then-entangle
# and does NOT match this paper). Chen et al.'s circuit diagram (Fig. 2)
# shows entangling control dots BEFORE the R(alpha,beta,gamma) rotation on
# each wire -- i.e. entangle first, then rotate -- the opposite order from
# Ceschini et al.'s design (used in QGRU_lit_j8.py / QLSTM_ces_lit_j8.py).
# ---------------------------------------------------------------------------

class ChenVQCCirc:
    def __init__(self, input_dim, n_layers, out_dim, vari_gate='rot'):
        self.n_qubits = max(input_dim, out_dim)
        self.input_dim = input_dim
        self.n_layers = n_layers
        self.out_dim = out_dim
        self.vari_gate = vari_gate
        self.n_weights_per_layer = num_req_params(self.n_qubits, vari_gate)
        self.num_weights = n_layers * self.n_weights_per_layer

    def __call__(self, features, weights):
        # features: (input_dim,) raw classical values; one feature per qubit.
        for q in range(self.input_dim):
            qml.Hadamard(wires=q)
            qml.RY(jnp.arctan(features[q]), wires=q)
            qml.RZ(jnp.arctan(features[q] ** 2), wires=q)
        qml.Barrier(wires=range(self.n_qubits))

        wi = 0
        for _ in range(self.n_layers):
            wj = wi + self.n_weights_per_layer
            ansatz_layer(weights[wi:wj], list(range(self.n_qubits)),
                         gate=self.vari_gate, order='entangle_then_rotate')
            wi = wj

        return [qml.expval(qml.PauliZ(i)) for i in range(self.out_dim)]


class ChenVQC:
    """Wraps ChenVQCCirc in a QNode, callable as vqc(features, weights) -> jnp.array."""
    def __init__(self, input_dim, n_layers, out_dim, vari_gate='rot'):
        self.circ = ChenVQCCirc(input_dim, n_layers, out_dim, vari_gate)
        self.num_weights = self.circ.num_weights
        dev = qml.device("default.qubit", wires=self.circ.n_qubits)
        self.qnode = qml.QNode(self.circ, dev)

    def __call__(self, features, weights):
        return jnp.array(self.qnode(features, weights))


# ---------------------------------------------------------------------------
# QLSTM cell: 4 gate VQCs (forget/input/candidate/output) + 2 readout VQCs
# (hidden, output), no classical FC layers anywhere.
# ---------------------------------------------------------------------------

class QLSTMCellChen:
    def __init__(self, input_dim, hidden_dim, readout_dim, n_layers_gate,
                 n_layers_readout=None, vari_gate='rot'):
        """
        n_layers_gate: ansatz depth for the 4 (full-scale, dconc-qubit) gate
          VQCs (forget/input/candidate/output).
        n_layers_readout: ansatz depth for the 2 (smaller, hidden_dim-qubit)
          readout VQCs (hidden-state readout, final output). Defaults to
          n_layers_gate if not given, for backward compatibility. Exposed
          separately because the two VQC "shapes" differ substantially in
          qubit count -- a single shared n_layers unnecessarily couples
          their depths.
        """
        if n_layers_readout is None:
            n_layers_readout = n_layers_gate

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.readout_dim = readout_dim
        self.dconc = hidden_dim + input_dim
        self.n_layers_gate = n_layers_gate
        self.n_layers_readout = n_layers_readout

        # 4 gate VQCs: encode vt (dconc features, 2-per-qubit), output hidden_dim values
        self.vqc_forget = ChenVQC(self.dconc, n_layers_gate, hidden_dim, vari_gate)
        self.vqc_input = ChenVQC(self.dconc, n_layers_gate, hidden_dim, vari_gate)
        self.vqc_cell = ChenVQC(self.dconc, n_layers_gate, hidden_dim, vari_gate)
        self.vqc_output = ChenVQC(self.dconc, n_layers_gate, hidden_dim, vari_gate)

        # 2 readout VQCs: encode ot*tanh(ct) (hidden_dim features, 2-per-qubit)
        self.vqc_hidden_out = ChenVQC(hidden_dim, n_layers_readout, hidden_dim, vari_gate)
        self.vqc_final_out = ChenVQC(hidden_dim, n_layers_readout, readout_dim, vari_gate)

        self.n_gate = self.vqc_forget.num_weights          # same for all 4 gate VQCs
        self.n_hidden_out = self.vqc_hidden_out.num_weights
        self.n_final_out = self.vqc_final_out.num_weights

        self.num_weights = 4 * self.n_gate + self.n_hidden_out + self.n_final_out

    def _unpack(self, params):
        wi = 0
        w_forget = params[wi:wi + self.n_gate]; wi += self.n_gate
        w_input = params[wi:wi + self.n_gate]; wi += self.n_gate
        w_cell = params[wi:wi + self.n_gate]; wi += self.n_gate
        w_output = params[wi:wi + self.n_gate]; wi += self.n_gate
        w_hidden_out = params[wi:wi + self.n_hidden_out]; wi += self.n_hidden_out
        w_final_out = params[wi:wi + self.n_final_out]; wi += self.n_final_out
        return w_forget, w_input, w_cell, w_output, w_hidden_out, w_final_out

    def __call__(self, x_t, h_prev, c_prev, params):
        w_forget, w_input, w_cell, w_output, w_hidden_out, w_final_out = self._unpack(params)

        vt = jnp.concatenate([x_t, h_prev])

        f_t = jax.nn.sigmoid(self.vqc_forget(vt, w_forget))
        i_t = jax.nn.sigmoid(self.vqc_input(vt, w_input))
        c_tilde = jnp.tanh(self.vqc_cell(vt, w_cell))
        c_t = f_t * c_prev + i_t * c_tilde
        o_t = jax.nn.sigmoid(self.vqc_output(vt, w_output))

        oh = o_t * jnp.tanh(c_t)
        h_t = self.vqc_hidden_out(oh, w_hidden_out)
        y_t = self.vqc_final_out(oh, w_final_out)

        return h_t, c_t, y_t


# ---------------------------------------------------------------------------
# Sequence-level wrapper, matching qgru_lit()'s init_fun/apply_fun signature.
# ---------------------------------------------------------------------------

def qlstm_chen_lit(in_dim, hidden_dim, readout_dim, n_layers_gate,
              n_layers_readout=None, vari_gate='rot', out_fun=None):
    """
    Literature-faithful QLSTM (Chen et al., 2022 design). Returns
    (init_fun, apply_fun), matching qgru_lit() in QGRU_lit_j8.py, so it
    plugs directly into the existing TRAIN class.

    n_layers_gate / n_layers_readout: see QLSTMCellChen's docstring --
    depths for the 4 full-scale gate VQCs and the 2 smaller readout VQCs
    can be set independently. n_layers_readout defaults to n_layers_gate.
    """
    cell = QLSTMCellChen(in_dim, hidden_dim, readout_dim, n_layers_gate,
                         n_layers_readout, vari_gate)

    def init_fun(rng, input_shape):
        # input_shape: (batch, seq_len, in_dim)
        n = cell.num_weights
        params = jax.random.uniform(rng, (n,), minval=-jnp.pi, maxval=jnp.pi)

        output_shape = (input_shape[0], readout_dim)
        if out_fun is not None:
            out_temp = jnp.ones(output_shape)
            out_temp = out_fun(out_temp)
            output_shape = out_temp.shape

        return output_shape, params

    def apply_fun(params, Xs, **kwargs):
        def run_one(x_seq):
            h0 = jnp.zeros(hidden_dim)
            c0 = jnp.zeros(hidden_dim)

            def step(carry, x_t):
                h, c = carry
                h_new, c_new, y = cell(x_t, h, c, params)
                return (h_new, c_new), y

            (h_final, c_final), ys = lax.scan(step, (h0, c0), x_seq)
            out = ys[-1]  # final-step output, matching QRU/QGRU_lit convention
            if out_fun is not None:
                out = out_fun(out[None, :])[0]
            return out

        return jax.vmap(run_one)(Xs)

    return init_fun, apply_fun


# ---------------------------------------------------------------------------
# Convenience: report parameter count without instantiating a circuit.
# ---------------------------------------------------------------------------

def qlstm_chen_lit_param_count(in_dim, hidden_dim, readout_dim, n_layers_gate,
                           n_layers_readout=None, vari_gate='rot'):
    if n_layers_readout is None:
        n_layers_readout = n_layers_gate

    dconc = hidden_dim + in_dim
    n_q_gate = max(dconc, hidden_dim)
    n_q_hidden_out = max(hidden_dim, hidden_dim)
    n_q_final_out = max(hidden_dim, readout_dim)

    per_layer_gate = num_req_params(n_q_gate, vari_gate)
    per_layer_hidden_out = num_req_params(n_q_hidden_out, vari_gate)
    per_layer_final_out = num_req_params(n_q_final_out, vari_gate)

    n_gate = n_layers_gate * per_layer_gate
    n_hidden_out = n_layers_readout * per_layer_hidden_out
    n_final_out = n_layers_readout * per_layer_final_out
    return 4 * n_gate + n_hidden_out + n_final_out


if __name__ == '__main__':
    # self-test: forward pass + parameter-count sanity check
    IN_DIM, HID_DIM, READOUT_DIM, N_L = 1, 2, 1, 2
    init_fun, apply_fun = qlstm_chen_lit(IN_DIM, HID_DIM, READOUT_DIM, N_L)

    key = jax.random.PRNGKey(0)
    seq_len = 30
    batch = 3
    dummy_X = jax.random.uniform(key, (batch, seq_len, IN_DIM))
    out_shape, params = init_fun(key, dummy_X.shape)

    print("output_shape:", out_shape)
    print("total num params:", params.shape[0])

    analytic = qlstm_chen_lit_param_count(IN_DIM, HID_DIM, READOUT_DIM, N_L)
    print("analytic count:", analytic)
    assert params.shape[0] == analytic, "parameter count mismatch!"

    out = apply_fun(params, dummy_X)
    print("forward pass output shape:", out.shape)
    assert out.shape == (batch, READOUT_DIM)

    print("\nOK -- forward pass and parameter accounting both check out.")
