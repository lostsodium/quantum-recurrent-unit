#!/usr/bin/env python
# coding: utf-8
#
# QLSTM_ces_lit_j8.py
# ---------------------------------------------------------------------------
# Ceschini-reformulated QLSTM (per Ceschini et al., 2024's own complexity
# analysis of QLSTM, Sec 4.3 of the QGRU paper -- NOT Chen et al.'s original
# 2022 design, which lives in QLSTM_lit_j8.py). This version retains the
# classical LSTM gating structure but implements each gate as a VQC fed
# through SHARED classical FC_in/FC_out layers, exactly mirroring the design
# of QGRU_lit_j8.py -- the two differ only in gate count (4 vs 3) and the
# addition of a classical cell-state buffer:
#
#     r_t = sigmoid(FC_out(VQC_forget(FC_in([h_{t-1}, x_t]))))
#     i_t = sigmoid(FC_out(VQC_input(FC_in([h_{t-1}, x_t]))))
#     c~_t = tanh(FC_out(VQC_cell(FC_in([h_{t-1}, x_t]))))
#     c_t = f_t * c_{t-1} + i_t * c~_t
#     o_t = sigmoid(FC_out(VQC_output(FC_in([h_{t-1}, x_t]))))
#     h_t = o_t * tanh(c_t)
#
# Parameter count (Ceschini et al.'s own formula, n=qubits, l=layers,
# d_hid=hidden dim, d_in=input dim per step):
#     num_params = n*(4*l + 2*d_hid + d_in + 1) + d_hid
# (quantum params: 4*n*l; the rest is the shared FC_in/FC_out, exactly as
# in QGRU_lit_j8.py -- no separate readout VQC needed, since h_t is used
# directly as output when hidden_dim == readout_dim, matching the
# QRU-parity 'raw' readout convention used throughout this project.)
#
# Reuses QGRUGateCirc from QGRU_lit_j8.py directly -- same VQC building
# block, just wired into 4 gates instead of 3, with an added cell-state
# carry. Interface mirrors qgru_lit()'s init_fun/apply_fun so it plugs
# directly into the existing TRAIN class.
# ---------------------------------------------------------------------------

import pennylane as qml

from jax import lax
import jax
import jax.numpy as jnp

from QGRU_lit_j8 import QGRUGateCirc


class QLSTMCellCes:
    def __init__(self, input_dim, hidden_dim, n_qubits, n_layers, vari_gate='rx'):
        # requires hidden_dim == readout_dim (QRU-parity raw readout) --
        # enforced by the caller (qlstm_ces_lit), which fixes readout_dim
        # to equal hidden_dim.
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits
        self.dconc = hidden_dim + input_dim

        self.gate_circ = QGRUGateCirc(n_qubits, n_layers, vari_gate)
        self.n_vqc = self.gate_circ.num_weights  # per gate

        self.n_fcin_w = self.dconc * n_qubits
        self.n_fcin_b = n_qubits
        self.n_fcout_w = n_qubits * hidden_dim
        self.n_fcout_b = hidden_dim

        self.n_fcin = self.n_fcin_w + self.n_fcin_b
        self.n_fcout = self.n_fcout_w + self.n_fcout_b

        # total: n(4l + 2*d_hid + d_in + 1) + d_hid
        self.num_weights = self.n_fcin + self.n_fcout + 4 * self.n_vqc

        dev = qml.device("default.qubit", wires=n_qubits)
        self.qnode = qml.QNode(self.gate_circ, dev)

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
        w_forget = params[wi:wi + self.n_vqc]; wi += self.n_vqc
        w_input = params[wi:wi + self.n_vqc]; wi += self.n_vqc
        w_cell = params[wi:wi + self.n_vqc]; wi += self.n_vqc
        w_output = params[wi:wi + self.n_vqc]; wi += self.n_vqc
        return fcin_w, fcin_b, fcout_w, fcout_b, w_forget, w_input, w_cell, w_output

    def _gate(self, concat_vec, fcin_w, fcin_b, fcout_w, fcout_b, vqc_w):
        z = jnp.tanh(concat_vec @ fcin_w + fcin_b) * jnp.pi
        q_out = jnp.array(self.qnode(z, vqc_w))
        return q_out @ fcout_w + fcout_b

    def __call__(self, x_t, h_prev, c_prev, params):
        fcin_w, fcin_b, fcout_w, fcout_b, w_forget, w_input, w_cell, w_output = self._unpack(params)

        vt = jnp.concatenate([h_prev, x_t])

        f_t = jax.nn.sigmoid(self._gate(vt, fcin_w, fcin_b, fcout_w, fcout_b, w_forget))
        i_t = jax.nn.sigmoid(self._gate(vt, fcin_w, fcin_b, fcout_w, fcout_b, w_input))
        c_tilde = jnp.tanh(self._gate(vt, fcin_w, fcin_b, fcout_w, fcout_b, w_cell))
        c_t = f_t * c_prev + i_t * c_tilde
        o_t = jax.nn.sigmoid(self._gate(vt, fcin_w, fcin_b, fcout_w, fcout_b, w_output))

        h_t = o_t * jnp.tanh(c_t)
        return h_t, c_t


def qlstm_ces_lit(in_dim, hidden_dim, n_qubits, n_layers, vari_gate='rx',
                   readout_dim=None, out_fun=None, readout_mode='raw'):
    """
    Ceschini-reformulated QLSTM. Returns (init_fun, apply_fun), matching
    qgru_lit(). See qgru_lit()'s docstring in QGRU_lit_j8.py for the full
    description of each readout_mode -- the same options apply here:
      'raw' (default): requires hidden_dim == readout_dim (readout_dim
        defaults to hidden_dim if not given). No extra parameters.
      'slice': requires hidden_dim >= readout_dim. Output =
        h_final[:readout_dim]. No extra parameters -- mirrors QRU's own
        readout convention (hidden_dim > readout_dim, take a slice).
      'scalar': requires hidden_dim == readout_dim. Adds 1 parameter.
      'linear': full trainable Linear(hidden_dim -> readout_dim) + bias.
        Matches Ceschini et al.'s own paper convention ("all the networks
        had a final FC layer to compile the output of the recurrent
        layer").
    """
    if readout_dim is None:
        readout_dim = hidden_dim
    if readout_mode in ('raw', 'scalar') and hidden_dim != readout_dim:
        raise ValueError(
            f"readout_mode='{readout_mode}' requires hidden_dim == readout_dim "
            f"(got hidden_dim={hidden_dim}, readout_dim={readout_dim}).")
    if readout_mode == 'slice' and hidden_dim < readout_dim:
        raise ValueError(
            f"readout_mode='slice' requires hidden_dim >= readout_dim "
            f"(got hidden_dim={hidden_dim}, readout_dim={readout_dim}).")

    cell = QLSTMCellCes(in_dim, hidden_dim, n_qubits, n_layers, vari_gate)
    if readout_mode in ('raw', 'slice'):
        n_readout = 0
    elif readout_mode == 'scalar':
        n_readout = 1
    else:
        n_readout = hidden_dim * readout_dim + readout_dim

    def init_fun(rng, input_shape):
        rng1, rng2 = jax.random.split(rng)
        n = cell.num_weights
        cell_params = jax.random.uniform(rng1, (n,), minval=-jnp.pi, maxval=jnp.pi)
        if readout_mode in ('raw', 'slice'):
            readout_params = jnp.zeros((0,))
        elif readout_mode == 'scalar':
            readout_params = jnp.ones((1,))
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
            c0 = jnp.zeros(hidden_dim)

            def step(carry, x_t):
                h, c = carry
                h_new, c_new = cell(x_t, h, c, cell_params)
                return (h_new, c_new), h_new

            (h_final, c_final), _ = lax.scan(step, (h0, c0), x_seq)
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


def qlstm_ces_lit_param_count(in_dim, hidden_dim, n_qubits, n_layers, vari_gate='rx',
                               readout_dim=None, readout_mode='raw'):
    if readout_dim is None:
        readout_dim = hidden_dim
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
    return n_fcin + n_fcout + 4 * n_vqc + n_readout


if __name__ == '__main__':
    IN_DIM, HID_DIM, N_Q, N_L = 8, 2, 4, 1
    p = qlstm_ces_lit_param_count(IN_DIM, HID_DIM, N_Q, N_L)
    print("analytic param count:", p)

    init_fun, apply_fun = qlstm_ces_lit(IN_DIM, HID_DIM, N_Q, N_L)
    key = jax.random.PRNGKey(0)
    batch = 4
    dummy_X = jax.random.uniform(key, (batch, 8, IN_DIM))
    out_shape, params = init_fun(key, dummy_X.shape)
    print("params.shape[0]:", params.shape[0])
    assert params.shape[0] == p

    out = apply_fun(params, dummy_X)
    print("output shape:", out.shape)
    assert out.shape == (batch, HID_DIM)
    print("\nOK -- forward pass and parameter accounting both check out.")
