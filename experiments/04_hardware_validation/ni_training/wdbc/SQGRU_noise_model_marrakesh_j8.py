#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pennylane as qml

from jax import lax

import jax
import jax.numpy as jnp

from functools import partial
import math

from VQC_j8 import temp_Variation, temp_VQC


import time


# In[2]:


class QGRUCirc:
    def __init__(self, input_dim, hidden_dim, number_of_layers=2,
                 enc_layers=0, enc_reupload=1, hid_layers=0, hid_reupload=1, pOut_dim=0,
                 enc_n_weights_each=2, hid_n_weights_each=2, enc_v_gate=None, enc_lay_v_gate='rot', 
                 hid_v_gate=None, hid_lay_v_gate='rot', gate_gate='rot', lay_gate='rot',
                 n_out_lay=None, out_gate='rot', n_hout_lay=None, hout_gate='rot'):
        # pOut_dim: >0: dimension of probability output
        super(QGRUCirc, self).__init__()
        
        if pOut_dim > 0:
            self.pOut_wires = math.ceil(jnp.log2(pOut_dim))
            self.pOut_dim = pOut_dim
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.qN_i = math.ceil(input_dim/2) # number of input qubits
        self.qN_h = math.ceil(hidden_dim/2) # number of input qubits
        self.qN_e = self.qN_i + self.qN_h
        
        self.qb_i = list(range(self.qN_i))
        self.qb_h = list(range(self.qN_i, self.qN_e))
        self.qb_e = list(range(self.qN_e))
        self.qb_a = list(range(self.qN_e, self.qN_e+2))
        
        self.num_lay = number_of_layers
        
        self.enc_v_gate = enc_v_gate
        self.enc_lay_v_gate = enc_lay_v_gate
        self.hid_v_gate = hid_v_gate
        self.hid_lay_v_gate = hid_lay_v_gate
        self.gate_gate = gate_gate
        self.n_out_lay = n_out_lay
        self.n_hout_lay = n_hout_lay
        if type(lay_gate) != list:
            self.lay_gate = [lay_gate]
        else:
            self.lay_gate = lay_gate
            
        if type(out_gate) != list:
            self.out_gate = [out_gate]
        else:
            self.out_gate = out_gate
            
        if type(hout_gate) != list:
            self.hout_gate = [hout_gate]
        else:
            self.hout_gate = hout_gate
        
        self.enc_n_weights_each = enc_n_weights_each
        self.hid_n_weights_each = hid_n_weights_each
        
        # encoding part
        self.enc_layers = enc_layers
        self.enc_reupload = enc_reupload
        self.n_enc_weights = temp_VQC.num_req_params(self.input_dim, self.qN_i, self.enc_layers, self.enc_reupload,
                                                     self.enc_n_weights_each, self.enc_v_gate, self.enc_lay_v_gate)
        
        # hidden-in part
        self.hid_layers = hid_layers
        self.hid_reupload = hid_reupload
        self.n_hid_weights = temp_VQC.num_req_params(self.hidden_dim, self.qN_h, self.hid_layers, self.hid_reupload,
                                                     self.hid_n_weights_each, self.hid_v_gate, self.hid_lay_v_gate)
    
        # layer part
        self.n_lay_weights = []
        lay_n_w = 0
        vn = len(self.lay_gate)
        for i in range(self.num_lay):
            n_weights = temp_Variation.num_req_params(self.qN_e, self.lay_gate[i%vn])
            self.n_lay_weights.append(n_weights)
            lay_n_w += n_weights
        
        # gate part
        ['rot', 'rx', 'ry', 'rz', 'u1', 'u2', 'u3'].index(gate_gate)
        if gate_gate == 'rot' or gate_gate == 'u3':
            self.gate_n_weights = 3
        elif gate_gate == 'u2':
            self.gate_n_weights = 2
        else:
            self.gate_n_weights = 1
            
        # data-out part
        if self.n_out_lay == None:
            self.n_out_weights = 0
        else:
            self.n_out_weights = temp_VQC.num_req_params(0, self.qN_i, self.n_out_lay, 0,
                                                         0, None, self.out_gate)
        
        # hidden-out part
        if self.n_hout_lay == None:
            self.n_hout_weights = 0
        else:
            self.n_hout_weights = temp_VQC.num_req_params(0, self.qN_h, self.n_hout_lay, 0,
                                                          0, None, self.hout_gate)
        
        # number of weights
        self.num_weights = self.n_enc_weights + self.n_hid_weights + lay_n_w + self.gate_n_weights +\
        self.n_out_weights + self.n_hout_weights
        
    def __call__(self, inputs, weights, probs_out=False):
        # inputs: includes type, inputs and hiddens
        # probs_out: True -> probability output; False -> expected value output
                
        # data in (input)
        temp_VQC(weights=weights[0:self.n_enc_weights], wires=self.qb_i, inputs=inputs[0:self.input_dim],
                 n_layers=self.enc_layers, n_reupload=self.enc_reupload, num_weights_each=self.enc_n_weights_each,
                 enc_vari_gate=self.enc_v_gate, lay_vari_gate=self.enc_lay_v_gate, id='Data in')
        
        wi = self.n_enc_weights
            
        # initialize hidden state
        hiddens = inputs[self.input_dim:self.input_dim+self.hidden_dim]
        hiddens = jnp.arcsin(hiddens)
        
        # Since the output of SQGRU was modified, following process becomes not necessary.
#         # [1,2,3,4,5,6,7,8] -> [1,5,2,6,3,7,4,8]
#         hiddens = hiddens.reshape((2,-1)).T
#         hiddens = hiddens.reshape((1,-1))
#         hiddens = hiddens.squeeze()
        # --------------------------------------
        
        # hidden in
        temp_VQC(weights=weights[wi:wi+self.n_hid_weights], wires=self.qb_h, inputs=hiddens,
                 n_layers=self.hid_layers, n_reupload=self.hid_reupload, num_weights_each=self.hid_n_weights_each,
                 enc_vari_gate=self.hid_v_gate, lay_vari_gate=self.hid_lay_v_gate, id='Hidden in')
        
        wi += self.n_hid_weights
        qml.Barrier(wires=range(self.qN_e))
            
        # gates
        qml.CNOT(wires=[self.qb_i[0], self.qN_i])
        qml.CSWAP(wires=[self.qb_h[0], self.qb_h[1], self.qb_h[-1]+1])
        
        if self.gate_gate == 'rot':
            qml.Rot(weights[wi], weights[wi+1], weights[wi+2], wires=self.qb_h[0])
        elif self.gate_gate == 'rx':
            qml.RX(weights[wi], wires=self.qb_h[0])
        elif self.gate_gate == 'ry':
            qml.RY(weights[wi], wires=self.qb_h[0])
        elif self.gate_gate == 'rz':
            qml.RZ(weights[wi], wires=self.qb_h[0])
        elif self.gate_gate == 'u1':
            qml.U1(weights[wi], wires=self.qb_h[0])
        elif self.gate_gate == 'u2':
            qml.U2(weights[wi], weights[wi+1], wires=self.qb_h[0])
        elif self.gate_gate == 'u3':
            qml.U3(weights[wi], weights[wi+1], weights[wi+2], wires=self.qb_h[0])
            
        wi += self.gate_n_weights
        qml.CSWAP(wires=[self.qb_h[0], self.qb_h[-1], self.qb_h[-1]+2])
        qml.Barrier(wires=range(self.qN_e))
        
        # layers
        vn = len(self.lay_gate)
        for i in range(self.num_lay):
            n_weights = self.n_lay_weights[i]
            temp_Variation(weights=weights[wi:wi+n_weights],
                           wires=self.qb_e, gate=self.lay_gate[i%vn], id='Variation')
            wi += n_weights
            
        # data out (output)
        if self.n_out_lay != None:
            temp_VQC(weights=weights[wi:wi+self.n_out_weights], wires=self.qb_i, inputs=None,
                     n_layers=self.n_out_lay, lay_vari_gate=self.out_gate, id='Data out')
            wi += self.n_out_weights
        
        # hidden out
        if self.n_hout_lay != None:
            temp_VQC(weights=weights[wi:wi+self.n_hout_weights], wires=self.qb_h, inputs=None,
                     n_layers=self.n_hout_lay, lay_vari_gate=self.hout_gate, id='Hidden out')
            wi += self.n_hout_weights
            
        if probs_out:
            return qml.probs(wires=range(self.pOut_wires))
        else:
            measurements = [qml.expval(qml.PauliZ(i)) for i in range(self.qN_e)] + [qml.expval(qml.PauliX(i)) for i in range(self.qN_e)]
            return measurements
        


# In[6]:


class SQGRU:
    def __init__(self, input_dim, hidden_dim, number_of_layers=2,
                 enc_layers=0, enc_reupload=1, hid_layers=0, hid_reupload=1, pOut_dim=0,
                 enc_n_weights_each=2, hid_n_weights_each=2, enc_v_gate=None, enc_lay_v_gate='rot',
                 hid_v_gate=None, hid_lay_v_gate='rot', gate_gate='rot', lay_gate='rot',
                 n_out_lay=None, out_gate='rot', n_hout_lay=None, hout_gate='rot'):
        # pOut_dim: >0: dimension of probability output
        super(SQGRU, self).__init__()
        
        num_qb_i = math.ceil(input_dim/2)
        num_qb_h = math.ceil(hidden_dim/2)
        num_qb = num_qb_i+num_qb_h+2
        
        self.num_qb_i = num_qb_i
        self.hidden_dim = num_qb_h*2
        self.in_dim = input_dim
        self.pOut_dim = pOut_dim
        self.out_dim = num_qb_i*2
        
        dev = qml.device("default.qubit", wires=num_qb)
        circuit = QGRUCirc(input_dim, hidden_dim, number_of_layers,
                           enc_layers, enc_reupload, hid_layers, hid_reupload, pOut_dim,
                           enc_n_weights_each, hid_n_weights_each, enc_v_gate, enc_lay_v_gate,
                           hid_v_gate, hid_lay_v_gate, gate_gate, lay_gate,
                           n_out_lay, out_gate, n_hout_lay, hout_gate)
        self.num_weights = circuit.num_weights
        self.qnode = qml.QNode(circuit, dev)    

    def __subCell__(self, X, hidden, params, key=None):
        inp = jnp.concatenate((X, hidden), axis=0)
        out = jnp.array(self.qnode(inp, params)).reshape([2,-1]).T
    
        # add noise
        # alpha_matrix: OTC calibrated (noise model, marrakesh)
        alpha_matrix = jnp.array([
            [0.9230, 0.9162],  # Z0, X0（未校準用平均）
            [0.9213, 0.9357],  # Z1, X1
            [0.9270, 0.9277],  # Z2, X2
            [0.8851, 0.9320],  # Z3, X3
            [0.9067, 0.8872],  # Z4, X4
        ])
        K_matrix = jnp.array([
            [0.0262, 0.0303],  # Z0, X0
            [0.0224, 0.0342],  # Z1, X1
            [0.0253, 0.0234],  # Z2, X2
            [0.0223, 0.0249],  # Z3, X3
            [0.0452, 0.0488],  # Z4, X4
        ])
        eps_std = K_matrix * jnp.sqrt(1.0 - (alpha_matrix * out) ** 2)
        key, subkey = jax.random.split(key)
        out = out * alpha_matrix + jax.random.normal(subkey, out.shape) * eps_std
        # ------------
    
        return out

        
        
    def __subCell2__(self, X, hidden, params):
        # for latest probability output
        inp = jnp.concatenate((X, hidden), axis=0)
        out = self.qnode(inp, params, probs_out=True)
        return out
    
    def __call__(self, Xs_b, hiddens, params, keys=None):
        fQGRU_subCell = partial(self.__subCell__, params=params)
        vQGRU_subCell = jax.vmap(fQGRU_subCell)
        # out = vQGRU_subCell(X=Xs_b, hidden=hiddens)
        if keys is None:
            out = vQGRU_subCell(X=Xs_b, hidden=hiddens)
        else:
            out = vQGRU_subCell(X=Xs_b, hidden=hiddens, key=keys)
        
        Ys1 = out[:,:,0]
        Ys2 = out[:,:,1]
                
        outs1 = Ys1[:,0:self.num_qb_i]
        outs2 = Ys2[:,0:self.num_qb_i]
        Hs1 = Ys1[:, self.num_qb_i:]
        Hs2 = Ys2[:, self.num_qb_i:]
        
        # [[1,2], [4,5], [7,8]] & [[11,12], [14,15], [17,18]]
        #     -> [[ 1,  2, 11, 12], [ 4,  5, 14, 15], [ 7,  8, 17, 18]]
#         Ys = jnp.concatenate((Ys1, Ys2), axis=1)
#         outs = jnp.concatenate((outs1, outs2), axis=1)
#         Hs = jnp.concatenate((Hs1, Hs2), axis=1)
                
        # [[1,2], [4,5], [7,8]] & [[11,12], [14,15], [17,18]]
        #     -> [[ 1, 11,  2, 12], [ 4, 14,  5, 15], [ 7, 17,  8, 18]]
        Ys = jnp.stack((Ys1, Ys2), axis=-1).reshape(len(Ys1) ,-1)
        outs = jnp.stack((outs1, outs2), axis=-1).reshape(len(outs1) ,-1)
        Hs = jnp.stack((Hs1, Hs2), axis=-1).reshape(len(Hs1) ,-1)

        return Ys, outs, Hs
    
    def __call2__(self, Xs_b, hiddens, params):
        # for latest probability output
        fQGRU_subCell = partial(self.__subCell2__, params=params)
        vQGRU_subCell = jax.vmap(fQGRU_subCell)
        out = vQGRU_subCell(X=Xs_b, hidden=hiddens)
        
        out = out[:,:self.pOut_dim]
        
        return out
    
    def draw_circuit(self, probs_out=False, level=None,
                     style='black_white', fontsize=24):
        qml.drawer.use_style(style)
        inp = jnp.ones(self.in_dim + self.hidden_dim)
        params = jnp.array(range(self.num_weights))
        self.circuit_fig, ax = qml.draw_mpl(self.qnode,level=level)\
        (inp, params, probs_out=probs_out)

        # set font size
        for text in ax.texts:
            text.set_fontsize(fontsize)
        
    def test(self):
        key = jax.random.PRNGKey(1)
        key1, key2 = jax.random.split(key, num=2)
        self.TEST_batch_size = 3
        self.TEST_inputs = jax.random.uniform(key1, (self.TEST_batch_size, self.in_dim))
        hiddens = jnp.zeros([self.TEST_batch_size, self.hidden_dim])
        params = jax.random.normal(key2, (self.num_weights,))
        self.TEST_outputs = self.__call__(self.TEST_inputs, hiddens, params)
        
        print('Test inputs:')
        print(self.TEST_inputs)
        print('Test outputs (Ys, outs, Hs):')
        print(self.TEST_outputs)
        
        if self.pOut_dim > 0:
            self.TEST_outputs2 = self.__call2__(self.TEST_inputs, hiddens, params)
            print('Test outputs 2 (last probability out):')
            print(self.TEST_outputs2)


# In[7]:


def qgru(in_dim, hidden_dim, number_of_layer=2, enc_layers=0, enc_reupload=1,
         hid_layers=0, hid_reupload=1, all_qubits_out=False, pred_length=1, out_type=-1,
         enc_n_weights_each=2, hid_n_weights_each=2, enc_v_gate=None, enc_lay_v_gate='rot',
         hid_v_gate=None, hid_lay_v_gate='rot', gate_gate='rot', lay_gate='rot',
         n_out_lay=None, out_gate='rot', n_hout_lay=None, hout_gate='rot', out_fun=None):
    # all_qubits_out
    #  True: all mesurments of qubits will be output (i.e. including hidden state qubits)
    #  False: only 'outs' mesurments will be output
    # out_type
    #  0: expected value, full sequence out
    #  -1: expected value, only the last value will be output
    #  >=1: probability, only the last value will be output; out_dim = out_type
    # out_fun
    #  output function: gruCell output values will be treated via this function to be qgru output values
        
    
    if pred_length < 1:
        raise ValueError(f"Prediction length must be greater than or equal to 1; got length {pred_length}.")
    
    gruCell = SQGRU(in_dim, hidden_dim, number_of_layer, enc_layers, enc_reupload,
                    hid_layers, hid_reupload, out_type,
                    enc_n_weights_each, hid_n_weights_each, enc_v_gate, enc_lay_v_gate,
                    hid_v_gate, hid_lay_v_gate, gate_gate, lay_gate,
                    n_out_lay, out_gate, n_hout_lay, hout_gate)
    
    def init_fun(rng, input_shape):
        # input_shape: (batch, x_length, x_dim)
        num_params = gruCell.num_weights
        params = jax.random.uniform(rng, (num_params,))*jnp.pi*2
        
        if all_qubits_out:
            out_dim = gruCell.out_dim + gruCell.hidden_dim # use all outputs of qubits
        else:
            out_dim = gruCell.out_dim
            
        if out_type == 0:
            # outputs of all steps with predictions (expected value)
            output_shape = (input_shape[0], input_shape[1]+pred_length-1, out_dim)
        elif out_type == -1:
            output_shape = (input_shape[0], out_dim) # get only the last output (expected value)
        else:
            output_shape = (input_shape[0], out_type) # get only the last output (probability)

        if out_fun != None:
            out_temp = jnp.ones(output_shape)
            out_temp = out_fun(out_temp)
            output_shape = out_temp.shape
        
        return output_shape, params
    
    def apply_fun(params, Xs, **kwargs):

        # 直接在這裡設定，不需要外面傳入
        rng = jax.random.PRNGKey(42)
        rng = jax.random.PRNGKey(int(time.time() * 1000) % (2**31))
        use_noise = True

        
        # initial hidden state
        h = jnp.zeros([jnp.shape(Xs)[0], gruCell.hidden_dim])
        batch_size = jnp.shape(Xs)[0]
        init_key = rng if use_noise else jax.random.PRNGKey(0)
 
        # ── scan functions ────────────────────────────────────────────────
        # carry: (hidden, key)
 
        def apply_fun_scan01(carry, X):
            hidden, key = carry
            key, subkey = jax.random.split(key)
            batch_keys = jax.random.split(subkey, num=batch_size)
            out, _, new_hidden = gruCell(X, hidden, params,
                                         keys=batch_keys if use_noise else None)
            if out_fun != None:
                out = out_fun(out)
            return (new_hidden, key), out
 
        def apply_fun_scan02(carry, X):
            hidden, key = carry
            key, subkey = jax.random.split(key)
            batch_keys = jax.random.split(subkey, num=batch_size)
            _, out, new_hidden = gruCell(X, hidden, params,
                                          keys=batch_keys if use_noise else None)
            if out_fun != None:
                out = out_fun(out)
            return (new_hidden, key), out
 
        # ── prediction functions ──────────────────────────────────────────
        # carry: [hidden, out_prev, key]
 
        def apply_fun_pred01(carry, X):
            hidden = carry[0]
            X     = carry[1]
            key   = carry[2]
            key, subkey = jax.random.split(key)
            batch_keys = jax.random.split(subkey, num=batch_size)
            out, _, new_hidden = gruCell(X, hidden, params,
                                          keys=batch_keys if use_noise else None)
            if out_fun != None:
                out = out_fun(out)
            return [new_hidden, out, key], out
 
        def apply_fun_pred02(carry, X):
            hidden = carry[0]
            X     = carry[1]
            key   = carry[2]
            key, subkey = jax.random.split(key)
            batch_keys = jax.random.split(subkey, num=batch_size)
            _, out, new_hidden = gruCell(X, hidden, params,
                                          keys=batch_keys if use_noise else None)
            if out_fun != None:
                out = out_fun(out)
            return [new_hidden, out, key], out
 
        # ── Move the time dimension to position 0 ────────────────────────
        Xs = jnp.moveaxis(Xs, 1, 0)
 
        if out_type > 0:
            if pred_length == 1:
                X_last = Xs[-1]
                Xs = Xs[:-1]
                p_len = pred_length - 1
            else:
                p_len = pred_length - 2
        else:
            p_len = pred_length - 1
 
        # ── lax.scan ─────────────────────────────────────────────────────
        if all_qubits_out:
            (h_new, _), out = lax.scan(apply_fun_scan01, (h, init_key), Xs)
        else:
            (h_new, _), out = lax.scan(apply_fun_scan02, (h, init_key), Xs)
 
        # ── prediction part ───────────────────────────────────────────────
        if p_len > 0:
            # check output shape (i.e. check output function)
            if Xs[0].shape != out[-1].shape:
                raise ValueError(f"The output shape {out[-1].shape} \
is not equal to the input shape {Xs[0].shape}. Please check the output function for qgru.")
 
            h2 = [h_new, out[-1], init_key]
            if all_qubits_out:
                new_hid_x, out2 = lax.scan(apply_fun_pred01, h2, jnp.zeros(p_len))
            else:
                new_hid_x, out2 = lax.scan(apply_fun_pred02, h2, jnp.zeros(p_len))
            h_new = new_hid_x[0]
 
        # ── output formatting (unchanged) ─────────────────────────────────
        if out_type == 0:
            out = jnp.moveaxis(out, 1, 0)
            if pred_length > 1:
                out2 = jnp.moveaxis(out2, 1, 0)
                out = jnp.concatenate([out, out2], axis=1)
        elif out_type == -1:
            if pred_length > 1:
                out = out2[-1]
            else:
                out = out[-1]
        else:
            if pred_length > 1:
                X_last = out2[-1]
            out = gruCell.__call2__(X_last, h_new, params)
 
        return out
        
    return init_fun, apply_fun

