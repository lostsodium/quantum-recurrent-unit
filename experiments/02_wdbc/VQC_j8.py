#!/usr/bin/env python
# coding: utf-8

# https://docs.pennylane.ai/en/stable/code/api/pennylane.BasicEntanglerLayers.html

# In[1]:


import pennylane as qml
from pennylane.operation import Operation, AnyWires

from jax import lax
import jax
import jax.numpy as jnp

from functools import partial
import math


# In[2]:


class temp_Encoding(Operation):
    num_wires = AnyWires

    def __init__(self, weights, wires, inputs, vari_gate=None, id=None):
        # vari_gate: None, 'rx', 'ry', 'rz', 'rot'
        [None, 'rx', 'ry', 'rz', 'rot', 'u1', 'u2', 'u3'].index(vari_gate)

        input_dim = len(inputs)
        num_weights = len(weights)

        num_wires = len(wires)
        num_wires_i = math.ceil(input_dim/2) # required number of qubits for input
        
        if num_wires != num_wires_i:
            raise ValueError(f"Wires must be of length {num_wires_i}; got length {num_wires}.")
        
        # number of parameters(weights)
        if vari_gate == None:
            vari_params = 0
        elif vari_gate == 'rot' or vari_gate == 'u3':
            vari_params = num_wires*3
        elif vari_gate == 'u2':
            vari_params = num_wires*2
        else:
            vari_params = num_wires
        
        req_params1 = input_dim + vari_params
        req_params2 = input_dim*2 + vari_params
        
        if num_weights != vari_params and num_weights != req_params1 and num_weights != req_params2:
            raise ValueError(f"Weights must be of length {vari_params} or {req_params1} or {req_params2}; got length {len(weights)}.")
        
        self._hyperparameters = {"inputs": inputs, "vari_gate": vari_gate}
        
        super().__init__(weights, wires=wires, id=id)
        
    def label(self, decimals=None, base_label=None, cache=None):
        return self.id
    
    @staticmethod
    def num_req_params(input_dim, num_weights_each, vari_gate):
        [None, 'rx', 'ry', 'rz', 'rot', 'u1', 'u2', 'u3'].index(vari_gate)
        num_wires = math.ceil(input_dim/2)
        if vari_gate == None:
            vari_params = 0
        elif vari_gate == 'rot' or vari_gate == 'u3':
            vari_params = num_wires*3
        elif vari_gate == 'u2':
            vari_params = num_wires*2
        else:
            vari_params = num_wires
        return input_dim*num_weights_each + vari_params

    @staticmethod
    def compute_decomposition(weights, wires, inputs, vari_gate):
        # vari_gate: None, 'rx', 'ry', 'rz', 'rot'
        [None, 'rx', 'ry', 'rz', 'rot', 'u1', 'u2', 'u3'].index(vari_gate)
        
        op_list = []
        input_dim = len(inputs)
        
        if vari_gate == None:
            e_n_weights = len(weights)
        elif vari_gate == 'rot' or vari_gate == 'u3':
            e_n_weights = len(weights) - len(wires)*3
        elif vari_gate == 'u2':
            e_n_weights = len(weights) - len(wires)*2
        else:
            e_n_weights = len(weights) - len(wires)
        
        num_weights_each = math.floor(e_n_weights/input_dim)
        
        # encode
        rx = True
        for i in range(input_dim):
            if num_weights_each == 0:
                theta = inputs[i]
            elif num_weights_each == 1:
                theta = inputs[i]*weights[i]
            else:
                theta = inputs[i]*weights[i*2]+weights[i*2+1]

            if rx:
                op_list.append(qml.RX(theta, wires[math.floor(i/2)]))
            else:
                op_list.append(qml.RY(theta, wires[math.floor(i/2)]))
            rx = not rx
            
        # variational part
        wi = num_weights_each*input_dim
        if len(weights) != 0:
            for wire in wires:
                if vari_gate == 'rot':
                    op_list.append(qml.Rot(weights[wi],weights[wi+1],weights[wi+2], wire))
                    wi += 3
                elif vari_gate == 'rx':
                    op_list.append(qml.RX(weights[wi], wire))
                    wi += 1
                elif vari_gate == 'ry':
                    op_list.append(qml.RY(weights[wi], wire))
                    wi += 1
                elif vari_gate == 'rz':
                    op_list.append(qml.RZ(weights[wi], wire))
                    wi += 1
                elif vari_gate == 'u1':
                    op_list.append(qml.U1(weights[wi], wire))
                    wi += 1
                elif vari_gate == 'u2':
                    op_list.append(qml.U2(weights[wi],weights[wi+1], wire))
                    wi += 2
                elif vari_gate == 'u3':
                    op_list.append(qml.U3(weights[wi],weights[wi+1],weights[wi+2], wire))
                    wi += 3

        return op_list


# In[3]:


class temp_Variation(Operation):
    num_wires = AnyWires

    def __init__(self, weights, wires, gate='rot', id=None):
        # gate: rx, ry, rz, rot, or None for no rotaion gates (only entanglement)

        num_wires = len(wires)
        
        # number of parameters(weights)
        req_params = self.num_req_params(num_wires, gate)
        
        if len(weights) != req_params:
            raise ValueError(f"Weights must be of length {req_params}; got length {len(weights)}.")
        
        self._hyperparameters = {"gate": gate}
        
        super().__init__(weights, wires=wires, id=id)
        
    def label(self, decimals=None, base_label=None, cache=None):
        return self.id
    
    @staticmethod
    def num_req_params(num_wires, gate):
        [None, 'rx', 'ry', 'rz', 'rot', 'u1', 'u2', 'u3'].index(gate)
        if gate == None:
            req_params = 0
        elif gate == 'rot' or gate == 'u3':
            req_params = 3*num_wires
        elif gate == 'u2':
            req_params = 2*num_wires
        else:
            req_params = num_wires
        return req_params

    @staticmethod
    def compute_decomposition(weights, wires, gate):
        [None, 'rx', 'ry', 'rz', 'rot', 'u1', 'u2', 'u3'].index(gate)
        
        op_list = []
        num_wires = len(wires)
        
        #---------- sub functions ---------------
        def __entangle__(wires):
            if len(wires) > 1:
                q1s = wires[:-1]
                q2s = wires[1:]
                for q1, q2 in zip(q1s, q2s):
                    op_list.append(qml.CNOT(wires=[q1, q2]))
                op_list.append(qml.CNOT(wires=[wires[-1], wires[0]]))

        def __vari__(weights, wires, gate):
            wi = 0
            for i in wires:
                if gate == 'rot':
                    op_list.append(qml.Rot(weights[wi],weights[wi+1],weights[wi+2], wires=i))
                    wi += 3
                elif gate == 'rx':
                    op_list.append(qml.RX(weights[wi], wires=i))
                    wi += 1
                elif gate == 'ry':
                    op_list.append(qml.RY(weights[wi], wires=i))
                    wi += 1
                elif gate == 'rz':
                    op_list.append(qml.RZ(weights[wi], wires=i))
                    wi += 1
                elif gate == 'u1':
                    op_list.append(qml.U1(weights[wi], wires=i))
                    wi += 1
                elif gate == 'u2':
                    op_list.append(qml.U2(weights[wi],weights[wi+1], wires=i))
                    wi += 2
                elif gate == 'u3':
                    op_list.append(qml.U3(weights[wi],weights[wi+1],weights[wi+2], wires=i))
                    wi += 3
        #--------------------------------------------

        if gate != None:
            __vari__(weights, wires, gate)
            
        __entangle__(wires)

        return op_list


# In[4]:


class temp_VQC(Operation):
    num_wires = AnyWires

    def __init__(self, weights, wires, inputs, n_layers=2, n_reupload=1, num_weights_each=2,
                 enc_vari_gate=None, lay_vari_gate='rot', id=None):
        # num_weights_each: number of weights for each encoding gate
        # if inputs == None, there will be no encoding part and only variation part.
        if inputs == None:
            input_dim = 0
        else:
            input_dim = len(inputs)

        num_wires = len(wires)
        num_wires_i = math.ceil(input_dim/2) # required number of qubits for input
        
        if num_wires < num_wires_i:
            raise ValueError('Number of wires is not enough for input.')
        
        # number of parameters(weights)
        req_params = self.num_req_params(input_dim, num_wires, n_layers, n_reupload, num_weights_each,
                                         enc_vari_gate, lay_vari_gate)
        
        if len(weights) != req_params:
            raise ValueError(f"Weights must be of length {req_params}; got length {len(weights)}.")
        
        self._hyperparameters = {"inputs": inputs, "n_layers": n_layers, "n_reupload": n_reupload,
                                 "num_weights_each": num_weights_each, "enc_vari_gate": enc_vari_gate,
                                 "lay_vari_gate": lay_vari_gate}
        
        super().__init__(weights, wires=wires, id=id)
        
    def label(self, decimals=None, base_label=None, cache=None):
        return self.id
    
    @staticmethod
    def num_req_params(input_dim, num_wires, n_layers, n_reupload, num_weights_each, enc_vari_gate, lay_vari_gate):
        if input_dim == 0:
            enc_params1 = enc_params2 = 0
        else:
            enc_params1 = temp_Encoding.num_req_params(input_dim, num_weights_each, enc_vari_gate)
            enc_params2 = temp_Encoding.num_req_params(input_dim, num_weights_each, None)
        
        if type(lay_vari_gate) != list:
            lay_vari_gate = [lay_vari_gate]
        var_params = 0
        for vari_gate in lay_vari_gate:
            var_params += temp_Variation.num_req_params(num_wires, vari_gate)
        
        req_params = (n_reupload-1)*enc_params1 + enc_params2 + n_layers*var_params
        return req_params

    @staticmethod
    def compute_decomposition(weights, wires, inputs, n_layers, n_reupload,
                              num_weights_each, enc_vari_gate, lay_vari_gate):
        
        op_list = []
        num_wires = len(wires)
        
        if type(lay_vari_gate) != list:
            lay_vari_gate = [lay_vari_gate]
        
        wi = 0
        
        # encode
        if inputs != None:
            input_dim = len(inputs)
            num_wires_i = math.ceil(input_dim/2)
            wires_i = wires[0:num_wires_i]        
            for i in range(n_reupload-1):
                wj = wi + temp_Encoding.num_req_params(input_dim, num_weights_each, enc_vari_gate)
                op_list += temp_Encoding.compute_decomposition(weights[wi:wj], wires_i, inputs, vari_gate=enc_vari_gate)
                wi = wj

            wj = wi + temp_Encoding.num_req_params(input_dim, num_weights_each, None)
            op_list += temp_Encoding.compute_decomposition(weights[wi:wj], wires_i, inputs, vari_gate=None)
            wi = wj

            op_list.append(qml.Barrier(wires))
        
        # layer
        vn = len(lay_vari_gate)
        for i in range(n_layers):
            vari_gate = lay_vari_gate[i % vn]
            wj = wi + temp_Variation.num_req_params(num_wires, vari_gate)
            op_list += temp_Variation.compute_decomposition(weights[wi:wj], wires, gate=vari_gate)
            wi = wj

        return op_list


# In[5]:


class SVQCirc:
    def __init__(self, input_dim, output_dim, output_type='e',
                 number_of_layers=2, number_of_reupload=1, number_of_qubits=0,
                 num_weights_each=2, enc_vari_gate=None, lay_vari_gate='rot'):
        super(SVQCirc, self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.output_type = output_type

        qN_i = math.ceil(input_dim/2) # required number of qubits for input
        
        # required number of qubits for output
        if self.output_type == 'p':
            qN_o = math.ceil(jnp.log2(output_dim))
        else:
            qN_o = output_dim
        
        self.qN = max(qN_i, qN_o, number_of_qubits)
        self.qN_i = qN_i
        
        self.wires = list(range(self.qN))
        self.wires_o = list(range(qN_o)) # required wires for output
        
        self.num_lay = number_of_layers
        self.num_reup = number_of_reupload
        
        self.num_weights_each = num_weights_each
        self.enc_vari_gate = enc_vari_gate
        self.lay_vari_gate = lay_vari_gate
        
        # number of weights
        self.num_weights = temp_VQC.num_req_params(input_dim=self.input_dim, num_wires=self.qN,
                                                   n_layers=self.num_lay, n_reupload=self.num_reup,
                                                   num_weights_each=self.num_weights_each, 
                                                   enc_vari_gate=self.enc_vari_gate, lay_vari_gate=self.lay_vari_gate)
        
    def __call__(self, inputs, weights):
        
        if len(inputs) != self.input_dim:
            raise ValueError('Input size is not equal to input dimension.')
            
        if self.input_dim == 0:
            inputs = None
        
        temp_VQC(weights=weights, wires=self.wires, inputs=inputs, n_layers=self.num_lay,
                 n_reupload=self.num_reup, num_weights_each=self.num_weights_each,
                 enc_vari_gate=self.enc_vari_gate, lay_vari_gate=self.lay_vari_gate, id='VQC')
        
        if self.output_type == 'p':
            return qml.probs(wires=self.wires_o)
        else:
            return [qml.expval(qml.PauliZ(i)) for i in range(self.output_dim)]
        


# In[6]:


class SVQC:
    def __init__(self, input_dim, output_dim, output_type='e', 
                 number_of_layers=2, number_of_reupload=1, number_of_qubits=0,
                 num_weights_each=2, enc_vari_gate=None, lay_vari_gate='rot'):
        # output_type: 'e': expection value; 'p': probability
        super(SVQC, self).__init__()
        
        self.in_dim = input_dim
        self.out_dim = output_dim
        self.out_type = output_type
        
        if self.out_type == 'p':
            num_qb = max(math.ceil(input_dim/2), math.ceil(jnp.log2(output_dim)), number_of_qubits)
        else:
            num_qb = max(math.ceil(input_dim/2), output_dim)
        
        self.num_qb = num_qb
        
        dev = qml.device("default.qubit", wires=num_qb)
        circuit = SVQCirc(input_dim, output_dim, output_type, number_of_layers, number_of_reupload,
                          number_of_qubits, num_weights_each, enc_vari_gate, lay_vari_gate)          
        self.num_weights = circuit.num_weights
        self.qnode = qml.QNode(circuit, dev)
    
    def __call__(self, Xs_b, params):
        fQGRU_subCell = partial(self.qnode, weights=params)
        vQGRU_subCell = jax.vmap(fQGRU_subCell)
        out = vQGRU_subCell(inputs=Xs_b)
        if self.out_type == 'p':
            out = out[:,:self.out_dim]
        else:
            out = jnp.transpose(jnp.array(out))
        return out
    
    def draw_circuit(self, expansion_strategy=None, style='black_white', fontsize=24):
        qml.drawer.use_style(style)
        inp = jnp.zeros(self.in_dim)
        params = jnp.zeros(self.num_weights)
        self.circuit_fig, ax = qml.draw_mpl(self.qnode, expansion_strategy=expansion_strategy)(inp, params)

        # set font size
        for text in ax.texts:
            text.set_fontsize(fontsize)
        
    def test(self):
        key = jax.random.PRNGKey(1)
        key1, key2 = jax.random.split(key, num=2)
        self.TEST_batch_size = 3
        self.TEST_inputs = jax.random.uniform(key1, (self.TEST_batch_size, self.in_dim))
        params = jax.random.normal(key2, (self.num_weights,))
        self.TEST_outputs = self.__call__(self.TEST_inputs, params)
        self.TEST_parameters = params
        
        print('Test inputs:')
        print(self.TEST_inputs)
        print('Test outputs:')
        print(self.TEST_outputs)


# In[7]:


def vqc(in_dim, out_dim, out_type, number_of_layer=2, number_of_reupload=1, number_of_qubits=0,
                 num_weights_each=2, enc_vari_gate=None, lay_vari_gate='rot'):
    vqcModel = SVQC(in_dim, out_dim, out_type, number_of_layer, number_of_reupload, number_of_qubits,
                 num_weights_each, enc_vari_gate, lay_vari_gate)
    
    def init_fun(rng, input_shape):
        # input_shape: (batch, x_length, x_dim)
        num_params = vqcModel.num_weights
        params = jax.random.uniform(rng, (num_params,))*jnp.pi*2
        out_dim = vqcModel.out_dim
        output_shape = (input_shape[0], out_dim) # get only the last output
        
        return output_shape, params
    
    def apply_fun(params, Xs, **kwargs):
        return vqcModel(Xs, params)
        
    return init_fun, apply_fun
