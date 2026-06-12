#!/usr/bin/env python
# coding: utf-8

# In[1]:


import jax
import jax.numpy as jnp
from jax import random
from jax import lax
from jax.nn.initializers import glorot_normal, normal
from jax.nn import sigmoid
from jax.scipy.special import expit
from jax.example_libraries import stax
from functools import partial


# ## RNN

# In[2]:


def simple_rnn(out_dim, W_init=glorot_normal(), b_init=normal(), sequence_out=True):
    def init_fun(rng, input_shape):

        k1, k2, k3 = random.split(rng, num=3)
        W_hh, U_hh, b_hh = (
            W_init(k1, (input_shape[2], out_dim)),
            W_init(k2, (out_dim, out_dim)),
            b_init(k3, (out_dim,)),
        )
           
        if sequence_out:
            output_shape = (input_shape[0], input_shape[1], out_dim) # outputs of all steps
        else:
            output_shape = (input_shape[0], out_dim) # get only the last output

        return (output_shape, (W_hh, U_hh, b_hh))

    def apply_fun(params, inputs, **kwargs):
        
        h = jnp.zeros([jnp.shape(inputs)[0], out_dim]) # maybe ones?

        def apply_fun_scan(params, hidden, inp):
            (W_hh, U_hh, b_hh) = params

            new_hidden = jnp.tanh(jnp.dot(inp, W_hh) + jnp.dot(hidden, U_hh) + b_hh)
            return new_hidden, new_hidden

        inputs = jnp.moveaxis(inputs, 1, 0)
        f = partial(apply_fun_scan, params)
        _, h_new = lax.scan(f, h, inputs)
        
        if sequence_out:
            return jnp.moveaxis(h_new, 1, 0) # output all steps
        else:
            return h_new[-1] # only output the last step
        
        return h_new

    return init_fun, apply_fun


# ## GRU

# In[3]:


def gru(out_dim, W_init=glorot_normal(), b_init=normal(), sequence_out=True):
    def init_fun(rng, input_shape):
        """ Initialize the GRU layer for stax """

        k1, k2, k3 = random.split(rng, num=3)
        update_W, update_U, update_b = (
            W_init(k1, (input_shape[2], out_dim)),
            W_init(k2, (out_dim, out_dim)),
            b_init(k3, (out_dim,)),)

        k1, k2, k3 = random.split(rng, num=3)
        reset_W, reset_U, reset_b = (
            W_init(k1, (input_shape[2], out_dim)),
            W_init(k2, (out_dim, out_dim)),
            b_init(k3, (out_dim,)),)

        k1, k2, k3 = random.split(rng, num=3)
        out_W, out_U, out_b = (
            W_init(k1, (input_shape[2], out_dim)),
            W_init(k2, (out_dim, out_dim)),
            b_init(k3, (out_dim,)),)
        # Input dim 0 represents the batch dimension
        # Input dim 1 represents the time dimension (before scan moveaxis)
           
        if sequence_out:
            output_shape = (input_shape[0], input_shape[1], out_dim) # outputs of all steps
        else:
            output_shape = (input_shape[0], out_dim) # get only the last output
            
        return (output_shape, ((update_W, update_U, update_b), (reset_W, reset_U, reset_b), (out_W, out_U, out_b),))

    def apply_fun(params, inputs, **kwargs):
        """ Loop over the time steps of the input sequence """
        h = jnp.zeros([jnp.shape(inputs)[0], out_dim]) # maybe ones?
#         h = params[0]

        def apply_fun_scan(params, hidden, inp):
            """ Perform single step update of the network """
            (update_W, update_U, update_b), (reset_W, reset_U, reset_b), (out_W, out_U, out_b) = params

            update_gate = sigmoid(jnp.dot(inp, update_W) +
                                  jnp.dot(hidden, update_U) + update_b)
            reset_gate = sigmoid(jnp.dot(inp, reset_W) +
                                 jnp.dot(hidden, reset_U) + reset_b)
            output_gate = jnp.tanh(jnp.dot(inp, out_W)
                                  + jnp.dot(jnp.multiply(reset_gate, hidden), out_U)
                                  + out_b)
            output = jnp.multiply(update_gate, hidden) + jnp.multiply(1-update_gate, output_gate)
            
            hidden = output
            return hidden, hidden

        # Move the time dimension to position 0
        inputs = jnp.moveaxis(inputs, 1, 0)
        f = partial(apply_fun_scan, params)
        _, h_new = lax.scan(f, h, inputs)
        
        if sequence_out:
            return jnp.moveaxis(h_new, 1, 0) # output all steps
        else:
            return h_new[-1] # only output the last step
        
        return h_new

    return init_fun, apply_fun


# ## LSTM

# In[4]:


def lstm(out_dim, W_init=glorot_normal(), b_init=normal(), sequence_out=True):
    def init_fun(rng, input_shape):
#         hidden = b_init(rng, (input_shape[0], out_dim))

        k1, k2, k3 = random.split(rng, num=3)
        W_ci, U_ci, b_ci = (
            W_init(k1, (input_shape[2], out_dim)),
            W_init(k2, (out_dim, out_dim)),
            b_init(k3, (out_dim,)),
        )

        k1, k2, k3 = random.split(rng, num=3)
        W_if, U_if, b_if = (
            W_init(k1, (input_shape[2], out_dim)),
            W_init(k2, (out_dim, out_dim)),
            b_init(k3, (out_dim,)),
        )

        k1, k2, k3 = random.split(rng, num=3)
        W_ig, U_ig, b_ig = (
            W_init(k1, (input_shape[2], out_dim)),
            W_init(k2, (out_dim, out_dim)),
            b_init(k3, (out_dim,)),
        )

        k1, k2, k3 = random.split(rng, num=3)
        W_io, U_io, b_io = (
            W_init(k1, (input_shape[2], out_dim)),
            W_init(k2, (out_dim, out_dim)),
            b_init(k3, (out_dim,)),
        )
           
        if sequence_out:
            output_shape = (input_shape[0], input_shape[1], out_dim) # outputs of all steps
        else:
            output_shape = (input_shape[0], out_dim) # get only the last output

#         return (output_shape, (hidden, (W_ci, U_ci, b_ci, W_if, U_if, b_if, W_ig, U_ig, b_ig, W_io, U_io, b_io)),)
        return (output_shape, ((W_ci, U_ci, b_ci, W_if, U_if, b_if, W_ig, U_ig, b_ig, W_io, U_io, b_io)))

    def apply_fun(params, inputs, **kwargs):
#         h = params[0]
        h = jnp.zeros([jnp.shape(inputs)[0], out_dim]) # maybe ones?

        def apply_fun_scan(params, hidden, inp):
            (W_ci, U_ci, b_ci, W_if, U_if, b_if, W_ig, U_ig, b_ig, W_io, U_io, b_io) = params

            i_t = expit(jnp.dot(inp, W_if) + jnp.dot(hidden, U_if) + b_if)
            f_t = expit(jnp.dot(inp, W_if) + jnp.dot(hidden, U_if) + b_if)
            g_t = jnp.tanh(jnp.dot(inp, W_ig) + jnp.dot(hidden, U_ig) + b_ig)
            o_t = expit(jnp.dot(inp, W_io) + jnp.dot(hidden, U_io) + b_io)

            c_t = f_t * hidden + i_t * g_t
            new_hidden = o_t * jnp.tanh(c_t)
            return new_hidden, new_hidden

        inputs = jnp.moveaxis(inputs, 1, 0)
        f = partial(apply_fun_scan, params)
        _, h_new = lax.scan(f, h, inputs)
        
        if sequence_out:
            return jnp.moveaxis(h_new, 1, 0) # output all steps
        else:
            return h_new[-1] # only output the last step
        
        return h_new

    return init_fun, apply_fun


# ## Count parameters

# In[19]:


def count_parameters(params):
    total_params = 0
    for p in params:
        if isinstance(p, tuple):
            total_params += count_parameters(p)
        else:
            total_params += jnp.size(p)
    return total_params
