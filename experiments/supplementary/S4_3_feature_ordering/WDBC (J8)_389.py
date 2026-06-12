#!/usr/bin/env python
# coding: utf-8

# ### Path

# In[1]:


# import ipynbname
# nb_name = ipynbname.name()
# nb_path = ipynbname.path()

# nb_name = nb_name + 'test'


# In[ ]:


import os.path


# In[ ]:


# for python script
nb_name = os.path.basename(__file__)[0:-3]
print("Current filename:", nb_name) 


# ## Import

# In[2]:


# Added to silence some warnings.
# from jax.config import config
# config.update("jax_enable_x64", True)

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
# import pennylane as qml
from jax.example_libraries import optimizers
from jax.example_libraries import stax
from jax import value_and_grad
from jax import lax

import optax
from functools import partial
import time

# from sklearn.datasets import load_breast_cancer
# from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import StratifiedKFold
from WDBC_dataset_2 import WDBC_dataset

import numpy as np
import random
import copy
import pickle
import matplotlib.pyplot as plt
import csv
import os
from datetime import datetime

from TRAIN_v4_debug_3 import TRAIN


# In[3]:


# No noise
from SQGRU_j8 import SQGRU, qgru


# In[ ]:





# In[4]:


SEED = 0
N_STEPS = 1000000
NUM_SEEDs = 50
BATCH_SIZE = 20


# In[5]:


jax.config.update('jax_platform_name', 'cpu') # for cpu


# ## Model

# ### QGRU
# ### Hyper parameters

# In[6]:


# model parameters
I_DIM = 1 # input dim
H_DIM = 8 # hidden dim
N_LAY = 3 # number of layers
E_LAY = 1 # enc_layers
E_REUL = 1 # enc_reupload
H_LAY = 1 # hid_layers
H_REUL = 1 # hid_reupload
PO_DIM = -1 # pOut_dim
A_Q_OUT = False # all_qubits_out
PRED_LEN = 1 # pred_length
OUT_TYP = -1 # out_type
ENWE = 2 # enc_n_weights_each
HNWE = 2 # hid_n_weights_each
EVG = None # enc_v_gate
ELVG = None # enc_lay_v_gate
HVG = None # hid_v_gate
HLVG = None # hid_lay_v_gate
G_GATE = 'rx' # gate_gate
L_GATE = ['rx', 'ry'] # lay_gate
N_O_LAY = None # n_out_lay
O_GATE = 'u2' # out_gate
N_HO_LAY = None # n_hout_lay
HO_GATE = 'u2' # hout_gate

# NOISE_MODEL = None
# SHOTS = 1024


# In[7]:


# out_fn = None
def out_fn(x):
    return x[:,0]


# In[8]:


# Trainable scale
def TrainableScale(init_scale=5.0):
    # 定義初始化函數，將 scale 參數初始化為 init_scale
    def init_fun(rng, input_shape):
        scale = 1.0*init_scale
        return input_shape, (scale,)

    # 定義應用函數，將輸入 x 乘以 scale
    def apply_fun(params, x, **kwargs):
        scale, = params
        return x * scale
    
    return init_fun, apply_fun


# In[9]:


init_fun, qgru_rnn = stax.serial(qgru(I_DIM,H_DIM,N_LAY, enc_layers=E_LAY, enc_reupload=E_REUL,
                                      hid_layers=H_LAY, hid_reupload=H_REUL, all_qubits_out=A_Q_OUT,
                                      pred_length=PRED_LEN, out_type=OUT_TYP, enc_n_weights_each=ENWE,
                                      hid_n_weights_each=HNWE, enc_v_gate=EVG, enc_lay_v_gate=ELVG,
                                      hid_v_gate=HVG, hid_lay_v_gate=HLVG, gate_gate=G_GATE, lay_gate=L_GATE,
                                      n_out_lay=N_O_LAY, out_gate=O_GATE, n_hout_lay=N_HO_LAY, hout_gate=HO_GATE,
                                      out_fun=out_fn), TrainableScale(100))


# In[8]:


# init_fun, qru_rnn = qgru(I_DIM,H_DIM,N_LAY, enc_layers=E_LAY, enc_reupload=E_REUL,
#                                       hid_layers=H_LAY, hid_reupload=H_REUL, all_qubits_out=A_Q_OUT,
#                                       pred_length=PRED_LEN, out_type=OUT_TYP, enc_n_weights_each=ENWE,
#                                       hid_n_weights_each=HNWE, enc_v_gate=EVG, enc_lay_v_gate=ELVG,
#                                       hid_v_gate=HVG, hid_lay_v_gate=HLVG, gate_gate=G_GATE, lay_gate=L_GATE,
#                                       n_out_lay=N_O_LAY, out_gate=O_GATE, n_hout_lay=N_HO_LAY, hout_gate=HO_GATE,
#                                       out_fun=out_fn)

# key = jax.random.PRNGKey(SEED)


# ## Train

# In[10]:


@jax.jit
def loss_fn(params, inputs, targets):
#     inputs = jnp.concatenate((inputs, inputs), axis=2) # 1D inputs -> 2D inputs (data reupload)
    # inputs = jnp.concatenate((inputs, inputs**2, inputs, inputs**2), axis=2) # 1D inputs -> 2D inputs (data reupload)
    logits = qgru_rnn(params, inputs)
    
    return jnp.mean(optax.sigmoid_binary_cross_entropy(logits, targets))

def result_fn(params, dataset):
    inp, tar = zip(*dataset)
    inp = jnp.array(inp)
    # inp = jnp.concatenate((inp, inp**2, inp, inp**2), axis=2)
    result = jax.nn.sigmoid(qgru_rnn(params, inp))
    s = 0
    for r, t in zip(result, tar):
        if abs(r-t) < 0.5:
            s += 1
    return s/len(result)


# In[9]:


# @jax.jit
# def loss_fn(params, inputs, targets):
#     outputs  = qru_rnn(params, inputs)
#     targets_pm = targets * 2 - 1  # 0/1 → -1/+1
#     return jnp.mean((outputs - targets_pm) ** 2)

# def result_fn(params, dataset):
#     inp, tar = zip(*dataset)
#     inp     = jnp.array(inp)
#     outputs = qru_rnn(params, inp)
#     s = 0
#     for out, t in zip(outputs, tar):
#         pred = 1 if out > 0 else 0
#         if pred == int(t):
#             s += 1
#     return s / len(outputs)


# ### Dataset

# In[11]:


def shuffle_feature_order(dataset, seed=None):
    """
    Apply a fixed random permutation to the feature sequence of all samples.
    
    Args:
        dataset: list of [array(30, 1), label]
        seed: random seed for reproducibility
    
    Returns:
        shuffled_dataset: same format, features reordered
        perm: the permutation index array used (for logging/reproducibility)
    """
    rng = np.random.default_rng(seed)
    n_features = dataset[0][0].shape[0]  # 30
    perm = rng.permutation(n_features)
    
    shuffled_dataset = [
        [sample[0][perm], sample[1]]
        for sample in dataset
    ]
    return shuffled_dataset, perm


# In[12]:


# seeds_5 = [42, 137, 389, 802, 1234]
seeds_5 = [389]
seeds_10 = [42, 137, 256, 389, 514, 671, 802, 943, 1057, 1234]


# In[ ]:


# seeds_5 = [42, 137]
# seeds_10 = [42, 137]


# In[ ]:


timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
filename = f"Results_389_{timestamp}.csv"
with open(filename, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['seed 1', 'Permutation', 'seed 2', 'acc.'])


    
k1 = 0
for seed1 in seeds_5:
    k1 += 1
    # 加載 WDBC 數據集
    WDBC = WDBC_dataset(seed1)
    shuffled_data, perm = shuffle_feature_order(WDBC.all_dataset, seed=seed1)
    WDBC.all_dataset = shuffled_data
    train_dataset, valid_dataset, test_dataset = WDBC.split([341,114,114]) # 340/113/116

    k2 = 0
    for seed2 in seeds_10:
        k2 += 1
        print(str(k1) + '_' + str(k2))
        save_name = nb_name + '_random_seq_seeds_' + str(seed1) + '_' + str(seed2)
        key = jax.random.PRNGKey(seed2)
        train = TRAIN(key, init_fun, loss_fn, train_dataset+valid_dataset, test_dataset, result_fn, save_name)
        train.N_STEPS = N_STEPS
        train.BATCH_SIZE = BATCH_SIZE
        train.NUM_SEEDs = 1
        train.STD_DEV = 0.0
        train.REC_INTE = 10
        train.VARI_FRE = 'epoch'
        train.ini_learning_rate = 0.01
        
        train.TRAIN_VALID_TEST = jnp.array([len(train_dataset), len(valid_dataset)])
        train.ES_THRES = 1e-2
        train.ES_LEN = 3
        train.ES_MODE = 'loss'
        train.ES_DATASET = 'train'
        
        train.train()

        acc = result_fn(train.vbest_params, test_dataset)

        # save
        with open(filename, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([int(seed1), str(perm), int(seed2), float(acc),])
        
        
        


# In[ ]:



