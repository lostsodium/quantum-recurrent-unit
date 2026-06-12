#!/usr/bin/env python
# coding: utf-8

# https://pennylane.ai/qml/demos/tutorial_jax_transformations.html

# In[1]:


# import ipynbname
# nb_name = ipynbname.name()
# nb_path = ipynbname.path()

# nb_name


# In[ ]:


import os.path


# In[ ]:


# for python script
nb_name = os.path.basename(__file__)[0:-3]
print("Current filename:", nb_name) 


# In[2]:


SEED = 2
N_STEPS = 100000
BATCH_SIZE = 50
STD_DEV = 0


# In[3]:


import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
from jax.example_libraries import optimizers
from jax.example_libraries import stax
from jax import value_and_grad
from jax import lax

import optax
from functools import partial
import time

from sklearn.model_selection import StratifiedKFold
from MNIST_3_5_dataset import MNIST_3_5_dataset

import numpy as np
import random
import copy
import pickle
import matplotlib.pyplot as plt

from TRAIN_v4_debug_4 import TRAIN


# In[ ]:


# # No Noise Model
# from SQGRU_j8 import SQGRU, qgru


# In[4]:


# # Noise Model (Marrakesh)
# from SQGRU_j8_MNIST_NoiseModel import SQGRU, qgru


# In[ ]:


# Fake Backend (Marrakesh)
from SQGRU_j8_MNIST_FakeMarrakesh import SQGRU, qgru


# In[ ]:





# In[5]:


jax.config.update('jax_platform_name', 'cpu') # for cpu


# ## Model
# ### Hyper parameters

# In[6]:


# model parameters
I_DIM = 8 # input dim
H_DIM = 8 # hidden dim
N_LAY = 4 # number of layers
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
G_GATE = 'rot' # gate_gate
L_GATE = 'rot' # lay_gate
N_O_LAY = None # n_out_lay
O_GATE = 'rot' # out_gate
N_HO_LAY = None # n_hout_lay
HO_GATE = 'rot' # hout_gate


# In[7]:


# out_fn = None
def out_fn(x):
    return x[:,:4:2]
    # return jnp.tan(x[:,:4:2] * jnp.pi / 2)


# ### RNN Model

# In[8]:

init_fun, qgru_rnn = qgru(I_DIM,H_DIM,N_LAY, enc_layers=E_LAY, enc_reupload=E_REUL,
                                      hid_layers=H_LAY, hid_reupload=H_REUL, all_qubits_out=A_Q_OUT,
                                      pred_length=PRED_LEN, out_type=OUT_TYP, enc_n_weights_each=ENWE,
                                      hid_n_weights_each=HNWE, enc_v_gate=EVG, enc_lay_v_gate=ELVG,
                                      hid_v_gate=HVG, hid_lay_v_gate=HLVG, gate_gate=G_GATE, lay_gate=L_GATE,
                                      n_out_lay=N_O_LAY, out_gate=O_GATE, n_hout_lay=N_HO_LAY, hout_gate=HO_GATE,
                                      out_fun=out_fn)

key = jax.random.PRNGKey(SEED)
key1, key2, key3 = jax.random.split(key, num=3)
_, params = init_fun(key1, (BATCH_SIZE, 8, 8))


# ## Training

# In[9]:


@jax.jit
def loss_fn(params, inputs, targets):
    logits = qgru_rnn(params, inputs)
    # return jnp.mean(optax.softmax_cross_entropy_with_integer_labels(logits, targets))

    # 把 integer label 轉成 ±1 的 target
    # class 0 → [+1, -1]
    # class 1 → [-1, +1]
    target = jnp.where(targets[:, None] == 0,
                       jnp.array([1.0, -1.0]),
                       jnp.array([-1.0, 1.0]))
    
    return jnp.mean((logits - target) ** 2)


def result_fn(params, dataset):
    inp, tar = zip(*dataset)
    inp = jnp.array(inp)
    logits = qgru_rnn(params, inp)
    s = 0
    for r, t in zip(logits, tar):
        if jnp.argmax(r) == t:
            s += 1
    return s/len(logits)


# In[ ]:





# In[ ]:





# In[ ]:


outer_N = 7
inner_N = outer_N - 1

MNIST = MNIST_3_5_dataset(SEED)
X = MNIST.dataset
y = [_[1] for _ in X]

# 外層交叉驗證
outer_kf = StratifiedKFold(n_splits=outer_N, shuffle=True, random_state=SEED)
# 開始嵌套交叉驗證
outer_fold = 1
for train_index, test_index in outer_kf.split(X, y):
    print(f"外層 Fold {outer_fold}:")
    
    # 劃分外層訓練集和測試集
    X_train_outer = [X[i] for i in train_index]
    X_test_outer = [X[i] for i in test_index]
    y_train_outer = [y[i] for i in train_index]
    # y_test_outer = [y[i] for i in test_index]
    # y_train_outer, y_test_outer = y[train_index], y[test_index]
    
    # 內層交叉驗證
    inner_kf = StratifiedKFold(n_splits=inner_N, shuffle=True, random_state=SEED)
    inner_fold = 1
    for inner_train_idx, inner_val_idx in inner_kf.split(X_train_outer, y_train_outer):
        # 劃分內層訓練集和驗證集
        X_train_inner = [X_train_outer[i] for i in inner_train_idx]
        X_val_inner = [X_train_outer[i] for i in inner_val_idx]
        # y_train_inner = [y_train_outer[i] for i in inner_train_idx]
        # y_val_inner = [y_train_outer[i] for i in inner_val_idx]
        # y_train_inner, y_val_inner = y_train_outer[inner_train_idx], y_train_outer[inner_val_idx]

        save_name = nb_name + '_' + str(outer_fold) + '-' + str(inner_fold)
        print(save_name)
        train = TRAIN(key, init_fun, loss_fn, X_train_inner+X_val_inner, X_test_outer, result_fn, save_name)
        key, _ = jax.random.split(key)
        
        train.N_STEPS = N_STEPS
        train.BATCH_SIZE = BATCH_SIZE
        train.NUM_SEEDs = 1
        train.STD_DEV = STD_DEV
        train.REC_INTE = 10
        train.VARI_FRE = 'epoch'
        train.ini_learning_rate = 0.005
        
        train.TRAIN_VALID_TEST = jnp.array([len(X_train_inner), len(X_val_inner)])
        train.ES_THRES = None
        train.ES_LEN = 10
        train.ES_MODE = 'loss'
        train.ES_DATASET = 'valid'
        
        train.train()
        
        inner_fold += 1
        break # only once
    
    outer_fold += 1
    break # only once for test


# In[ ]:





# In[ ]:





# In[ ]:


# -------------------- END ------------------------


# In[ ]:





# In[ ]:





# In[ ]:




