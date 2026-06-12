#!/usr/bin/env python
# coding: utf-8

# # TRAIN class

# In[1]:


import jax
import jax.numpy as jnp
from jax.example_libraries import optimizers
from jax import value_and_grad

import ipynbname
import optax
import time
import copy
import pickle
import os.path
import shutil

from collections import deque


# In[ ]:


class EarlyStoppingChecker:
    def __init__(self, maxlen, mode='loss', threshold=0.01):
        # mode = 'loss' & threshold != None
        #  stop if the loss decreases and changes within threshold.
        #  usually for training loss
        # mode = 'loss' & threshold = None
        #  stop if the loss no longer decreases or remains unchanged.
        #  usually for validation loss
        # mode = 'acc'
        #  stop if the accuracy no longer improves or remains unchanged.
        #  usually for validation accuracy
        self.maxlen = maxlen
        self.data = deque(maxlen=maxlen)
        self.threshold = threshold
        self.mode = mode

    def add(self, value):
        if self.mode == 'loss' and self.threshold == None:
            value *= -1
        self.data.append(value)

    def is_increasing(self):
        return all(self.data[i] <= self.data[i + 1] for i in range(len(self.data) - 1))

    def is_decreasing(self):
        return all(self.data[i] >= self.data[i + 1] for i in range(len(self.data) - 1))

    def is_change_within_range(self):
        return all(
            abs(self.data[i + 1] - self.data[i]) / abs(self.data[i]) <= self.threshold
            for i in range(len(self.data) - 1)
            if self.data[i] != 0  # 防止除以零
        )

    def reset(self):
        self.data.clear()

    def check(self):
        # to stop if return True
        if len(self.data) == self.maxlen:
            if self.mode == 'loss' and self.threshold != None:
                r = self.is_decreasing() and self.is_change_within_range()
            else:
                oldest = self.data[0]
                r1 = all(elem < oldest for elem in list(self.data)[1:])
                r2 = all(elem == oldest for elem in list(self.data)[1:])
                r = r1 or r2
        else:
            r = False
            
        return r


# In[ ]:


class TRAIN:
    def __init__(self, key, ini_function, loss_function, dataset, test_dataset=None, result_function=None, save_name=None):
        # initial settings
        ## save file name
        if save_name == None:
            ## using self file name
            self.nb_name = ipynbname.name() # for jupyter notebook
            # self.nb_name = os.path.basename(__file__)[0:-3] # for python script
        else:
            self.nb_name = save_name
        
        ## default settings
        self.N_STEPS = 1000
        self.BATCH_SIZE = 30
        self.TRAIN_VALID_TEST = jnp.array([4, 1])  # 4/5 for training, 1/5 for validation
        # self.TRAIN_VALID_TEST = jnp.array([4, 1, 1])  # 4/6 for training, 1/6 for validation, 1/6 for test (ignore test_dataset)
        
        self.NUM_SEEDs = 0
        self.NOISE_TYPE = 'normal'
        self.STD_DEV = 0.0
        self.REC_INTE = 10
        self.VARI_FRE = 'epoch'
        self.ini_learning_rate = 1e-3
        self.min_learning_rate = 1e-64
        self.optimizer_model = optax.adam

        self.ES_LEN = 3
        self.ES_MODE = 'loss'
        self.ES_DATASET = 'train'
        self.ES_THRES = 1e-2
        # self.min_lr_ratio = 1e-2
        
        ## parameters, datasets
        self.ini_function = ini_function
        key1, self.key2, key3 = jax.random.split(key, num=3)
        input_shape = (self.BATCH_SIZE,) + dataset[0][0].shape
        _, self.params = self.ini_function(key1, input_shape)
        self.loss_fn = loss_function
        # self.dataset = dataset
        self.org_dataset = dataset
        self.test_dataset = test_dataset # if length of self.TRAIN_VALID_TEST >= 3, this will be ignored.
        self.result_fn = result_function
        
        ## training recordings
        self.lossList = []
        self.vReltList = []
        self.minLoss = 1000
        self.vminLoss = 1000
        self.vlocLoss = 1000
        self.vmaxAcc = 0.0
        self.vlocAcc = 0.0
        self.best_params = self.params
        self.vbest_params = self.params
        self.loc_params_list = []
        self.vloc_params_list = []
        self.reset_steps = []
        self.acc_results = []
        self.all_loss_list = []
        self.all_vloss_list = []
        
        self.update_lr = 0
        self.locLossList = [1000,1000,1000]
        self.locLoss = 1000
        self.loc_params = self.params
        self.vloc_params = self.params
        
    def __load_previous_state__(self):
        # load previous state if it exists
        self.cur_name = self.nb_name+'_current.pkl'
        if os.path.isfile(self.cur_name):
            with open(self.cur_name, 'rb') as f:
                (self.locLossList, self.locLoss, self.vlocLoss, self.lossList, self.vReltList, self.loss,
                 self.minLoss, self.vminLoss, self.vlocAcc, self.vmaxAcc, self.params, self.loc_params,
                 self.vloc_params, self.best_params, self.vbest_params, self.loc_params_list,
                 self.vloc_params_list, self.reset_steps, self.acc_results, self.i, self.key2,
                 self.train_dataset, self.valid_dataset, self.test_dataset, self.opt_state, self.update_lr,
                 self.learning_rate, self.all_loss_list, self.all_vloss_list, self.dataset_indices,
                 self.ES_checker) = pickle.load(f)
                
            self.optimizer = self.optimizer_model(self.learning_rate)
            
            if len(self.acc_results) > 0:
                print('No.', len(self.loc_params_list)-1, 'acc.', self.acc_results[-1])
            print('Loading previous state successfully')
        else:
            self.i = 0
            print('No previous state recorded.')

    def __save_current_state__(self):
        # Save current state       
        if os.path.isfile(self.cur_name):
            shutil.copyfile('./'+self.cur_name, './'+self.nb_name+'_current_bak.pkl')
        with open(self.nb_name+'_current.pkl', 'wb') as f:
            pickle.dump((self.locLossList, self.locLoss, self.vlocLoss, self.lossList, self.vReltList,
                         self.loss, self.minLoss, self.vminLoss, self.vlocAcc, self.vmaxAcc, self.params,
                         self.loc_params, self.vloc_params, self.best_params, self.vbest_params,
                         self.loc_params_list, self.vloc_params_list, self.reset_steps, self.acc_results,
                         self.i, self.key2, self.train_dataset, self.valid_dataset, self.test_dataset,
                         self.opt_state, self.update_lr, self.learning_rate, self.all_loss_list,
                         self.all_vloss_list, self.dataset_indices, self.ES_checker), f)

    def __create_dataloader__(self, key, dataset, batch_size):
        if key != None:
            # shuffle the dataset
            dataset = copy.deepcopy(dataset)
            indices = list(range(len(dataset)))
            shuffled_indices = jax.random.permutation(key, jnp.array(indices))
            dataset = [dataset[i] for i in shuffled_indices]
        m = int(len(dataset) / batch_size)
        mod = len(dataset) % batch_size
        loader = []
        for i in range(m):
            train_sub = [j[0] for j in dataset[i*batch_size:(i+1)*batch_size]]
            target_sub = [j[1] for j in dataset[i*batch_size:(i+1)*batch_size]]
            loader.append((jnp.array(train_sub), jnp.array(target_sub)))
            
        if mod > 0:
            train_sub = [j[0] for j in dataset[m*batch_size:]]
            target_sub = [j[1] for j in dataset[m*batch_size:]]
            loader.append((jnp.array(train_sub), jnp.array(target_sub)))
        return loader

    def __ck_lr__(self, old_lr, new_lr):
        d_lr = old_lr - new_lr
        return d_lr >= 0 and d_lr/new_lr < self.min_lr_ratio

    def __update__(self, params, inputs, targets, opt_state):
        loss, grads = value_and_grad(self.loss_fn)(params, inputs, targets)
        updates, opt_state = self.optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss, grads
    
    def train(self):
        # training, validation, and test dataset
        dataset = self.org_dataset
        total_len = len(dataset)
        self.dataset_indices = [list(range(total_len))]
        n = self.TRAIN_VALID_TEST.sum()
        train_len = int(total_len * self.TRAIN_VALID_TEST[0] / n)
        self.train_dataset = dataset[:train_len]
        if len(self.TRAIN_VALID_TEST) < 3:
            self.valid_dataset = dataset[train_len:]
        else:
            valid_len = int(total_len * self.TRAIN_VALID_TEST[1] / n)
            self.valid_dataset = dataset[train_len:train_len+valid_len]
            self.test_dataset = dataset[train_len+valid_len:]

        # optimizer, learning rate
        self.learning_rate = self.ini_learning_rate
        self.optimizer = self.optimizer_model(self.learning_rate)
        self.opt_state = self.optimizer.init(self.params)

        # early stopping settings
        self.ES_checker = EarlyStoppingChecker(self.ES_LEN, self.ES_MODE, self.ES_THRES)
        
        # load previous state
        self.__load_previous_state__()
        
        # validation loader
        if len(self.valid_dataset) > 0:
            self.vdata_loader = self.__create_dataloader__(None, self.valid_dataset, len(self.valid_dataset))
        else:
            self.vdata_loader = None
            
        vloss = 1000
        vacc = 0.0
        st_time = time.time()
        while self.i < self.N_STEPS:
            
            if self.update_lr != 0:
                if self.update_lr < 0:
                    self.params = self.loc_params
                else:
                    self.locLossList = [1000,1000,1000]
                    self.ES_checker.reset()
                    self.locLoss = 1000
                    self.vlocLoss = 1000
                    self.loc_params_list.append(self.loc_params)
                    self.vloc_params_list.append(self.vloc_params)
                    self.reset_steps.append(self.i)

                    print('Reset times:', len(self.reset_steps))
                    
                    # record local best & validted best results
                    if self.result_fn != None:
                        best_train_r = self.result_fn(self.loc_params, self.train_dataset)
                        vbest_train_r = self.result_fn(self.vloc_params, self.train_dataset)
                        if self.test_dataset != None:
                            best_test_r = self.result_fn(self.loc_params, self.test_dataset)
                            vbest_test_r = self.result_fn(self.vloc_params, self.test_dataset)
                        else:
                            best_test_r = vbest_test_r = 'none'
                        
                        self.acc_results.append([best_train_r, best_test_r, vbest_train_r, vbest_test_r])
                        print('No.', len(self.loc_params_list)-1, 'acc.', best_train_r, best_test_r, vbest_train_r, vbest_test_r)
                    
                    # restart from a new seed
                    if self.NUM_SEEDs >= 1:
                        # stop when collect X local best parameters
                        # if len(self.loc_params_list) >= self.NUM_SEEDs:
                        #     break

                        # new random key
                        key1, self.key2 = jax.random.split(self.key2, num=2)
                        # reset dataset
                        dataset = self.org_dataset
                        if len(self.TRAIN_VALID_TEST) < 3:
                            # dataset = self.train_dataset + self.valid_dataset
                            indices = list(range(len(dataset)))
                            shuffled_indices = jax.random.permutation(key1, jnp.array(indices))
                            dataset = [dataset[i] for i in shuffled_indices]
                            self.train_dataset = dataset[:train_len]
                            self.valid_dataset = dataset[train_len:]
                        else:
                            # dataset = self.train_dataset + self.valid_dataset + self.test_dataset
                            indices = list(range(len(dataset)))
                            shuffled_indices = jax.random.permutation(key1, jnp.array(indices))
                            dataset = [dataset[i] for i in shuffled_indices]
                            self.train_dataset = dataset[:train_len]
                            self.valid_dataset = dataset[train_len:train_len+valid_len]
                            self.test_dataset = dataset[train_len+valid_len:]

                        self.dataset_indices.append(shuffled_indices)
                        if len(self.valid_dataset) > 0:
                            self.vdata_loader = self.__create_dataloader__(None, self.valid_dataset, len(self.valid_dataset))
                        else:
                            self.vdata_loader = None

                        # reset parameters
                        input_shape = (self.BATCH_SIZE,) + dataset[0][0].shape
                        _, self.params = self.ini_function(key1, input_shape)
                        self.loc_params = self.params
                    
                self.update_lr = 0
                self.optimizer = self.optimizer_model(self.learning_rate)
                self.opt_state = self.optimizer.init(self.params)
                
                print('learning rate:', self.learning_rate)
            
            # stop when collect X local best parameters
            if (self.NUM_SEEDs >= 1) and (len(self.loc_params_list) >= self.NUM_SEEDs):
                break
            # ---------------- stop (end)
            
            avg_loss = 0
            n = 0
            
            key1, self.key2 = jax.random.split(self.key2, num=2)
            self.train_loader = self.__create_dataloader__(key1, self.train_dataset, self.BATCH_SIZE)
            
            for seq, targets in self.train_loader:
                pre_params = self.params
                
                # validation, every batch
                if self.VARI_FRE == 'batch':
                    if self.vdata_loader != None:
                        if self.ES_MODE == 'loss':
                            vloss = self.loss_fn(pre_params, self.vdata_loader[0][0], self.vdata_loader[0][1])
                            if vloss < self.vminLoss:
                                self.vbest_params = pre_params
                                self.vminLoss = vloss
                            if vloss < self.vlocLoss:
                                self.vloc_params = pre_params
                                self.vlocLoss = vloss
                        else:
                            vacc = self.result_fn(pre_params, self.valid_dataset)
                            if (vacc >= self.vmaxAcc) or (vacc >= self.vlocAcc):
                                vloss = self.loss_fn(pre_params, self.vdata_loader[0][0], self.vdata_loader[0][1])
                            if (vacc >= self.vmaxAcc) and (vloss < self.vminLoss):
                                self.vbest_params = pre_params
                                self.vmaxAcc = vacc
                                self.vminLoss = vloss
                            if (vacc >= self.vlocAcc) and (vloss < self.vlocLoss):
                                self.vloc_params = pre_params
                                self.vlocAcc = vacc
                                self.vlocLoss = vloss

                            
                            # if vacc >= self.vmaxAcc:
                            #     self.vbest_params = pre_params
                            #     self.vmaxAcc = vacc
                            # if vacc >= self.vlocAcc:
                            #     self.vloc_params = pre_params
                            #     self.vlocAcc = vacc

                    
                    else:
                        vloss = 'none'
                        vacc = 'none'
                # ------------------------ validation, every batch (end)
        
                # add noise to inputs 2024/10/08 (modifid 2024/10/12)
                if self.STD_DEV > 0:
                    if self.NOISE_TYPE == 'uniform':
                        noise = jax.random.uniform(self.key2, shape=seq.shape, dtype=seq.dtype, minval=-1.0, maxval=1.0) * self.STD_DEV
                    else:
                        noise = jax.random.normal(self.key2, shape=seq.shape, dtype=seq.dtype) * self.STD_DEV
                    seq2 = seq + noise
                else:
                    seq2 = seq
                # -------------- add noise (end)
                
                # update parameters
                self.params, self.opt_state, self.loss, self.grads = self.__update__(self.params, seq2, targets, self.opt_state)
                if self.ES_MODE == 'loss':
                    print('{}, {}: {} {}  {} sec      '.format(self.i, n, self.loss, vloss, time.time()-st_time), end='\r')
                else:
                    print('{}, {}: {} {}  {} sec      '.format(self.i, n, self.loss, vacc, time.time()-st_time), end='\r')
                
                avg_loss += self.loss
                n += 1
            # end for loop
                
            # validation, every epoch
            if self.VARI_FRE == 'epoch':
                if self.vdata_loader != None:
                    if self.ES_MODE == 'loss':
                        vloss = self.loss_fn(pre_params, self.vdata_loader[0][0], self.vdata_loader[0][1])
                        if vloss < self.vminLoss:
                            self.vbest_params = pre_params
                            self.vminLoss = vloss
                        if vloss < self.vlocLoss:
                            self.vloc_params = pre_params
                            self.vlocLoss = vloss
                    else:
                        vacc = self.result_fn(pre_params, self.valid_dataset)
                        if (vacc >= self.vmaxAcc) or (vacc >= self.vlocAcc):
                            vloss = self.loss_fn(pre_params, self.vdata_loader[0][0], self.vdata_loader[0][1])
                        if (vacc >= self.vmaxAcc) and (vloss < self.vminLoss):
                            self.vbest_params = pre_params
                            self.vmaxAcc = vacc
                            self.vminLoss = vloss
                        if (vacc >= self.vlocAcc) and (vloss < self.vlocLoss):
                            self.vloc_params = pre_params
                            self.vlocAcc = vacc
                            self.vlocLoss = vloss
                        
                        # if vacc >= self.vmaxAcc:
                        #     self.vbest_params = pre_params
                        #     self.vmaxAcc = vacc
                        # if vacc >= self.vlocAcc:
                        #     self.vloc_params = pre_params
                        #     self.vlocAcc = vacc

                
                else:
                    vloss = 'none'
                    vacc = 'none'
            # -------------- validation, every epoch (end)

            # record histories of avg_loss, vloss 
            self.i += 1
            avg_loss /= n
            self.all_loss_list.append(avg_loss)
            self.all_vloss_list.append(vloss)
                
            if avg_loss < self.minLoss:
                self.best_params = pre_params
                self.minLoss = avg_loss
            if avg_loss < self.locLoss:
                self.loc_params = pre_params
                self.locLoss = avg_loss
            if (self.i) % self.REC_INTE == 0:
                # early stopping data
                if self.ES_MODE == 'loss':
                    if self.ES_DATASET == 'train':
                        self.ES_checker.add(avg_loss)
                    else:
                        self.ES_checker.add(vloss)
                else:
                    if self.ES_DATASET == 'train':
                        tmp_acc = self.result_fn(pre_params, self.train_dataset)
                    else:
                        # tmp_acc = self.result_fn(pre_params, self.valid_dataset)
                        tmp_acc = vacc
                    self.ES_checker.add(tmp_acc)
                
                # update learning rate
                if avg_loss > self.locLossList[0] and self.locLossList[1] > self.locLossList[0] and self.locLossList[2] > self.locLossList[0]:
                    self.learning_rate /= 2
                    self.update_lr = -1
                    # check learning rate
                    if self.learning_rate < self.min_learning_rate:
                        self.learning_rate = self.ini_learning_rate
                        self.update_lr = 1
                elif self.ES_checker.check():
                    self.learning_rate = self.ini_learning_rate
                    self.update_lr = 1
                
                self.locLossList[0] = self.locLossList[1]
                self.locLossList[1] = self.locLossList[2]
                self.locLossList[2] = avg_loss
                
                self.lossList.append(avg_loss)
                
                if self.ES_MODE == 'loss':
                    self.vReltList.append(vloss)
                    print('{}, {}: {} {}  {} sec      '.format(self.i-1, n, avg_loss, vloss, time.time()-st_time))
                else:
                    self.vReltList.append(vacc)
                    print('{}, {}: {} {}  {} sec      '.format(self.i-1, n, avg_loss, vacc, time.time()-st_time))
                
                # Save current state
                self.__save_current_state__()

        # save final state
        self.__save_current_state__()

    @staticmethod
    def help():
        print('Methods:')
        print('__init__(self, key, ini_function, loss_function, dataset, test_dataset=None, result_function=None, save_name=None)')
        print('   ini_function(key, input_shape)')
        print('   loss_function(params, inputs, targets)')
        print('   dataset: for training, validation, (and test if length of TRAIN_VALID_TEST >= 3.)')
        print('   test_dataset: for test (it will be ignored if length of TRAIN_VALID_TEST >= 3.)')
        print('   result_function(params, dataset)')
        print('train()')
        print('Properties:')
        print('N_STEPS: Maximum number of total steps (episodes). (default: 1000)')
        print('BATCH_SIZE: Batch size. (default: 30)')
        print('NUM_SEEDs: >0: Maximum reset times, reset random seed (shuffle the dataset) and learning rate. 0: Unlimited and only reset learning rate. (default: 0)')
        print('NOISE_TYPE: \'normal\' or \'uniform\' (default: \'normal\')')
        print('STD_DEV: Standand deviation of noise added to inputs. (default: 0.0)')
        print('REC_INTE: Interval steps for recording. (default: 10)')
        print('VARI_FRE: Validation frequency. \'epoch\' or \'batch\' (default: \'epoch\')')
        print('TRAIN_VALID_TEST: The ratio of size of training dataset, validation dataset, and test dataset. (default: 4:1:0)')
        print('   eg. jnp.array([4, 1]): training:validation = 4:1')
        print('   eg. jnp.array([4, 1, 1]): training:validation:test = 4:1:1 and ignore the parameter test_dataset.')
        print('ini_learning_rate: Initial learning rate. (default: 1e-3)')
        print('min_learning_rate: The minmum learning rate. (default: 1e-64)')
        print('optimizer_model: Optimizer. (default: Adam)')
        # Early stopping
        print('Early stopping settings:')
        print('ES_LEN: Length of early stopping data. (default: 3)')
        print('ES_MODE: \'loss\' or \'acc\' (default: \'loss\')')
        print('ES_DATASET: \'train\' or \'valid\' (default: \'valid\')')
        print('ES_THRES: Threshold of loss-mode early stopping. (default: 0.01)')


# In[ ]:




