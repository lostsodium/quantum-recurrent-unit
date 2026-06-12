#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import random

from sklearn.datasets import load_breast_cancer


# In[2]:


def normalize1D(arr):
    x1 = min(arr)
    x2 = max(arr)
    dx = x2 - x1
    return (arr - x1) /dx

def normalize2D(arr, idx='column'):
    arr = np.array(arr)
    if idx == 'column':
        arr = arr.transpose()
        
    for i in range(arr.shape[0]):
        arr[i] = normalize1D(arr[i])
        
    if idx == 'column':
        arr = arr.transpose()
        
    return arr

def split_dataset(dataset, ratios):
        total = sum(ratios)
        n = len(dataset)
        splits = [int(r / total * n) for r in ratios]
        splits[-1] = n - sum(splits[:-1])
        split_points = [sum(splits[:i]) for i in range(len(splits) + 1)]
        return [dataset[split_points[i]:split_points[i + 1]] for i in range(len(ratios))]


# In[3]:


class WDBC_dataset:
    def __init__(self, seed=None):
        self.seed = seed
        self.org_data = load_breast_cancer()
        self.data_input = normalize2D(self.org_data['data'])
        self.data_target = self.org_data['target'].astype(float)
        self.all_dataset = [[inp.reshape([30,1]), tar] for inp, tar in zip(self.data_input, self.data_target)]
        self.b_dataset = [_ for _ in self.all_dataset if _[1] == 1.0]
        self.m_dataset = [_ for _ in self.all_dataset if _[1] == 0.0]
    def split(self, ratios):
        if self.seed != None:
            random.seed(self.seed)
        random.shuffle(self.b_dataset)
        random.shuffle(self.m_dataset)
        b_splits = split_dataset(self.b_dataset, ratios)
        m_splits = split_dataset(self.m_dataset, ratios)

        splits = [m+b for m, b in zip(m_splits, b_splits)]
        [random.shuffle(_) for _ in splits]
        self.splits = splits
        return splits
        
