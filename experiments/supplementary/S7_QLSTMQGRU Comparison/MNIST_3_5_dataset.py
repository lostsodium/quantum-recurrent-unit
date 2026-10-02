#!/usr/bin/env python
# coding: utf-8

# In[1]:


import jax
import jax.numpy as jnp
import pickle
import random


# In[3]:


class MNIST_3_5_dataset:
    def __init__(self, seed=None):
        with open('mnist_pixels_3-5_8x8.pkl', 'rb') as f:   # 0~6130: 3 6131~11551: 5
            self.org_train_dataset, self.org_test_dataset = pickle.load(f)
    
        self.org_train_images, self.org_train_labels = zip(*self.org_train_dataset)
        self.org_train_images = jnp.array(self.org_train_images)
        self.org_train_labels = jnp.array(self.org_train_labels)
        
        self.org_test_images, self.org_test_labels = zip(*self.org_test_dataset)
        self.org_test_images = jnp.array(self.org_test_images)
        self.org_test_labels = jnp.array(self.org_test_labels)
        
        v1 = min(self.org_train_images.min(), self.org_test_images.min())
        v2 = max(self.org_train_images.max(), self.org_test_images.max())
        
        self.org_train_images_N = (self.org_train_images - v1) / (v2 - v1)
        self.org_test_images_N = (self.org_test_images - v1) / (v2 - v1)
    
        self.train_dataset = list(zip(self.org_train_images_N, self.org_train_labels))
        self.test_dataset = list(zip(self.org_test_images_N, self.org_test_labels))
        self.dataset = self.train_dataset + self.test_dataset

        if seed != None:
            random.seed(seed)
            random.shuffle(self.dataset)


# In[ ]:




