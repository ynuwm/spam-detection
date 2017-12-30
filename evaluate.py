# -*- coding: utf-8 -*-
"""
Created on Fri Dec 29 13:36:18 2017
@author: ynuwm
"""
import numpy as np
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

y_pre= np.load('./array/y_pre.npy')
y_test_shuffled = np.load(('./array/y_test_shuffled.npy'))
      
print(classification_report(y_test_shuffled,y_pre))   
print(accuracy_score(y_test_shuffled,y_pre))

  
