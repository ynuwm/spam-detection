# -*- coding: utf-8 -*-
"""
Created on Fri Dec 28 16:22:18 2017
@author: ynuwm
"""
import random
import numpy as np


from sklearn import svm 
from sklearn.metrics import classification_report


B = np.load("./array/array_B.npy")

reviewerID = np.load("./array/array_reviewerID.npy")
reviewer_info = np.load("./array/array_reviewer_info.npy")

productID = np.load("./array/array_productID.npy")
product_info = np.load("./array/array_product_info.npy")

fake = np.load("./array/array_fake.npy")
real = np.load("./array/array_real.npy")

all_reviews = np.load("./array/array_all_reviews.npy")










#==============================================================================
# 50：50  训练集 770(fake) + 770(real) 测试集77+77
#==============================================================================
shuffle_indices = np.random.permutation(np.arange(5072))
index = list(shuffle_indices)

train_fake = fake[:700]
train_real = [real[j] for j in index[:700]] 
train = train_fake + train_real

y_train = [0 for i in range(700)]+[1 for j in range(700)]
x_train = []  

for i,item in enumerate(train):
    tmp = reviewerID.index(item[1])
    column = [x[tmp] for x in B]
    x_train.append(column)

x_train = np.array(x_train)
y_train = np.array(y_train) 

# shuffle data
sf_indices = np.random.permutation(np.arange(len(y_train)))
x_shuffled = x_train[sf_indices]
y_shuffled = y_train[sf_indices] 

clf = svm.SVC(kernel='linear')  # class   
clf.fit(x_shuffled, y_shuffled)  # training the svc model  
#==============================================================================
# predict    
#==============================================================================
test_fake = fake[700:]
test_real = [real[j] for j in index[700:777]]                
test = test_fake + test_real

y_gold = [0 for i in range(77)]+[1 for j in range(77)]        
x_test = []        
        
for i,item in enumerate(test):
    tmp = reviewerID.index(item[1])
    column = [x[tmp] for x in B]
    x_test.append(column)

x_test = np.array(x_test)
y_gold = np.array(y_gold)         
     
sf_indices = np.random.permutation(np.arange(len(y_gold)))
x_test_shuffled = x_test[sf_indices]
y_test_shuffled = y_gold[sf_indices] 



   
y_pre = clf.predict(x_test_shuffled)        
print(classification_report(y_test_shuffled,y_pre))   
        
np.save("./array/y_pre",y_pre)
np.save("./array/y_test_shuffled",y_test_shuffled)        
        

#==============================================================================
# ND  训练集700+700   测试集77+500
#==============================================================================      
shuffle_indices = np.random.permutation(np.arange(5072))
index = list(shuffle_indices)

train_fake = fake[:700]
train_real = [real[j] for j in index[:700]] 
train = train_fake + train_real

y_train = [0 for i in range(700)]+[1 for j in range(700)]
x_train = []  

for i,item in enumerate(train):
    tmp = reviewerID.index(item[1])
    column = [x[tmp] for x in B]
    x_train.append(column)

x_train = np.array(x_train)
y_train = np.array(y_train) 

# shuffle data
sf_indices = np.random.permutation(np.arange(len(y_train)))
x_shuffled = x_train[sf_indices]
y_shuffled = y_train[sf_indices] 

clf = svm.SVC(kernel='linear')  # class   
clf.fit(x_shuffled, y_shuffled)  # training the svc model  
#==============================================================================
# predict     
#==============================================================================
test_fake = fake[700:]
test_real = [real[j] for j in index[700:1200]]                
test = test_fake + test_real

y_gold = [0 for i in range(77)]+[1 for j in range(500)]        
x_test = []        
        
for i,item in enumerate(test):
    tmp = reviewerID.index(item[1])
    column = [x[tmp] for x in B]
    x_test.append(column)

x_test = np.array(x_test)
y_gold = np.array(y_gold)         
     
sf_indices = np.random.permutation(np.arange(len(y_gold)))
x_test_shuffled = x_test[sf_indices]
y_test_shuffled = y_gold[sf_indices] 
  
y_pre = clf.predict(x_test_shuffled)        
print(classification_report(y_test_shuffled,y_pre))   
        
np.save("./array/y_pre",y_pre)
np.save("./array/y_test_shuffled",y_test_shuffled) 
