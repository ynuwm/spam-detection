# -*- coding: utf-8 -*-
"""
Created on Thu Dec 27 09:46:22 2017
@author: ynuwm
"""
import time
import nltk
import math
import sqlite3
import random

import pandas as pd
import numpy as np

from sklearn import svm 
from sklearn.metrics import classification_report
from nltk.classify.scikitlearn import SklearnClassifier
from nltk.tokenize import word_tokenize 

from scipy.sparse import lil_matrix
from rescal import rescal_als

conn = sqlite3.connect('./yelpHotelData/yelpHotelData.db')
query = 'SELECT date, reviewerID, reviewContent, rating, usefulCount, coolCount, funnyCount, hotelID FROM review WHERE flagged = "Y"'
fake = pd.read_sql(query, conn)
query = 'SELECT date, reviewerID, reviewContent, rating, usefulCount, coolCount, funnyCount, hotelID FROM review WHERE flagged = "N"'
real = pd.read_sql(query, conn)
conn.close()
del query

all_reviews = pd.concat([fake,real])

reviewerID = all_reviews['reviewerID'] 
reviewerID = list(set(list(reviewerID)))

productID = all_reviews['hotelID'] 
productID = list(set(list(productID)))

#==============================================================================
# 读取评论者信息和产品信息  reviewer_info , product_info
#==============================================================================
mydb = sqlite3.connect('./yelpHotelData/yelpHotelData.db')
cursor=mydb.cursor()  
reviewer = list()

def get_raw_data(char):
    cursor.execute('Select * From ' + char)
    mail_list=[]  
    #获取所有结果       
    results = cursor.fetchall()  
    result=list(results)      
    for r in result:
        mail_list.append(r)  
    return mail_list     
    
reviewer = get_raw_data('reviewer')
product = get_raw_data('hotel')

for i in range(len(reviewer)):
    reviewer[i] = list(reviewer[i])
for i in range(len(product)):
    product[i] = list(product[i])
del i

reviewer_info = list()
product_info = list()
for reviewer_id in reviewerID:
    for line in reviewer:
        if reviewer_id == line[0]:
            reviewer_info.append(line)        
for product_id in productID:
    for line in product:
        if product_id == line[0]:
            product_info.append(line)     

#==============================================================================
# 在reviewerID中删除多余的不被识别的id
#==============================================================================
k = []       
for line in reviewer_info:
    k.append(line[0])          
j = []
for reviewer_id in reviewerID:
    if reviewer_id not in k:
        j.append(reviewer_id)
for k in j:
    reviewerID.remove(k)
    
#==============================================================================
# 变换datafr 到list
#==============================================================================
fake = np.array(fake)
fake = fake.tolist()  

real = np.array(real)
real = real.tolist()  

"""
all_reviews = np.array(all_reviews)
all_reviews = all_reviews.tolist()    
"""    

# 在fake和nonfake中删除多余的条目
for k in j:
    for line in fake:
        if k == line[1]:
            fake.remove(line)
    for line in real:
        if k == line[1]:
            real.remove(line)
            
all_reviews = fake+real
            
del j,k,line
del product,reviewer
del reviewer_id,product_id

#==============================================================================
# 表示11种关系
#==============================================================================
# No1 have reviewed
x0 = np.zeros((len(productID)+len(reviewerID),len(productID)+len(reviewerID)),dtype=np.float32)

have_reviewed = list()

for j in reviewerID:
    tmp = []
    for k in all_reviews:
        if j == k[1]:
            tmp.append(productID.index(k[7]))
    have_reviewed.append(tmp)     

for i,item in enumerate(have_reviewed):
    if item != '':
        for item2 in item:
            x0[i][len(reviewerID)+item2] = 1.0
            x0[len(reviewerID)+item2][i] = 1.0
               
del i,j,k,tmp,item,item2

# N02 rating score
x1 = np.zeros((len(productID)+len(reviewerID),len(productID)+len(reviewerID)),dtype=np.float32)

rating_score = list()

for j in reviewerID:
    tmp = []
    for k in all_reviews:
        if j == k[1]:
            tmp.append(float(k[3]))
    rating_score.append(tmp) 
 
for i in range(len(rating_score)):
    for j in range(len(rating_score[i])):  
        x1[i][len(reviewerID)+have_reviewed[i][j]] = rating_score[i][j]
        x1[len(reviewerID)+have_reviewed[i][j]][i] = rating_score[i][j]

del i,j,k,tmp

# N03 commonly reviewed products 用户A和用户B是否共同评价某产品
x2 = np.zeros((len(productID)+len(reviewerID),len(productID)+len(reviewerID)),dtype=np.float32)

def common_element(a,b):
    set_c = set(a) & set(b)
    list_c = list(set_c)   
    if not list_c == []:
        return 1
    else:
        return 0              
        
common = []
for i in range(len(have_reviewed)):
    y=[]
    for j in range(i+1,len(have_reviewed)):
        if common_element(have_reviewed[i], have_reviewed[j])==1:
            y.append(j)
    common.append(y)           
          
for i in range(len(common)):
    for j in range(len(common[i])):
        x2[i][common[i][j]] = 1.0
        x2[common[i][j]][i] = 1.0 

del i,j,y

# N04 commonly reviewed time difference
x3 = np.zeros((len(productID)+len(reviewerID),len(productID)+len(reviewerID)),dtype=np.float32)

def get_date_difference(a,b):   
    def transform_date(char):
        line = char.strip().split('/')
        if int(line[0])<=9:
            line[0]=str(0)+line[0]
        if int(line[1])<=9:
            line[1]=str(0)+line[1]   
        return line[2]+'-' + line[0] + '-' +line[1] 

    c=transform_date(a)
    d=transform_date(b)
    t_c = float(time.mktime(time.strptime(c, "%Y-%m-%d")))
    t_d = float(time.mktime(time.strptime(d, "%Y-%m-%d")))
    return (t_c-t_d)/20000000.0


def get_common_review(revierew_id1,reviewer_id2):   
    t1  = []
    t2  = []

    for item in all_reviews:
        if item[1] == revierew_id1:
            t1.append(item)
        if item[1] == reviewer_id2:
            t2.append(item)
    for item2 in t1:
        for item3 in t2:
            if item2[7] == item3[7]:
                return item2,item3

common_date_differ = []
for i in range(len(common)):
    y=[]
    for j in range(len(common[i])):
        c,d = get_common_review(reviewerID[i],reviewerID[common[i][j]])
        date_difference = get_date_difference(c[0],d[0])
        y.append(date_difference)
    common_date_differ.append(y) 


for i in range(len(common_date_differ)):
    for j in range(len(common_date_differ[i])):
        x3[i][common[i][j]] = common_date_differ[i][j]
        x3[common[i][j]][i] = common_date_differ[i][j] 

del i,j,c,d 
del date_difference

# N05 commonly reviewed rating difference
x4 = np.zeros((len(productID)+len(reviewerID),len(productID)+len(reviewerID)),dtype=np.float32)

def get_score_difference(a,b):    
    return a[3]-b[3]

def get_common_review(revierew_id1,reviewer_id2):   
    t1  = []
    t2  = []

    for item in all_reviews:
        if item[1] == revierew_id1:
            t1.append(item)
        if item[1] == reviewer_id2:
            t2.append(item)
    for item2 in t1:
        for item3 in t2:
            if item2[7] == item3[7]:
                return item2,item3

common_rating_score_difference = []
for i in range(len(common)):
    y=[]
    for j in range(len(common[i])): 
        c,d = get_common_review(reviewerID[i],reviewerID[common[i][j]])
        score_difference = get_score_difference(c,d)
        y.append(score_difference)
    common_rating_score_difference.append(y) 


for i in range(len(common_date_differ)):
    for j in range(len(common_date_differ[i])):
        x4[i][common[i][j]] = common_rating_score_difference[i][j]
        x4[common[i][j]][i] = common_rating_score_difference[i][j]  

del i,j,y,c,d,common_rating_score_difference

# N06 date difference of websites joined
x5 = np.zeros((len(productID)+len(reviewerID),len(productID)+len(reviewerID)),dtype=np.float32)
'''
January 
February
March
April
May 
June 
July
August 
September 
October 
November
December
'''
join_time_difference = []

def get_joindate_difference(charA,charB):
    lineA = charA.split(' ') 
    lineB = charB.split(' ') 
    return float(lineA[1])-float(lineB[1])

for i in range(len(reviewerID)):
    y=[]
    for j in range(i+1,len(reviewerID)):
        tmp = get_joindate_difference(reviewer_info[i][3],reviewer_info[j][3])
        y.append(tmp)
    join_time_difference.append(y)  
    
for i in range(len(join_time_difference)):
    for j in range(len(join_time_difference[i])):
        x5[i][i+j+1] = join_time_difference[i][j]       
        x5[i+j+1][i] = join_time_difference[i][j]  

del i,j,y,tmp,join_time_difference

# NO7 Average rating difference
x6 = np.zeros((len(productID)+len(reviewerID),len(productID)+len(reviewerID)),dtype=np.float32)

p = np.zeros([5019,5019])

def average_list(l):
    s=0
    for i in l:
        s=s+i
    return s/len(l) 

sum_s=[]
for j,item in enumerate(have_reviewed):
    score=[] 
    for i,item2 in enumerate(item):       
        for rev in all_reviews:
            if rev[-1]==productID[item2] and rev[1]==reviewerID[j]:
                score.append(rev[3])                 
    sum_s.append(score)   

for j,item in enumerate(sum_s):
    sum_s[j] = average_list(item)


for i in range(len(sum_s)):
    for j in range(i+1,len(sum_s)):
        p[i][j] = sum_s[i]-sum_s[j]
        p[j][i] = sum_s[i]-sum_s[j]
    
q = np.zeros([5019,72])     
r = np.hstack((p,q)) 

t = np.zeros([72,5091])       

x6 = np.vstack((r,t))  

del r,q,t,sum_s,i,j,item,item2,p



# NO8 friend count difference
x7 = np.zeros((len(productID)+len(reviewerID),len(productID)+len(reviewerID)),dtype=np.float32)

friend_count_difference = []


for i in range(len(reviewerID)):
    y=[]
    for j in range(i+1,len(reviewerID)):
        tmp = reviewer_info[i][4]-reviewer_info[j][4]
        y.append(tmp)
    friend_count_difference.append(y)  
    
for i in range(len(friend_count_difference)):
    for j in range(len(friend_count_difference[i])):
        x7[i][i++1+j] = friend_count_difference[i][j]
        x7[j+i+1][i] = friend_count_difference[i][j]  

del i,j,tmp

# NO9 Have the same location or not
x8 = np.zeros((len(productID)+len(reviewerID),len(productID)+len(reviewerID)),dtype=np.float32)

q = np.zeros([5019,72])

for item in all_reviews:
    for k,re_info in enumerate(reviewer_info):
        if re_info[0] == item[1]:
            i = k
            
    for k,pr_info in enumerate(product_info):
        if pr_info[0] == item[7]:
            j = k        
            
    str = product_info[j][2]
    line = (str.split('-')[-1]).strip()
    if reviewer_info[i][2] == line:   
        q[i][j] = 1 

for i in range(q.shape[0]):
    for j in range(q.shape[1]):
        x8[i][j+5019] = q[i][j]
        x8[j+5019][i] = q[i][j]

del pr_info
# N10 common reviewers
x9 = np.zeros((len(productID)+len(reviewerID),len(productID)+len(reviewerID)),dtype=np.float32)

common_reviewers = []
p = np.zeros([72,72])

for pro_id in productID:
    temp = []
    for review in all_reviews:
        if review[7] == pro_id:
            temp.append(review[1])
    common_reviewers.append(temp)  

for i in range(len(common_reviewers)):
    y=[]
    for j in range(i+1,len(common_reviewers)):
        set_c = set(common_reviewers[i]) & set(common_reviewers[j])
        p[i][j] = len(set_c)       
        p[j][i] = len(set_c)  

for i in range(len(p)):
    for j in range(i,len(p[i])):
        x9[5019+i][j+5019] = p[i][j]
        x9[j+5019][i+5019] = p[i][j]         
        
del i,j,temp,set_c,p,pro_id

# N11 review count difference
x10 = np.zeros((len(productID)+len(reviewerID),len(productID)+len(reviewerID)),dtype=np.float32)

product_review_difference = []

for i in range(len(productID)):
    y=[]
    for j in range(i+1,len(productID)):
        tmp = product_info[i][3]-product_info[j][3]
        y.append(tmp)
    product_review_difference.append(y)  
    
for i in range(len(product_review_difference)):
    for j in range(len(product_review_difference[i])):
        x10[5019+i][i+1+j+5019] = product_review_difference[i][j]
        x10[j+i+1+5019][i+5019] = product_review_difference[i][j]  

del i,j,tmp

#==============================================================================
# sigmoid 正则化
#==============================================================================
def sig_func(x):
    return 1.0/(1+math.exp(-x))
    
for i in range(5091):
    for j in range(5091):
        x7[i,j] = sig_func(x7[i,j]) 
        
del i,j,line,item
#==============================================================================
# 拼接成连接矩阵
#==============================================================================
T = np.zeros((5091,5091,11))    

T[:,:,0] = x0
T[:,:,1] = x1
T[:,:,2] = x2
T[:,:,3] = x3
T[:,:,4] = x4
del x0,x1,x2,x3,x4

T[:,:,5] = x5
T[:,:,6] = x6
T[:,:,7] = x7
T[:,:,8] = x8
T[:,:,9] = x9
T[:,:,10] = x10
del x5,x6,x7,x8,x9,x10

#==============================================================================
# 张量分解
#==============================================================================
X = [lil_matrix(T[:, :, k]) for k in range(T.shape[2])]
# Decompose tensor using RESCAL-ALS
A, R, fit, itr, exectimes = rescal_als(X, 300, init='nvecs', lambda_A=10, lambda_R=10)
# A's transpose
B = A.transpose()
#保存数组
np.save("./array/array_T.npy",T)
np.save("./array/array_A.npy",A)
np.save("./array/array_B.npy",B)

np.save("./array/array_reviewerID",reviewerID)
np.save("./array/array_reviewer_info",reviewer_info)

np.save("./array/array_productID",productID)
np.save("./array/array_product_info",product_info)

np.save("./array/array_fake",fake)
np.save("./array/array_real",real)
np.save("./array/array_all_reviews",all_reviews)

del A,R,fit,itr,exectimes
#==============================================================================
# 训练  训练集 770(fake) + 770(real)
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
# 测试     
#==============================================================================
test_fake = fake[700:]
test_real = [real[j] for j in index[800:877]]                
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
        






'''
from sklearn.metrics import classification_report

X = [[0, 2], [1, 1], [1, 3],[2,4],[7,9],[10,5]]  # training samples   
y = [0, 1, 1,0,1,1]  # training target  
clf = svm.SVC()  # class   
clf.fit(X, y)  # training the svc model  

x_test = [[1,2],[0,2],[1,4]] 
y_gold = [1,0,0]

target_names=['class0','class1']

print(classification_report(y_gold,y_pre,target_names=target_names))
'''       
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
# 测试     
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


