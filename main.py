#==============================================================================
#  net start MySQL57 测试连接
#
#    import pymysql  
#    conn = pymysql.connect(host='127.0.0.1', port=3306, user='root', passwd='123456', db='mysql') #这里写上面设置的密码  
#    cursor = conn.cursor()  
#    cursor.execute("SELECT VERSION()")  
#    row = cursor.fetchone()  
#    print("MySQL server version:", row[0])  
#    cursor.close()  
#    conn.close() 
import sqlite3  
import numpy as np

#==============================================================================
# Get data
#==============================================================================
mydb = sqlite3.connect('./yelpHotelData/yelpHotelData.db')
cursor=mydb.cursor()  

cursor.execute("PRAGMA table_info(review)")
print(cursor.fetchall())                            

review = list()   
reviewer = list()
hotel = list()

def get_raw_data(char):
    cursor.execute('Select * From ' + char)
    mail_list=[]  
    #获取所有结果       
    results = cursor.fetchall()  
    result=list(results)      
    for r in result:
        mail_list.append(r)  
    return mail_list     

review = get_raw_data('review')
reviewer = get_raw_data('reviewer')
hotel = get_raw_data('hotel')

reviewerID =list()
reviewerID_in_review = list()

for i in range(len(reviewer)):
    reviewer[i] = list(reviewer[i])
    reviewerID.append(reviewer[i][0])
for i in range(len(review)):
    review[i] = list(review[i])
    reviewerID_in_review.append(review[i][2])
for i in range(len(hotel)):
    hotel[i] = list(hotel[i])
    
#==============================================================================
# 取出评论    
#==============================================================================
def unique_index(L,e):
    return [i for (i,j) in enumerate(L) if j == e]                              
  
tmp_list = list()
only_index = list()
for id in reviewerID:                      
   tmp_list.append(unique_index(reviewerID_in_review,id) )
 
for item in tmp_list:
    for t in range(len(item)):
        only_index.append(item[t])

del item        
index = list({}.fromkeys(only_index).keys())                       

def get_fake_or_nonfake_revier(char):
    k = list()         
    for j in index:
        if review[j][8] == char:
            k.append(j)
    return k
    
"""
在review中
取得fake 评论777条，
取得非fake 评论5072条，
"""   
# reviewID,reviewerID,reviewContent,rating,flagged,# hotelID 
fake = get_fake_or_nonfake_revier('Y')                           
Non_fake = get_fake_or_nonfake_revier('N')  
                        
fake_raw_data = list()
nonfake_raw_data = list()   

for i in fake:
    tp = [review[i][1],review[i][2],review[i][3],review[i][4],review[i][8],review[i][9]]
    fake_raw_data.append(tp)
for i in Non_fake:
    tp = [review[i][1],review[i][2],review[i][3],review[i][4],review[i][8],review[i][9]]
    nonfake_raw_data.append(tp)                   
del tp
    
hotel_id = list()
for i in Non_fake+fake:
    hotel_id.append(review[i][9])
del i   

 
"""
获取hotel 唯一id  (72个产品)
"""
unique_hotel_id = list({}.fromkeys(hotel_id).keys())  
unique_hotel_id.sort(key=hotel_id.index) 

 
"""
获取 用户——>评论index 对应矩阵
"""
all = Non_fake + fake
new_reviewer_to_review_index = list()
for tmp in tmp_list:
    tp = list()
    for i in range(len(tmp)):      
        if tmp[i] in all:
            tp.append(tmp[i])
    new_reviewer_to_review_index.append(tp)

del i
del t
del id    
del tp  
del tmp          
del tmp_list
del index
del only_index


#==============================================================================
# Relations representation
#==============================================================================
# 1,Have reviewed
x0 = np.zeros((len(reviewer)+len(unique_hotel_id),len(reviewer)+len(unique_hotel_id)),dtype=np.float32)

have_reviewed = list()

for i,item in enumerate(new_reviewer_to_review_index):
    tmp = []
    for item2 in item:
        tmp.append(unique_hotel_id.index(review[item2][9]))
    have_reviewed.append(tmp)      

for i,item in enumerate(have_reviewed):
    if item != '':
        for item2 in item:
            x0[i][len(reviewer)+item2] = 1.0
            x0[len(reviewer)+item2][i] = 1.0

# 2,Rating score
x1 = np.zeros((len(reviewer)+len(unique_hotel_id),len(reviewer)+len(unique_hotel_id)),dtype=np.float32)

for i,item in enumerate(have_reviewed):
    if item != '':
        for j,item2 in enumerate(item):
            x1[i][len(reviewer)+item2] = float(review[new_reviewer_to_review_index[i][j]][4])
            x1[len(reviewer)+item2][i] = float(review[new_reviewer_to_review_index[i][j]][4])

# 3,Commonly reviewed products
x2 = np.zeros((len(reviewer)+len(unique_hotel_id),len(reviewer)+len(unique_hotel_id)),dtype=np.float32)










x = np.array([[1, 2, 3], [1, 3, 4], [1, 5, 6]])
y = np.array([[1, 9,10], [3, 12,11], [5, 9, 10]])
z = np.array([[5, 2, 3], [9, 5, 6], [90,8, 6]])
p = np.array([[5, 2, 3], [9, 4, 6], [9,8, 6]]) 
 

a2 = np.zeros((3,3,4), dtype='int16')     # 创建3*3*4全0三维数组

a2[:,:,0] = x
a2[:,:,1] = y
a2[:,:,2] = z
a2[:,:,3] = p

from scipy.sparse import lil_matrix
from rescal import rescal_als

# Load Matlab data and convert it to dense tensor format
T = a2
X = [lil_matrix(T[:, :, k]) for k in range(T.shape[2])]

# Decompose tensor using RESCAL-ALS
A, R, fit, itr, exectimes = rescal_als(X, 2, init='nvecs', lambda_A=10, lambda_R=10)




'''
from rescal import rescal_als
from scipy.sparse import csr_matrix 
X1 = csr_matrix(([1,1,1], ([2,1,3], [0,2,3])), shape=(4, 4))
X2 = csr_matrix(([1,1,1,1], ([0,2,3,3], [0,1,2,3])), shape=(4, 4))
A, R, fval, iter, exectimes = rescal_als([X1, X2], 2)


import scipy.io as sio
import numpy as np

###下面是讲解python怎么读取.mat文件以及怎么处理得到的结果###
load_data = sio.loadmat('./src/ICPAES/alyawarra.mat')
load_matrix = load_data['matrix']
'''








