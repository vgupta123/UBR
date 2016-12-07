import sqlite3
import csv
import pandas as pd
import numpy as np
import nltk
import string
import matplotlib.pyplot as plt
import numpy as np

from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from itertools import izip
from sklearn.metrics import roc_curve, auc
from nltk.stem.porter import PorterStemmer

Score=[]
UserId=[]
ProductId=[]
Summary=[]
Text=[]
HelpfulnessNumerator=[]
HelpfulnessDenominator=[]

#Id,ProductId,UserId,ProfileName,HelpfulnessNumerator,HelpfulnessDenominator,Score,Time,Summary,Text
#load all data from csv 
#and calculate transformed score of UBR-1
data=csv.reader(open('../preprocess/food_saved.csv','r'))
i=0
for row in data:
    if i==0:
        i+=1
        continue
    Score.append(float(row[6]))
    UserId.append(row[2])
    ProductId.append(row[1])
    Summary.append(row[8].decode('utf-8'))
    Text.append(row[9].decode('utf-8'))
    
    
transformed_score=[]
Summary_Text=[]
for summary,text in izip(Summary,Text):
    Summary_Text.append(summary+' $ '+text)
user_mean={}
product_mean={}
product_mode={}
user_mode={}
user_rating={}
product_rating={}
user_size={}
product_size={}
user_std={}
size=len(UserId)
for i in range(0,size):
    if UserId[i] not in user_mean:
    user_rating[UserId[i]]=[0,0,0,0,0,0]
        user_mean[UserId[i]]=Score[i]
        user_std[UserId[i]]=0
        user_size[UserId[i]]=1
    else:
    user_rating[UserId[i]][int(Score[i])]+=1
        user_mean[UserId[i]]+=Score[i]
        user_size[UserId[i]]+=1
UserIdSet = set(UserId)
for userid in UserIdSet:
   user_mean[userid] = (user_mean[userid]*1.0/user_size[userid])
   user_mode[userid] = user_rating[userid].index(max(user_rating[userid])) 
for i in range(0,size):
    if ProductId[i] not in product_mean:
        product_rating[ProductId[i]]=[0,0,0,0,0,0]
        product_mean[ProductId[i]]=Score[i]
        product_size[ProductId[i]]=1
    else:
        product_rating[ProductId[i]][int(Score[i])]+=1
        product_mean[ProductId[i]]+=Score[i]
        product_size[ProductId[i]]+=1
ProductIdSet = set(ProductId)
for productid in ProductIdSet:
   product_mean[productid] = (product_mean[productid]*1.0/product_size[productid])
   product_mode[productid] = product_rating[productid].index(max(product_rating[productid]))

for i in range(0,size):
    user_std[UserId[i]]+=(user_mean[UserId[i]]-Score[i])**2

UserIdSet = set(UserId)
for userid in UserIdSet:
   user_std[userid] = (user_std[userid]/user_size[userid])**0.5

for i in range(0,size):
    if user_std[UserId[i]]!=0:
        transformed_score.append((Score[i]-user_mean[UserId[i]])/user_std[UserId[i]])
    else:
        transformed_score.append(0)

print '----------------------------LOADING DB DONE------------------------------------------------------'
