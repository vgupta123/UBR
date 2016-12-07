import sys
import sqlite3
import pandas as pd
import numpy as np
import nltk
import ast
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

#Run SVM and UBR using tf-idf features
#UBR-1 is run if sys.argv[1]=='1'
#UBR-2 is run if sys.argv[1]=='2'
if sys.argv[1]=='1':
    from load_db_movies import *
elif sys.argv[1]=='2':
    from load_db_movies_2 import *

sampling=open('../preprocess/sampling_movies.txt','r').read()
sampling=ast.literal_eval(sampling)
for name,entity in izip(['Summary_Text'],[Summary_Text]):
    check='nh'
    for name2,vocabulary in izip(['Unigram','Bigram'],[1,2]):
        print name2
        #--- Training set
        corpus = []
    y_train=[]
    yt_train=[]
    i=-1
    for text in entity:
        i+=1
        if check=='h' and sampling[i+1]!=333 and sampling[i]!=0:
            sample_weights.append(sampling[i+1]**2)
        elif check=='nh' and sampling[i]!=3:
            temp=1      
        else:
            continue 
            corpus.append(text)
        y_train.append(Score[i])
        yt_train.append(transformed_score[i])
        
        count_vect = CountVectorizer(stop_words=['$',','],ngram_range=(1, vocabulary),max_features=25000)
        X_train_counts = count_vect.fit_transform(corpus)        
        tfidf_transformer = TfidfTransformer()
        X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

        #--- Test set

        test_set = []
    y_test=[]
    yt_test=[]
    u_test=[]
    i=-1
        for text in entity:
        i+=1
        if sampling[i]!=3:
            continue
            test_set.append(text)
        y_test.append(Score[i])
        yt_test.append(transformed_score[i])
        u_test.append(UserId[i])
    X_new_counts = count_vect.transform(test_set)
        X_test_tfidf = tfidf_transformer.transform(X_new_counts)

        prediction = dict()

        from sklearn.multiclass import OneVsRestClassifier
        from sklearn.svm import LinearSVC
        from sklearn import linear_model
    from sklearn.svm import LinearSVR
    from sklearn.linear_model import Ridge
    ans_simple=LinearSVC().fit(X_train_tfidf, y_train).predict(X_test_tfidf)
        ans_transformed=linear_model.LinearRegression().fit(X_train_tfidf, yt_train).predict(X_test_tfidf)

    mae_simple=0
    mae_transformed=0
    rmse_simple=0
    rmse_transformed=0
        acc_transformed=0
        for i,ans in izip(range(0,len(y_test)),y_test):
        mae_simple+=abs(ans_simple[i]-ans)
        rmse_simple+=(ans_simple[i]-ans)**2
        print 'mae_SVM',mae_simple*1.0/(len(y_test))
    print 'rmse_SVM',(rmse_simple*1.0/(len(y_test)))**(0.5)

        size=len(ans_transformed)
        i=0
        for user in u_test:
        if sys.argv[1]=='1':
            ans_transformed[i]=ans_transformed[i]*user_std[user]+user_mean[user]
        elif sys.argv[1]=='2':
                ans_transformed[i]=ans_transformed[i]+user_bias[user]
            minm=1000
            ans=0
            for j in range(1,6):
                    if((ans_transformed[i]-j)**2<minm):
                        minm=(ans_transformed[i]-j)**2
                        ans=j
            ans_transformed[i]=ans
            i+=1

        for i,ans in izip(range(0,len(y_test)),y_test):
        mae_transformed+=abs(ans_transformed[i]-ans)
                rmse_transformed+=(ans_transformed[i]-ans)**2
        print 'mae_UBR',mae_transformed*1.0/(len(y_test))
    print 'rmse_UBR',(rmse_transformed*1.0/(len(y_test)))**(0.5)
