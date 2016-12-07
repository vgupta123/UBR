import sqlite3
import sys
import pandas as pd
import numpy as np
import nltk
import string
import matplotlib.pyplot as plt
from sklearn.decomposition import LatentDirichletAllocation
import numpy as np

from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from itertools import izip
from sklearn.metrics import roc_curve, auc
from nltk.stem.porter import PorterStemmer

#Run SVM and UBR using LDA features
import ast
from load_db_electronics import Summary_Text,Score,UserId
sampling=ast.literal_eval(open('../preprocess/sampling_electronics.txt','r').read())
for name,entity in izip(['Summary_Text'],[Summary_Text]):
    for check in ['nh']:
    	#--- Training set
    	corpus = []
	y_train=[]
	i=-1
   	for text in entity:
		i+=1
		if check=='nh' and sampling[i]!=3:
			temp=1
		else:
			continue 
        	corpus.append(text)
		y_train.append(Score[i])
		
    	count_vect = CountVectorizer(max_features=25000)
    	X_train_counts = count_vect.fit_transform(corpus)        
            

    	#--- Test set

    	test_set = []
	y_test=[]
	u_test=[]
	i=-1
    	for text in entity:
		i+=1
		if sampling[i]!=3:
			continue
        	test_set.append(text)
		y_test.append(Score[i])
		u_test.append(UserId[i])
	X_new_counts = count_vect.transform(test_set)
	model_lda = LatentDirichletAllocation(n_topics=100)
        X_train_lda=model_lda.fit_transform(X_train_counts)
        X_test_lda=model_lda.transform(X_new_counts)
        from sklearn.multiclass import OneVsRestClassifier
	from sklearn.svm import LinearSVC
        from sklearn import linear_model
	ans_simple=OneVsRestClassifier(LinearSVC(random_state=0)).fit(X_train_lda, y_train).predict(X_test_lda)
	rmse_simple=0
        mae_simple=0
        for i,ans in izip(range(0,len(y_test)),y_test):
                        mae_simple+=abs(ans_simple[i]-ans)
                        rmse_simple+=(ans_simple[i]-ans)**2
        print 'mae_SVM',mae_simple*1.0/(len(y_test))
        print 'rmse_SVM',(rmse_simple*1.0/(len(y_test)))**(0.5)
	for name2,algo in izip(['UBR-1','UBR-2'],[1,2]):
		print name2
		if algo==1:
			from load_db_electronics import transformed_score,user_mean,user_std
		else:
			from load_db_electronics_2 import transformed_score,user_bias
		yt_train=[]
	        i=-1
        	for text in entity:
               		i+=1
                	if check=='nh' and sampling[i]!=3:
                        	temp=1
                	else:
                        	continue
                	yt_train.append(transformed_score[i])
 
		ans_transformed=linear_model.LinearRegression().fit(X_train_lda, yt_train).predict(X_test_lda)
                mae_transformed=0
	        rmse_transformed=0

		size=len(ans_transformed)
		i=0
		for user in u_test:
			if algo==2:
				ans_transformed[i]=ans_transformed[i]+user_bias[user]
			else:
				ans_transformed[i]=ans_transformed[i]*user_std[user]+user_mean[user]
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

