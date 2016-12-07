import sqlite3
import pandas as pd
import numpy as np
import nltk
import gc
import ast
import string
from scipy.sparse import hstack
from itertools import izip
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from scipy.sparse import csc_matrix
import numpy as np
from gensim.models import word2vec
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from itertools import izip
from sklearn.metrics import roc_curve, auc
from nltk.stem.porter import PorterStemmer
from load_db_movies import *

#test four baselines
#(i) User Mean
#(ii) User Mode
#(iii) Product Mean
#(iv) Product Mode
sampling=open('../preprocess/sampling_movies.txt','r').read()
sampling=ast.literal_eval(sampling)

for name,entity in izip(['Summary'],[Summary]):
    check='nh'
    for vocabulary in [2]:
    	#--- Training set
	y_train=[]
	yt_train=[]
	u_train=[]
	p_train=[]
	i=-1
   	for text in entity:
		i+=1
		if check=='h' and sampling[i+1]!=333 and sampling[i]!=0:
			sample_weights.append(sampling[i+1]**2)
		elif check=='nh' and sampling[i]!=3:
			temp=1		
		else:
			continue
		y_train.append(Score[i])
		yt_train.append(transformed_score[i])
		u_train.append(UserId[i])
	#--- Test set

	y_test=[]
	yt_test=[]
	u_test=[]
	p_test=[]
	i=-1
    	for text in entity:
		i+=1
		if sampling[i]!=3:
			continue
		y_test.append(Score[i])
		yt_test.append(transformed_score[i])
		u_test.append(UserId[i])
		p_test.append(ProductId[i])
	print 'user_mean'
	mae_transformed=0
	rmse_transformed=0
    	for i,ans in izip(range(0,len(y_test)),y_test):
		mae_transformed+=abs(user_mean[u_test[i]]-ans)
                rmse_transformed+=(user_mean[u_test[i]]-ans)**2
    	print 'mae',mae_transformed*1.0/(len(y_test))
	print 'rmse',(rmse_transformed*1.0/(len(y_test)))**(0.5)
	print 'product mean'
	mae_transformed=0
        rmse_transformed=0
        for i,ans in izip(range(0,len(y_test)),y_test):
                mae_transformed+=abs(product_mean[p_test[i]]-ans)
                rmse_transformed+=(product_mean[p_test[i]]-ans)**2
        print 'mae',mae_transformed*1.0/(len(y_test))
        print 'rmse',(rmse_transformed*1.0/(len(y_test)))**(0.5)
	
	print 'user_mode'
        mae_transformed=0
        rmse_transformed=0
        for i,ans in izip(range(0,len(y_test)),y_test):
                mae_transformed+=abs(user_mode[u_test[i]]-ans)
                rmse_transformed+=(user_mode[u_test[i]]-ans)**2
        print 'mae',mae_transformed*1.0/(len(y_test))
        print 'rmse',(rmse_transformed*1.0/(len(y_test)))**(0.5)
        print 'product mode'
        mae_transformed=0
        rmse_transformed=0
        for i,ans in izip(range(0,len(y_test)),y_test):
                mae_transformed+=abs(product_mode[p_test[i]]-ans)
                rmse_transformed+=(product_mode[p_test[i]]-ans)**2
        print 'mae',mae_transformed*1.0/(len(y_test))
        print 'rmse',(rmse_transformed*1.0/(len(y_test)))**(0.5)

