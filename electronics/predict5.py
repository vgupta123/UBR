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

from load_db_electronics import *
#Always predict 5(majority vote)

sampling=open('sampling_electronics.txt','r').read()
sampling=ast.literal_eval(sampling)
for name,entity in izip(['Summary'],[Summary]):
        print name
        check='nh'
    	#--- Training set
	y_train=[]
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
        
    	#--- Test set

	y_test=[]
	i=-1
    	for text in entity:
		i+=1
		if sampling[i]!=3:
			continue
		y_test.append(Score[i])

	mae_simple=0
	rmse_simple=0
        c=[0,0,0,0,0]
    	for i,ans in izip(range(0,len(y_test)),y_test):
		for j in range(1,6):
			if j==ans:
				c[j-1]+=1
		mae_simple+=abs(5-ans)
		rmse_simple+=(5-ans)**2
    	print 'mae',mae_simple*1.0/(len(y_test))
	print 'rmse',(rmse_simple*1.0/(len(y_test)))**(0.5)