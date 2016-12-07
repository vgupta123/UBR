import sqlite3
import sys
import csv
import pandas as pd
import numpy as np
import nltk
import string
import math
import matplotlib.pyplot as plt
import numpy as np
import random

from itertools import izip
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.metrics import roc_curve, auc
from nltk.stem.porter import PorterStemmer

#Split the data in 80:20 train test
data=csv.reader(open(sys.argv[1],'r'))
count=0
i=0
Id=[]
for item in data:
	if i==0:
		i+=1
		continue
	Id.append(count)
	count+=1	

check_list=[]
for idj in Id:
	check_list.append(1)

Id_train,Id_test= train_test_split(Id,test_size=0.2, random_state=42)
for test in Id_test:
	check_list[int(test)]=3#test

w=open(sys.argv[2],'w')
w.write(str(check_list))
print len(check_list)
