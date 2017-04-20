import gensim
from gensim.models.doc2vec import TaggedDocument
from collections import namedtuple
import io
from gensim.models import Doc2Vec
import gensim.models.doc2vec
from collections import OrderedDict
import multiprocessing
from gensim.test.test_doc2vec import ConcatenatedDoc2Vec
import scipy
import numpy as np
import statsmodels.api as sm
from random import sample

# for timing
from contextlib import contextmanager
from timeit import default_timer
import time 
from collections import defaultdict
from random import shuffle
from scipy import io as sio
import datetime

from load_db_movies import *
SentimentDocument = namedtuple('SentimentDocument', 'words tags')
alldocs=[]
for name,entity in izip(['Summary_Text'],[Summary_Text]):
	i = 0
        for text in entity:
		words = text.split()
		tags = [i]
		alldocs.append(SentimentDocument(words, tags))		
		i = i+1	
doc_list = alldocs[:]  # for reshuffling per pass

cores = multiprocessing.cpu_count()

simple_models = [
    # PV-DBOW
    Doc2Vec(dm=0, size=400, negative=5, hs=0, min_count=20, workers=cores, dbow_words=1),
]

simple_models[0].build_vocab(alldocs)  # PV-DM/concat requires one special NULL word so it serves as template
print(simple_models[0])

models_by_name = OrderedDict((str(model), model) for model in simple_models)

best_error = defaultdict(lambda :1.0)  # to selectively-print only best errors achieved

alpha, min_alpha, passes = (0.025, 0.001, 5)
alpha_delta = (alpha - min_alpha) / passes

print("START %s" % datetime.datetime.now())

for epoch in range(passes):
    shuffle(doc_list)  # shuffling gets best results
    
    for name, train_model in models_by_name.items():
        # train
        duration = 'na'
        train_model.alpha, train_model.min_alpha = alpha, alpha
        #with elapsed_timer() as elapsed:
        train_model.train(doc_list,total_examples = train_model.corpus_count,epochs = train_model.iter)
        #    duration = '%.1f' % elapsed()
            
    print('completed pass %i at alpha %f' % (epoch + 1, alpha))
    alpha -= alpha_delta

for name, train_model in models_by_name.items():
	sio.savemat('movies_doc2vec.mat', {'movies_doc2vec':train_model.docvecs.doctag_syn0})
