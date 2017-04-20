## UBR: User Bias Removal in Fine Grained Sentiment Analysis


### Introduction

* Major problem in current sentiment classification models is noise due to presence of user biases in reviews rating.
* We worked on two simple statistical methods to remove user bias noise to improve fine grained sentimental classification.
* We applied our methods on SNAP published Amazon Fine Food Reviews data-set and two major categories Electronics and Movies & TV of e-Commerce Reviews data-set. Correspondingly, there are 3 folders, food, electronics and movies.

### Setup

Run "setup.sh" for setting up.

### Testing

Scripts for testing is in three folders.

* electronics

* food

* movies

cd to appropriate folder and then:

####For getting PV-DBoW features
python doc2vec.py

####For testing various baselines

python baseline.py #User mean,mode etc.

python predict5.py #Always predict 5

####For testing UBR-1 and UBR-2 with LDA faetures
python lda_implement.py

####For testing UBR-1 with tf-idf faetures
python tfidf.py 1

####For testing UBR-2 with tf-idf faetures
python tfidf.py 2

### Authors

- Rahul Kumar Wadbude (IIT Kanpur)(warahul@iitk.ac.in)
- Vivek Gupta (Microsoft Research)(t-vigu@microsoft.com)
- Dheeraj Mekala (IIT Kanpur)(dheerajm@iitk.ac.in)
- Janish Jindal (IIT Kanpur)(janish@iitk.ac.in)
- Harish Karnick (IIT Kanpur)(hk@iitk.ac.in)
