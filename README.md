## [UBR: User Bias Removal in Fine Grained Sentiment Analysis] (https://dl.acm.org/doi/10.1145/3152494.3152520)


### Introduction

* Major problem in current sentiment classification models is noise due to presence of user biases in reviews rating.
* We worked on two simple statistical methods to remove user bias noise to improve fine grained sentimental classification.
* We applied our methods on SNAP published Amazon Fine Food Reviews data-set and two major categories Electronics and Movies & TV of e-Commerce Reviews data-set. Correspondingly, there are 3 folders, food, electronics and movies.

### Setup

Run "setup.sh" for setting up.

```
bash setup.sh
```

### Testing

Scripts for testing is in three folders.

* electronics

* food

* movies

cd to appropriate folder and then:

#### For getting PV-DBoW features

```
python doc2vec.py
```

#### For testing various baselines

```
python baseline.py #User mean,mode etc.
python predict5.py #Always predict 5
````

#### For testing UBR-1 and UBR-2 with LDA features

```
python lda_implement.py
```

#### For testing UBR-1 with tf-idf features
```
python tfidf.py 1
```

### For testing UBR-2 with tf-idf faetures
```
python tfidf.py 2
```

### Citation

```
@inproceedings{10.1145/3152494.3152520,
author = {Wadbude, Rahul and Gupta, Vivek and Mekala, Dheeraj and Karnick, Harish},
title = {User Bias Removal in Review Score Prediction},
year = {2018},
isbn = {9781450363419},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3152494.3152520},
doi = {10.1145/3152494.3152520},
booktitle = {Proceedings of the ACM India Joint International Conference on Data Science and Management of Data},
pages = {175–179},
numpages = {5},
keywords = {bias removal, score prediction, user modeling},
location = {Goa, India},
series = {CoDS-COMAD ’18}
}


```
