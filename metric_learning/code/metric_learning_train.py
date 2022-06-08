# -*- coding: utf-8 -*-
"""
Created on Wed Feb 16 14:54:21 2022

@author: Lenovo
"""

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from metric_learn import MMC
import numpy as np
from sklearn.decomposition import PCA
import random
from collections import Counter
from sentence_transformers import SentenceTransformer



def aggr(df):
    if str(df['ABSTRACT']) != 'nan':
        return df['TITLE'] + ' ' + df['ABSTRACT']
    else:
        return df['TITLE']
    
def cleaning(sen):
    sen = [ps.stem(i) for i in word_tokenize(sen) if i not in stop_words]
    return ' '.join(list(set([i for i in sen if i != 'ozon'])))



# load pretrained bert model
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
stop_words = stopwords.words('english')
ps = PorterStemmer()

# load positive and negative samples
neg_df = pd.read_csv('neg_citation_sampled.csv')
pos_df = pd.read_csv('pos_citation.csv')
# load test dataset, concatenate title and abstract
test = pd.read_csv('reference_metadata_2020_BR_5-2-2022.csv')
test['text'] = test.apply(aggr, axis = 1)




# concatenate positive and negative samples as training set
neg = list(zip(model.encode(neg_df['context'].tolist()), model.encode(neg_df['text'].tolist()), [0 for i in range(len(neg_df))]))
pos = list(zip(model.encode(pos_df['context'].tolist()), model.encode(pos_df['text'].tolist()), [1 for i in range(len(neg_df))]))
print('transform complete')
train = neg+pos
np.save('train_vec.npy', np.array(train))
train = np.load('train_vec.npy', allow_pickle = True)
# for the usage of metric learning, we need to transform negative labels from 0 to -1
for i in train:
    if i[-1] == 0: 
        i[-1] = -1
random.shuffle(train)
train_x, train_y = [list(i[:2]) for i in train], [i[2] for i in train]



# train the model
mmc = MMC(max_iter=10000)
print('start_training')
mmc.fit(train_x, train_y)
test = model.encode(test['text'].tolist())
np.save('test_vec.txt', np.array(test))
# load context paragraphs to calculate maximum scores
paras = pd.read_csv('citation_context_2013_BR_5-3-2022.csv', encoding = 'utf-8')['CONTEXT_PARAGRAPH'].tolist()
paras = list(set(paras))
paras = model.encode(paras)


# get maximum scores and store them
scores = []
num = 0
for i in test:
    print(num)
    scores.append(max(mmc.score_pairs([[g, i] for g in paras])))
    num += 1
scores = pd.Series(scores)
scores.to_csv('scores.csv')

