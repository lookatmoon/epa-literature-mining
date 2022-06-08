# -*- coding: utf-8 -*-
"""
Created on Wed Feb  9 10:41:04 2022

@author: Lenovo
"""

import pandas as pd
import numpy as np
# from nltk.corpus import stopwords
# from nltk.tokenize import word_tokenize
# from nltk.stem import PorterStemmer
import nltk
from sklearn.feature_extraction.text import CountVectorizer
import json
import requests 

# stop_words = stopwords.words('english')
# ps = PorterStemmer()

def aggr(df):
    if str(df['ABSTRACT']) != 'nan':
        return df['TITLE'] + ' ' + df['ABSTRACT']
    else:
        return df['TITLE']

def cleaning(sen):
    sen = [ps.stem(i) for i in word_tokenize(sen) if i not in stop_words]
    return ' '.join(list(set([i for i in sen if i != 'ozon'])))

def overlap(x,y):
    return np.sum(x*y)
    



tuples = []
df1 = pd.read_csv('citation_context_2013_BR_5-3-2022.csv', encoding = 'utf-8')
df2 = pd.read_csv('reference_metadata_2013_BR_5-2-2022.csv', encoding = 'utf-8')
df1 = df1.merge(df2, on = 'REFERENCE_ID')
df1['text'] =df1.apply(aggr, axis = 1)
df2['text'] =df2.apply(aggr, axis = 1)
paras = df1['CONTEXT_PARAGRAPH']
cited_text = df1['text']
text = df2['text'].tolist()

# load postive samples
pos = pd.DataFrame()
pos['context'] = paras
pos['text'] = cited_text
pos.to_csv('pos_citation.csv', encoding = 'utf-8')


# load negative samples
cited_text = set(cited_text.tolist())
text = list(set(text) - cited_text)
neg = pd.DataFrame()
neg['text'] = pd.Series(text)
neg.to_csv('neg_citation.csv', encoding = 'utf-8')


    
paras = pd.read_csv('pos_citation.csv', encoding = 'utf-8')['context'].tolist()
c_text = pd.read_csv('pos_citation.csv', encoding = 'utf-8')['text'].tolist()
text = pd.read_csv('neg_citation.csv', encoding = 'utf-8')['text'].dropna().tolist()
paras = list(set(paras))


# using a Solr index to get top 10 negative results as sampling
neg_con, neg_text = [],[]
tokenizer = nltk.RegexpTokenizer(r"\w+")
num = 0
for g in paras:
    q_expression = ' OR%20text%3A%20'.join(tokenizer.tokenize(g))
    # may require VPN, I'll try to make the Solr index public later
    r = requests.get("http://neurobridges-ml.edc.renci.org:8983/solr/PairGene/select?indent=true&q.op=OR&q=text%3A%20" + q_expression)
    # use try/except, in case that there are less than 10 results returned by Solr
    try:
        cand = [i['text'] for i in r.json()['response']['docs'][:10]]
        for i in cand:
            neg_con.append(g)
            neg_text.append(i[0])
    except:
        pass
    num += 1
    print(num)
    
# save sampled negative set
neg = pd.DataFrame()
neg['context'] = pd.Series(neg_con)
neg['text'] = pd.Series(neg_text)
neg.to_csv('neg_citation_sampled.csv')


