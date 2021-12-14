# -*- coding: utf-8 -*-
"""
Created on Wed Oct 27 01:18:23 2021

@author: Lenovo
"""

import networkx as nx
import pandas as pd
from sklearn.linear_model import LogisticRegression
from gensim.models import Word2Vec
import numpy as np
from sklearn.metrics import classification_report
from sklearn.metrics import make_scorer
import matplotlib.pyplot as plt
import random
from random import choice


# Extract feature given a list of words
def feature_ext(x):
    nodevec = Word2Vec.load("union_emb.txt")
    feats = []
    for i in x:
        try:
            feats.append([i, nodevec.wv[str(i)]])
        except:
            feats.append([i, list(np.around(np.array([0 for i in range(100)]).astype("float"),decimals=7).astype("bytes"))])
    return feats

# Labeling data by judging whether it is cited or not
def labeling(all_file, cited_file):
    all_list = list(pd.read_csv(all_file)['PMID'].astype(str))
    all_list = [str(i).split('.')[0] for i in all_list]
    cited_list = list(pd.read_csv(cited_file)['PMID'].astype(str))
    cited_list = [str(i).split('.')[0] for i in cited_list]
    labels = []
    for i in all_list:
        if i not in cited_list:
            labels.append(0)
        else:
            labels.append(1)
    return all_list,labels

# Obtain train and test data
x_train,y_train = labeling('2013_all.csv','2013_cited.csv')
x_test,y_test = labeling('2020_all.csv','2020_cited.csv')
x_train_feat, x_test_feat = [i[1] for i in feature_ext(x_train)],  [i[1] for i in feature_ext(x_test)]
test_data = list(zip(x_test_feat,y_test))
# train the classfier
LR = LogisticRegression()
LR.fit(x_train_feat,y_train)


def iteration(LR,pre_x, pre_y, dataset):
    batch_size = 50
    x_feat, y =  [i[0] for i in dataset], [i[1] for i in dataset]
    prob = [i[1] for i in LR.predict_proba(x_feat)]
    pair = list(zip(x_feat, prob, y))
    mean = np.mean(prob)
    cands = []
    for i in pair:
        if i[1]<=mean+0.1 and i[1]>=mean-0.1:
            cands.append(i)
    # samples = random.sample(pair, batch_size)
    samples = random.sample(cands, batch_size)
    samples_x = [i[0] for i in samples]
    samples_y = [i[2] for i in samples]
    return pre_x+samples_x,pre_y+samples_y


# optional: do iterative training
point_x, point_y = [], []
iter_x, iter_y = x_train_feat,y_train
num_iter = 0
for g in range(num_iter):
    iter_x, iter_y = iteration(LR,iter_x, iter_y, test_data)
    LR.fit(iter_x, iter_y)


# generate results
y_pred = LR.predict(x_test_feat)
prob = LR.predict_proba(x_test_feat)





# write data to csv file
y_test_ = [str(i).split('.')[0] for i in y_test]
output = list(zip(x_test, y_test_,[i[1] for i in prob]))
output = sorted(output, key = lambda x: x[2],reverse = True)
output_df = pd.DataFrame(output, columns=['PMID', 'Label', 'Score'])
id_df = pd.read_csv("D:/EPA/Data and Code from EPA/data/"+"ozone_2020_litsearch_10-5-2021.csv")
id_df = id_df[['REFERENCE_ID', 'PMID']]
id_df['PMID'] = id_df['PMID'].apply(lambda x:str(x).split('.')[0])
df = id_df.merge(output_df, left_on = 'PMID', right_on = 'PMID', how = 'outer').sort_values('Score',ascending=False).drop_duplicates(subset=['REFERENCE_ID'])
df.to_csv('network_based_score.csv')


# plot the recall curve
pair = list(zip(y_test,[i[1] for i in prob]))
pair = sorted(pair, key = lambda x: x[1],reverse = True)
another_points = [1000*i for i in range(75)]
check_points = [1000*i for i in range(1,75)]
prec, recall = [1],[0]
for point in check_points:
    sublist = pair[:point]
    pos_pre = [i for i in sublist[-1000:] if i[0] == 1]
    pos_cul = [i for i in sublist if i[0] == 1]
    prec.append(len(pos_pre)/len(sublist))
    recall.append(len(pos_cul)/868)
h_points = [3000, 34000, 49000]
v_points = [0.5564516129032258, 0.9009216589861752, 0.9539170506912442]
plt.vlines(h_points, 0, v_points, linestyle="dashed")
plt.hlines(v_points, 0, h_points, linestyle="dashed")
plt.plot(another_points,recall)
plt.plot(another_points,prec)
plt.xlabel('Length of Ranking List')
plt.ylabel('Recall Ratio')
plt.legend(["Recall"],loc='lower right')
# plt.annotate('recall@k=50%', xy = (1000,0.3))
# plt.annotate('recall@k=90%', xy = (28000,0.7))
# plt.annotate('recall@k=95%', xy = (45000,0.8))
plt.xlim((0,75000))
plt.ylim((0,1))
plt.show()

    