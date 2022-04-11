# -*- coding: utf-8 -*-
"""
Created on Sun Apr 10 22:26:26 2022

@author: Lenovo
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb


# read metric_learning result
ml_based_df = pd.read_csv("ml_score.csv")
ml_based_df['ml_rank'] = pd.Series(ml_based_df.index).apply(lambda x: x+1)
ml_based_df = ml_based_df.reset_index(drop=True)
ml_based_df.rename(columns={'Score' : 'ml_score'}, inplace = True)

# read fine_grained text classification result
fg_based_df = pd.read_csv("fine_grained_score_with_title.csv")
fg_based_df['fg_rank'] = pd.Series(fg_based_df.index).apply(lambda x: x+1)
fg_based_df = fg_based_df.reset_index(drop=True)
fg_based_df.rename(columns={'Score' : 'fg_score'}, inplace = True)

# read result of network-based method, allow the existence of missing value
nw_based_df = pd.read_csv("network_based_score.csv")
nw_based_df['nw_rank'] = pd.Series(nw_based_df.index).apply(lambda x: x+1)
nw_based_df = nw_based_df.reset_index(drop=True)
nw_based_df.rename(columns={'Score' : 'nw_score'}, inplace = True)
nw_based_df['nw_rank'] = np.where(nw_based_df['nw_score']>0, nw_based_df['nw_rank'], NaN)

# do the merging
double_df = fg_based_df.merge(ml_based_df[['HeroId', 'ml_score', 'ml_rank']], on = ['HeroId'])
triple_df = double_df.merge(nw_based_df[['HeroId', 'nw_score', 'nw_rank']], on = ['HeroId'])


# do the averaging, treat articles with pmid and without pmid differently
pmid_df = triple_df[triple_df['nw_score'].notna()]
no_pmid_df = triple_df[triple_df['nw_score'].isna()]
pmid_df['ave_rank'] = (pmid_df['ml_rank'] + pmid_df['fg_rank'] + pmid_df['nw_rank'])/3
no_pmid_df['ave_rank'] = (no_pmid_df['ml_rank'] + no_pmid_df['fg_rank'])/2
triple_df = pmid_df.append(no_pmid_df)
triple_df = triple_df.sort_values(by = ['ave_rank'], ascending=True).reset_index(drop=True)

# syntheize fake label given the rank
label = triple_df['Label'].tolist()
fake_label = [1 for i in label if i == 1] + [0 for i in label if i == 0]
triple_df['fake_label'] = pd.Series(fake_label).reset_index(drop=True)
triple_df.to_csv('triple.csv')


# use xgboost to do the basic classification, as it is capable of handling missing value
train_feat = triple_df[['ml_rank', 'fg_rank', 'nw_rank']]
train_label = triple_df[['fake_label']]
test_label = triple_df[['Label']]
train = xgb.DMatrix(train_feat, label = train_label)
test = xgb.DMatrix(train_feat, label = test_label)
param = {'max_depth':100, 'eta':1, 'objective':'binary:logistic' }
num_round = 100
bst = xgb.train(param, train, num_round)
preds = bst.predict(test)
triple_df['xg_score'] = pd.Series(preds).reset_index(drop=True)
triple_df.to_csv('triple.csv')

