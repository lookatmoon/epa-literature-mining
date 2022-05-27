# -*- coding: utf-8 -*-
"""
Created on Sun Dec  5 02:18:36 2021

@author: Lenovo
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

# font size of plot
matplotlib.rcParams.update({'font.size':13})
matplotlib.rc('xtick', labelsize=16)
matplotlib.rc('ytick', labelsize=16)
 
# function of drawing plot
def recall_plot(df):
    label = df['Label'].tolist()
    check_points = [1000*i for i in range(1,171)]
    prec, recall = [1],[0]
    percentile = []
    for point in check_points:
        sublist = label[:point]
        pos_num = [i for i in sublist if i == 1]
        recall.append(len(pos_num)/1152*100)
        if len(pos_num)/1152*100>=95:
            percentile.append(point)
    print(percentile[0]/171393)
    standard_points = [1/171*i*100 for i in range(171)]
    plt.plot(standard_points,recall)


# read ranking scores of different methods
ml_based_df = pd.read_csv("ml_rank.csv")
ml_based_df['ml_rank'] = pd.Series(ml_based_df.index).apply(lambda x: x+1)
ml_based_df.rename(columns={'Score' : 'ml_score'}, inplace = True)

text_based_df = pd.read_csv("text_based_ensemble_ranker_score.csv")
text_based_df['text_rank'] = pd.Series(text_based_df.index).apply(lambda x: x+1)
text_based_df.rename(columns={'Score' : 'text_score'}, inplace = True)


network_based_df = pd.read_csv("network_based_score.csv")
network_based_df['network_rank'] = pd.Series(network_based_df.index).apply(lambda x: x+1)
network_based_df.rename(columns={'Score' : 'network_score'}, inplace = True)

simple_text_based_df = pd.read_csv("text_based_simple_ranker_score.csv")
simple_text_based_df['simple_text_rank'] = pd.Series(simple_text_based_df.index).apply(lambda x: x+1)
simple_text_based_df.rename(columns={'Score' : 'simple_text_score'}, inplace = True)

# do the combination by using average
combined_df = text_based_df.merge(ml_based_df[['HeroId', 'ml_score', 'ml_rank']], on = ['HeroId'])
combined_df = combined_df.merge(simple_text_based_df[['HeroId', 'simple_text_score', 'simple_text_rank']], on = ['HeroId'])
combined_df = combined_df.merge(network_based_df[['HeroId', 'network_score', 'network_rank']], on = ['HeroId'])
combined_df['combined_rank'] = (combined_df['simple_text_rank'] + combined_df['text_rank']+combined_df['ml_rank'] + combined_df['network_rank'])/4
combined_df['combined_score'] = (combined_df['simple_text_score'] + combined_df['text_score']+combined_df['ml_score'] + combined_df['network_score'])/4





# drawing recall curves of them
simple_text_df = combined_df.sort_values(by = ['simple_text_score'], ascending=False)
recall_plot(simple_text_df)

text_df = combined_df.sort_values(by = ['text_score'], ascending=False)
recall_plot(text_df)

network_df = combined_df.sort_values(by = ['network_score'], ascending=False)
recall_plot(network_df)

ml_df = combined_df.sort_values(by = ['ml_score'], ascending=False)
recall_plot(ml_df)

combined_df = combined_df.sort_values(by = ['combined_score'], ascending=False)
recall_plot(combined_df)


# generate fake labels for the sake of active learning
fake_label = [1 for i in range(2063)] + [0 for i in range(len(combined_df) - 2063)]
combined_df["fake_label"] = fake_label
combined_df.to_csv('combination_with_fake_label.csv')


plt.plot([0,100], [0,100])
plt.plot([0,100], [95,95], linestyle="dashdot")


plt.xlabel('% of references screened in 2020 ozone ISA S set', fontsize = 16)
plt.ylabel('% of recall', fontsize = 16)

plt.legend([ "Text-based Simple Ranker" , "Text-based Ensemble Ranker" ,  "Network-based Ranker", "Context Paragraph-based Ranker", "Combined Ranker", "Random Ranking Baseline", "Recall=95%"],loc='lower right')
plt.xlim((0,100))
plt.ylim((0,100))
plt.tight_layout()
plt.show()
