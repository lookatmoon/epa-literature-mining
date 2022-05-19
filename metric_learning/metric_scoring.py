# -*- coding: utf-8 -*-
"""
Created on Thu Feb 17 14:41:08 2022

@author: Lenovo
"""

import pandas as pd
import matplotlib.pyplot as plt

def labeling(all_file, cited_file):
    all_list = list(pd.read_csv(all_file)['REFERENCE_ID'].astype(str))
    all_list = [str(i).split('.')[0] for i in all_list]
    cited_list = list(pd.read_csv(cited_file)['hero_id'].astype(str))
    cited_list = [str(i).split('.')[0] for i in cited_list]
    labels = []
    for i in all_list:
        if i not in cited_list:
            labels.append(0)
        else:
            labels.append(1)
    return all_list,labels


df = pd.read_csv("scores.csv")

df_main = pd.read_csv('reference_metadata_2020_BR_5-2-2022.csv')
df_main['score'] = df['0']
df_main = df_main[df_main['IN_SEARCH'] == 'Y']

#df_main = pd.read_csv('ozone_2020_litsearch_10-5-2021.csv')

#df_pos =  pd.read_csv('ozone_2020_header_heroid_map.csv')



#label = labeling('ozone_2020_litsearch_10-5-2021.csv', 'ozone_2020_header_heroid_map.csv')[1]
#df_main['score'] = df['0']
#df_main['label'] = pd.Series(label)
df_main = df_main.sort_values(by = ['score'], ascending=False)

label = df_main['CITED'].tolist()
label = [1 if i == 'Y' else 0 for i in label]
score = df_main['score']


pair = list(zip(label, score))
another_points = [1000*i for i in range(171)]
check_points = [1000*i for i in range(1,171)]
prec, recall = [1],[0]
for point in check_points:
    sublist = pair[:point]
    pos_pre = [i for i in sublist[-1000:] if i[0] == 1]
    pos_cul = [i for i in sublist if i[0] == 1]
    prec.append(len(pos_pre)/len(sublist))
    recall.append(len(pos_cul)/1153)
    print(len(pos_cul)/1153)
    if len(pos_cul)/1153 >= 0.95:
      print(check_points/170223)
# h_points = [3000, 34000, 49000]
# v_points = [0.5564516129032258, 0.9009216589861752, 0.9539170506912442]
# plt.vlines(h_points, 0, v_points, linestyle="dashed")
# plt.hlines(v_points, 0, h_points, linestyle="dashed")
plt.plot(another_points,recall)
plt.plot(another_points,prec)
plt.xlabel('Length of Ranking List')
plt.ylabel('Recall Ratio')
plt.legend(["Recall"],loc='lower right')
# plt.annotate('recall@k=50%', xy = (1000,0.3))
# plt.annotate('recall@k=90%', xy = (28000,0.7))
# plt.annotate('recall@k=95%', xy = (45000,0.8))
plt.xlim((0,171000))
plt.ylim((0,1))
#plt.show()

