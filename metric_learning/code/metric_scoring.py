# -*- coding: utf-8 -*-
"""
Created on Thu Feb 17 14:41:08 2022

@author: Lenovo
"""

import pandas as pd
import matplotlib.pyplot as plt



# rank articles
df_score = pd.read_csv("scores.csv")
df_main = pd.read_csv('reference_metadata_2020_BR_5-2-2022.csv')
df_main['score'] = df_score['score']
df_main = df_main[df_main['IN_SEARCH'] == 'Y']
df_main = df_main.sort_values(by = ['score'], ascending=False)

# output a standard file, for the combination of methods
label = df_main['CITED'].tolist()
label = [1 if i == 'Y' else 0 for i in label]
score = df_main['score']
output_df = pd.DataFrame()
output_df['PMID'] = df_main['PMID']
output_df['HeroId'] = df_main['REFERENCE_ID']
output_df['Label'] = label
# normalization, for a better combination
score = (score-score.min())/(score.max()-score.min())
output_df['Score'] = score
output_df.reset_index(drop = True)
output_df.to_csv('ml_rank.csv')


