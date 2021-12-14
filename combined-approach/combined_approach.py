# -*- coding: utf-8 -*-
"""
Created on Sun Dec  5 02:18:36 2021

@author: Lenovo
"""

import pandas as pd
import numpy as np

network_based_df = pd.read_csv("network_based_score.csv")
network_based_df['network_rank'] = pd.Series(network_based_df.index).apply(lambda x: x+1)
network_based_df.rename(columns={'Score' : 'network_score'}, inplace = True)

text_based_df = pd.read_csv("text_based_score.csv")
text_based_df['text_rank'] = pd.Series(text_based_df.index).apply(lambda x: x+1)
text_based_df.rename(columns={'Score' : 'text_score'}, inplace = True)


combined_df = text_based_df.merge(network_based_df[['REFERENCE_ID', 'network_score', 'network_rank']], on = ['REFERENCE_ID'])
double_methods_df = combined_df[combined_df['network_score'].notna()]
single_method_df = combined_df[combined_df['network_score'].isna()]
double_methods_df['combined_rank'] = (double_methods_df['text_rank']+double_methods_df['network_rank'])/2
double_methods_df['combined_score'] = (double_methods_df['text_score']+double_methods_df['network_score'])/2
single_method_df['combined_rank'] = single_method_df['text_rank']
single_method_df['combined_score'] = single_method_df['text_score']
combined_df = double_methods_df.append(single_method_df)[['REFERENCE_ID', 'PMID', 'Label', 'combined_rank', 'combined_score']]
combined_df.to_csv("combined_approach.csv")