# -*- coding: utf-8 -*-
"""
Created on Wed Oct 27 00:22:17 2021

@author: Lenovo
"""

import pandas as pd

filepath = "D:/EPA/Data and Code from EPA/data/"
cited = 'ozone_2020_header_heroid_map.csv'
all_lit = "ozone_2020_litsearch_10-5-2021.csv"

cited_df = pd.read_csv(filepath+cited)
all_df = pd.read_csv(filepath+all_lit)
merged = cited_df.merge(all_df, left_on = 'hero_id', right_on = 'REFERENCE_ID').drop_duplicates(subset=['hero_id'])#.dropna(subset=['PMID'])
print(merged)
# all_df = all_df.dropna(subset=['PMID'])
# merged.to_csv('2020_cited.csv')
# all_df.to_csv('2020_all.csv')