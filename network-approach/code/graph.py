# -*- coding: utf-8 -*-
"""
Created on Thu Oct 14 16:27:11 2021

@author: Lenovo
"""

import networkx as nx
import numpy as np
import requests
import pandas as pd
import time
import matplotlib.pyplot as plt
from node2vec import Node2Vec
from gensim.models import Word2Vec


# Construct network using icite API, then store the graph in form of .gml file
def get_arc(id):
    id = str(id)
    r = requests.get(f"https://icite.od.nih.gov/api/pubs?pmids={id}")
    citing_df = (
            pd.DataFrame(r.json()["data"])[
                ["pmid", "year", "title", "authors", "cited_by"]
            ]
        )
    cited = list(citing_df["cited_by"])[0]
    cited = [str(i).split('.')[0] for i in cited]
    references_df = (
            pd.DataFrame(r.json()["data"])[
                ["pmid", "year", "title", "authors", "references"]
            ]
        )

    refer = list(references_df["references"])[0]
    refer = [str(i).split('.')[0] for i in refer]
    return cited,refer

# use union of S sets
df_13 = pd.read_csv("reference_metadata_2013_BR_5-2-2022.csv")
df_20 = pd.read_csv("reference_metadata_2020_BR_5-2-2022.csv")
df_13 = df_13[df_13['IN_SEARCH'] == 'Y']
df_20 = df_20[df_20['IN_SEARCH'] == 'Y']
ids_13 = df_13['PMID'].reset_index(drop=True)
ids_20 = df_20['PMID'].reset_index(drop=True)
ids = ids_13.append(ids_20).dropna().drop_duplicates()
ids = [str(i).split('.')[0] for i in ids]
G = nx.DiGraph()
t = time.time()
arcs = []
for d in range(len(ids)):
    print(d)
    i = ids[d]
    if i!='nan':
        try:
            cited, refer = get_arc(i)
            cited_arcs, refer_arcs = [(i, g) for g in cited], [(g, i) for g in refer]
            arcs = arcs + cited_arcs + refer_arcs
        except:
            print('Incomplete information in the webpage.')
arcs = list(set([i for i in arcs if i[0] in ids and i[1] in ids]))
G.add_edges_from(arcs)
print(time.time()-t)
nx.write_gml(G, 'new_union.gml')

# G = nx.read_gml('union.gml')


# train a node2vec model for learning representation of nodes(articles)
print('start training')
node2vec = Node2Vec(G, dimensions=100, walk_length=30, num_walks=500)#, workers=4)
model = node2vec.fit(window=10, min_count=1, batch_words=4)  
# check representation of an article
vect = model.wv['9465268']
print(vect)
# save the model
model.save("union_emb_.txt")
