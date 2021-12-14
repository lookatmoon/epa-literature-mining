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



def get_arc(id):
    
    
    # use icite api to get citation relation
    id = str(id)
    r = requests.get(f"https://icite.od.nih.gov/api/pubs?pmids={id}")
    citing_df = (
            pd.DataFrame(r.json()["data"])[
                ["pmid", "year", "title", "authors", "cited_by"]
            ]
            #.set_index(["pmid", "year", "title", "authors"])
            #.explode("cited_by")
        )
    cited = list(citing_df["cited_by"])[0]
    cited = [str(i).split('.')[0] for i in cited]
    references_df = (
            pd.DataFrame(r.json()["data"])[
                ["pmid", "year", "title", "authors", "references"]
            ]
            #.set_index(["pmid", "year", "title", "authors"])
            #.explode("references")
        )

    refer = list(references_df["references"])[0]
    refer = [str(i).split('.')[0] for i in refer]
    return cited,refer


# construct graph
ids = pd.read_csv("D:/EPA/Data and Code from EPA/data/S_union.csv")['PMID'].astype(str)
ids = [str(i).split('.')[0] for i in ids]
G = nx.DiGraph()
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
nx.write_gml(G, 'union.gml')

G = nx.read_gml('union.gml')
node2vec = Node2Vec(G, dimensions=100, walk_length=30, num_walks=500)#, workers=4)
model = node2vec.fit(window=10, min_count=1, batch_words=4)  
model.save("union_emb_.txt")
