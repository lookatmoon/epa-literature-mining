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



def get_edge(id):
    
    
    # use icite api to get citation relations
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


# construct graph

# generate edges and store them in form of list
ids = pd.read_csv("D:/EPA/Data and Code from EPA/data/S_union.csv")['PMID'].astype(str)
ids = [str(i).split('.')[0] for i in ids]
G = nx.DiGraph()
edges = []
for d in range(len(ids)):
    print(d)
    i = ids[d]
    if i!='nan':
        try:
            cited, refer = get_edge(i)
            cited_edges, refer_edges = [(i, g) for g in cited], [(g, i) for g in refer]
            edges = edges + cited_edges + refer_edges
        except:
            print('Incomplete information in the webpage.')
            
            
# use networkx library to establish the graph
edges = list(set([i for i in arcs if i[0] in ids and i[1] in ids]))
G.add_edges_from(edges)
# store the graph as gml file
nx.write_gml(G, 'union.gml')



G = nx.read_gml('union.gml')
# use node2vec to conduct unsupervised learning to obtain a representation of articles
node2vec = Node2Vec(G, dimensions=100, walk_length=30, num_walks=500)
model = node2vec.fit(window=10, min_count=1, batch_words=4)  
# save the output vectors
model.save("union_emb_.txt")
