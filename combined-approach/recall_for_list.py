# -*- coding: utf-8 -*-

# Note: the filepath input must be in the desired format: 
# (combined_result.csv can be an example)
# first column: hero_id/reference_id
# second column: PMID
# third column: Label (boolean value) (actual value of being cited)
# fourth column: Score (predicted probability of being cited)

import numpy as np
def recall_at_k_curve_data(filepath,pmid_only,num):
  #filepath is a list
  pair = []
  filepath.sort(key=lambda y: y[3],reverse=True)
  if pmid_only == True:
    filepath=list(filter(lambda c: np.isnan(c[1]) == False, filepath))
  for i in range(1,len(filepath),num):
    tp = 0
    fn = 0
    for j in range(1,i):
      if filepath[j][2] == 1:
        tp += 1
    for k in range(i+1,len(filepath)):
      if filepath[k][2] == 1:
        fn += 1
    pair.append((i,tp/(tp+fn)))
  return pair
