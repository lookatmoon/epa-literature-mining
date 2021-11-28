# -*- coding: utf-8 -*-


def recall_at_k_curve_data(filepath,pmid_only,num):
  #when filepath is a data frame
  pair = []
  filepath.sort_values(by='Score', ascending=False)  
  if pmid_only == True:
    filepath=filepath.loc[filepath.PMID.notnull()]
    filepath=filepath.reset_index(drop=True)
    #filepath=filepath.drop(filepath[filepath['PMID'].notnull()].index,inplace=True)
  for i in range(1,len(filepath),num):
    tp = 0
    fn = 0
    for j in range(0,i):
      if filepath['Label'][j] == 1:
        tp += 1
    for k in range(i+1,len(filepath)):
      if filepath['Label'][k] == 1:
        fn += 1
    pair.append((i,tp/(tp+fn)))
  return pair


