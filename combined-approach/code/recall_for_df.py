# -*- coding: utf-8 -*-

# Note: the filepath input must be in the desired format: 
# (combined_result.csv can be an example)
# first column: hero_id/reference_id
# second column: PMID
# third column: Label (boolean value) (actual value of being cited)
# fourth column: Score (predicted probability of being cited)

def recall_at_k_curve_data(filepath,pmid_only,num):
  #when filepath is a data frame
  #pmid_only specifies whether we only count the articles with PMIDs or not
  #num is the number of articles to go through for each calculation of recall (suggested value: 1000)
  pair = []
  filepath_sorted = filepath.sort_values(by='Score', ascending=False,ignore_index = True)   
  if pmid_only == True:
    filepath_sorted=filepath_sorted.loc[filepath.PMID.notnull()]
    filepath_sorted=filepath_sorted.reset_index(drop=True)
    #filepath_sorted=filepath_sorted.drop(filepath_sorted[filepath_sorted['PMID'].notnull()].index,inplace=True)
  for i in range(1,len(filepath_sorted),num):
    tp = 0
    fn = 0
    for j in range(0,i):
      if filepath_sorted['Label'][j] == 1:
        tp += 1
    for k in range(i+1,len(filepath_sorted)):
      if filepath_sorted['Label'][k] == 1:
        fn += 1
    pair.append((i,tp/(tp+fn)))
  return pair


