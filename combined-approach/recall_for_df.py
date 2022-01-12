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


