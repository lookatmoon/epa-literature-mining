# Combined-based Approach

In this section, we combine the results gained from text-based and network-based approaches and re-rank the articles.

For calculating the combined rank/score: run combined_approach.py. 

For computing the recall of a list or dataframe, use the function in recall_for_list.py or recall_for_df.py.

Note: the list or dataframe must be in the desired format: 
(combined_result.csv is an exmaple)

first column: hero_id/reference_id

second column: PMID

third column: Label (boolean value) (actual value of being cited)

fourth column: Score (predicted probability of being cited)

## Datasets

Dataset will be released after obtaining permission from providers(U.S. Environmental Protection Agency).
