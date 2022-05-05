# Text-based Approach

In the data exploration process,
    we count the most frequent unigrams/bigrams in the 2013/2020 titles and abstracts and perform topic modeling).
    Please run data_exploration.py.

In the training process,
    we extract textual features and train the classifier.
    Please run text_based_LR.py.
    
In the fine grained method:
    We divide the training set into 16 portions by chapters (or groups of sections) and train these independent models separately.

## Some tools used in this pipeline:

Topic Model: Latent Dirichlet Allocation (LDA).

Classifier: Logistic regression.

## Datasets

Dataset will be released after obtaining permission from providers(U.S. Environmental Protection Agency).
