# epa-literature-mining
# introduction
This work introduces the ISA literature screening
dataset and the associated research challenges to the information
and knowledge management community. Our pilot experiments
show that combining multiple approaches in tackling this challenge
is both promising and necessary. 

# Data
The dataset is publicly available
at https://ils.unc.edu/~wangyue/isa-dataset
Preprocessed data can be found here for running text based LR
https://drive.google.com/drive/folders/1wkQN4_byG8uVf6AduA8nsC3QHN5ZOXH6
Data for network based approach can be found here:
https://ils.unc.edu/~wangyue/isa-dataset/isa-data/reference_metadata_2020.csv

# Methods and ranker generator
Text-based Simple Ranker
text_based_simple_LR.py
Text-based Ensemble Ranker
text_based_ensemble_LR.py
Network-based Ranker.
network_based_LR.py
Context Paragraph-based Ranker

Combined Ranker
combined_approach.py

# Graph generator for all methods
combined_approach.py
This file will not only generator ranker for combined approach. 
Line 63-77 will generate recall graphs for all methods used.
All ranker scores generated from other methods can be found under combined-approach folder
