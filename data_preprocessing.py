import pandas as pd
import numpy as np

fact_values_df = pd.read_csv('Files/factValues.csv', header=None)
fact_values_df.columns = ['source_id', 'sentence_id', 'event_pred', 'fact_label']

sentences_df = pd.read_csv('Files/sentences.csv', header=None)
sentences_df.columns = ['source_id', 'sentence_id', 'sentence']

# merge dataframes and shuffle records
data = pd.merge(sentences_df, fact_values_df, how='inner', on=['source_id', 'sentence_id']).sample(frac=1)

# remove entries in which additional predicates in the same sentence are considered
data = data.drop_duplicates(['source_id', 'sentence_id'], keep= 'first')

# eliminate records for which factuality label is 'Uu'
data.drop(data.loc[data['fact_label'] == 0].index, inplace=True)

# create a positive integer mapping for the labels
data.fact_label.replace([-3, -2, -1, 1, 2, 3], [0, 1, 2, 3, 4, 5], inplace=True)

data.drop(['source_id', 'sentence_id'], axis=1, inplace=True)

# save processed data to a csv file
data.to_csv('factData.csv', index = False, header = True)