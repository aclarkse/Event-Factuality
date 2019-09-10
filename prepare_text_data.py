import pandas as pd
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

# read-in and prepare data
data = pd.read_csv('factData.csv').sample(frac=1)
sentences = data['sentence'].values
labels = data['fact_label'].values.reshape(-1, 1)

# convert sentences into integers
tokenizer = Tokenizer()
tokenizer.fit_on_texts(sentences)
sequences = tokenizer.texts_to_sequences(sentences)


def max_len(sentences):
    """Finds the maximum sequence length among the processed sentences.

    Args:
        sentences (list): list of integer-encoded sentences
    Returns:
        max_len (int): the maximum sequence length

    """
    lengths = []
    for sentence in sentences:
        lengths.append(len(sentence))
    
    lengths_arr = np.asarray(lengths)
    max_len = np.amax(lengths_arr)
    
    return max_len


# pad the sequences so that we have an N x T matrix
data = pad_sequences(sequences, maxlen=max_seq_len)

# make train-val-test split
X_train, X_test, Y_train, Y_test = train_test_split(data, labels, test_size=0.2, random_state=42)
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.2, random_state=42)

print('Train input shape:', X_train.shape)
print('Train label length:', len(Y_train))
print('Val input shape:', X_val.shape)
print('Val label length:', len(Y_val))
print('Test input shape:', X_test.shape)
print('Test label length:', len(Y_test))