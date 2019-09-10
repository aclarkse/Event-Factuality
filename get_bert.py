def get_bert(word2idx):
    """Returns a BERT embedding of the word tokens.
    
    Args:
        word2idx (dict): word to index pairs {'word': index}
    Returns:
        embedding_matrix (np.array): embedding matrix of shape (vocabulary_size + 1, embedding_dim=3072)
    
    """
    embedding_matrix = np.zeros((vocabulary_size + 1, 3072))
    bert = BertEmbeddings()
    for word, index in tqdm(word2idx.items()):
        try:
            word_ = Sentence(word)
            bert.embed(word_)
            embedding_vector = word_[0].embedding.cpu().detach().numpy()
            embedding_matrix[index] = embedding_vector
        except KeyError:
            embedding_matrix[index] = np.random.normal(0, np.sqrt(0.25), 3072)
    
    return embedding_matrix


# save embeddings
pickle_out = open("bert_3072.pickle","wb")
pickle.dump(embedding_matrix, pickle_out)
pickle_out.close()