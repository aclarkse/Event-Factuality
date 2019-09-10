from flair.embeddings import Sentence, ELMoEmbeddings
import numpy as np
import pickle
import tqdm as tqdm


def get_elmo(word2idx):
    embedding_matrix = np.zeros((vocabulary_size + 1, 3072))
    elmo = ELMoEmbeddings()
    for word, index in tqdm(word2idx.items()):
        try:
            word_ = Sentence(word)
            elmo.embed(word_)
            embedding_vector = word_[0].embedding.cpu().detach().numpy()
            embedding_matrix[index] = embedding_vector
        except KeyError:
            embedding_matrix[index] = np.random.normal(0, np.sqrt(0.25), 3072)
    
    return embedding_matrix

# save embeddings
pickle_out = open("elmo_3072.pickle","wb")
pickle.dump(embedding_matrix, pickle_out)
pickle_out.close()