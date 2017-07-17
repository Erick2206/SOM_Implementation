from gensim.models import KeyedVectors,word2vec
from os.path import join, exists, split
import os
import numpy as np

def train_word2vec(sentence_matrix, vocabulary_inv,
                   num_features=300, min_word_count=1, context=10):
    """
    Trains, saves, loads Word2Vec model
    Returns weights for embedding layer.

    inputs:
    sentence_matrix # int matrix: num_sentences x max_sentence_len
    vocabulary_inv  # dict {int: str}
    """
    model_dir = 'models'
    model_name = "{:d}features_{:d}minwords_{:d}context".format(num_features, min_word_count, context)
    model_name = join(model_dir, model_name)
    if exists(model_name):
        embedding_weights = KeyedVectors.load(model_name)
        print('Load existing Word2Vec model \'%s\'' % split(model_name)[-1])
    else:
        print "Making new vector"
        embedding_weights=KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)
        # Saving the model for later use. You can load it later using Word2Vec.load()
        if not exists(model_dir):
            os.mkdir(model_dir)
        print('Saving Word2Vec model \'%s\'' % split(model_name)[-1])
        embedding_weights.save(model_name)

    # add unknown words
    embedding_weights = {key: embedding_weights[word] if word in embedding_weights else
                              np.random.uniform(-0.25, 0.25, embedding_weights.vector_size)
                         for key, word in vocabulary_inv.items()}

    return embedding_weights


if __name__ == '__main__':
    import data_helpers
    print("Loading data...")
    x, _, _, vocabulary_inv_list = data_helpers.load_data()
    vocabulary_inv = {key: value for key, value in enumerate(vocabulary_inv_list)}
    w = train_word2vec(x, vocabulary_inv)
