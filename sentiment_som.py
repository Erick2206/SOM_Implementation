import numpy as np
import data_helpers
from w2v import train_word2vec

np.random.seed(0)


def load_data():
    x, y, vocabulary, vocabulary_inv_list = data_helpers.load_data()
    vocabulary_inv = {key: value for key, value in enumerate(vocabulary_inv_list)}
    y = y.argmax(axis=1)
        # Shuffle data
    shuffle_indices = np.random.permutation(np.arange(len(y)))
    x = x[shuffle_indices]
    y = y[shuffle_indices]
    train_len = int(len(x) * 0.9)
    x_train = x[:train_len]
    y_train = y[:train_len]
    x_test = x[train_len:]
    y_test = y[train_len:]
    embedding_weights = train_word2vec(np.vstack((x_train, x_test)), vocabulary_inv, num_features=300, min_word_count=1, context=10)
    x_train=np.array([[embedding_weights[j] for j in i] for i in x_train])
    x_test=np.array([[embedding_weights[j] for j in i] for i in x_test])

    return x_train, y_train, x_test, y_test, vocabulary_inv

if __name__=="__main__":
    # Data Preparation
    print("Load data...")
    x_train, y_train, x_test, y_test, vocabulary_inv = load_data()
    print x_train[0]
    print vocabulary_inv[1]
    print y_train[0]

    print("x_train shape:", x_train.shape)
    print("x_test shape:", x_test.shape)
    print("Vocabulary Size: {:d}".format(len(vocabulary_inv)))
