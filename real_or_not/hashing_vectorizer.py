import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import HashingVectorizer


def get_hash_vectorizer(fname, n=20):
    dfx = pd.read_csv(fname)
    sentence_list = [sent for sent in dfx['clean_text'].values]
    hasher = HashingVectorizer(n_features=n)
    hasher.fit(sentence_list)
    return hasher
    pass


def get_hash_data(fname, hasher):
    dfx = pd.read_csv(fname)
    sentence_list = [sent for sent in dfx['clean_text'].values]
    data = hasher.transform(sentence_list)
    return data.toarray()


def main():
    trainfile = '~/Data/Kaggle/real_or_not/clean_train.csv'
    testfile = '~/Data/Kaggle/real_or_not/clean_test.csv'

    hasher = get_hash_vectorizer(trainfile, n=64)

    train_hash = get_hash_data(trainfile, hasher)
    test_hash = get_hash_data(testfile, hasher)

    # save the hash
    np.save('train_data_64.npy', train_hash)
    np.save('test_data_64.npy', test_hash)
    pass


if __name__ == "__main__":
    main()
