import numpy as np
import pandas as pd
import gensim

model = gensim.models.KeyedVectors.load_word2vec_format(
            '~/Data/PretrainedModels/word2vec/GoogleNews-vectors-negative300.bin',
            binary=True
)

vocab = list(model.wv.vocab.keys())
print("Loaded the word2vec model")


def get_unit_vector(x):
    x_hat = x / np.linalg.norm(x)
    return x_hat


def sent2vec(sent_tok, mode='sum'):

    if len(sent_tok) == 0:
        sent_vec = np.random.randn(1, 300)
        return sent_vec

    vectors = [model[w] for w in sent_tok]
    vec = np.array(vectors)

    if mode == 'sum':
        sent_vec = np.sum(vec, axis=0, keepdims=True)
    if mode == 'avg':
        sent_vec = np.avg(vec, axis=0, keepdims=True)

    return sent_vec


def tokenize_sent(sent):
    sent_tok = sent.split(" ")
    sent_tok = [tok for tok in sent_tok if tok in vocab]
    return sent_tok


def vectorizer(sent):
    sent_tokens = tokenize_sent(sent)
    sent_vec = sent2vec(sent_tokens, mode='sum')
    unit_sent_vec = get_unit_vector(sent_vec)
    return unit_sent_vec


def main():
    train_csv = '~/Data/Kaggle/real_or_not/clean_train.csv'
    test_csv = '~/Data/Kaggle/real_or_not/clean_test.csv'

    output_train = 'train_s2v_300.npy'
    output_test = 'test_s2v_300.npy'

    train_dfx = pd.read_csv(train_csv)
    test_dfx = pd.read_csv(test_csv)

    n_dim = 300
    train_mat = np.zeros((train_dfx.shape[0], n_dim))
    test_mat = np.zeros((test_dfx.shape[0], n_dim))

    # Vectorize data
    for i in range(train_dfx.shape[0]):
        print("Train: ", i, "/", train_dfx.shape[0])
        sent = train_dfx['clean_text'].iloc[i]
        sent_vec = vectorizer(sent)
        train_mat[i, :] = sent_vec

    for i in range(test_dfx.shape[0]):
        print("Test: ", i, "/", test_dfx.shape[0])
        sent = test_dfx['clean_text'].iloc[i]
        sent_vec = vectorizer(sent)
        test_mat[i, :] = sent_vec

    # Save data
    np.save(output_train, train_mat)
    np.save(output_test, test_mat)


if __name__ == "__main__":
    main()
