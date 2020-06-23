import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


def get_model():
    model = LogisticRegression(penalty='l2',
                               random_state=42,
                               max_iter=1000,
                               verbose=2,
                               n_jobs=-1)
    return model
    pass


def main():
    # Get data
    train_csv = '~/Data/Kaggle/real_or_not/train.csv'
    subm_csv = '~/Data/Kaggle/real_or_not/sample_submission.csv'

    train_data_file = 'train_data_64.npy'
    test_data_file = 'test_data_64.npy'

    train_dfx = pd.read_csv(train_csv)
    subm_dfx = pd.read_csv(subm_csv)

    train_mat = np.load('train_data.npy', allow_pickle=True)
    test_mat = np.load('test_data.npy', allow_pickle=True)

    # Get the model
    model = get_model()

    # Train model
    model.fit(train_mat, train_dfx['target'].values)

    # Score model
    y_hat = model.predict(test_mat)

    subm_dfx['target'] = y_hat

    # Save scores
    subm_dfx.to_csv('~/Data/Kaggle/real_or_not/submissions/subm_logistic_64.csv', index=False)
    pass


if __name__ == "__main__":
    main()
