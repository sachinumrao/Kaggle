import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import RandomizedSearchCV


def get_model():
    model = xgb.XGBClassifier(random_state=42)
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

    x_train = np.load(train_data_file, allow_pickle=True)
    x_test = np.load(test_data_file, allow_pickle=True)

    y_train = train_dfx['target'].values

    # Get model
    model = get_model()

    # Parameter dictionary for hyper-parameter tuning
    params = {
        'eta': [0.01, 0.03, 0.05],
        'min_child_weight': [0.6, 0.8, 1.0],
        'max_depth': [5, 6, 7, 8],
        'max_leaf_node': [4, 6, 8],
        'gamma': [0.0, 0.1, 0.2],
        'subsample': [0.7, 0.8, 0.8],
        'colsample_bytree': [0.8, 1.0],
        'lambda': [0.5, 1.0, 1.5],
        'alpha': [0.5, 0.1],
        'n_estimators': [200, 300, 400],
    }

    # Train model
    cross_val = RepeatedStratifiedKFold(n_splits=5,
                                        n_repeats=1,
                                        random_state=42)

    random_cv = RandomizedSearchCV(estimator=model,
                                   param_distributions=params,
                                   n_iter=10,
                                   n_jobs=-1,
                                   cv=cross_val,
                                   scoring='roc_auc',
                                   refit=True,
                                   verbose=2,
                                   return_train_score=True)

    random_cv.fit(x_train, y_train)

    # Save hyper parameter tuning results
    cv_df = pd.DataFrame.from_dict(random_cv.cv_results_)
    cv_df.to_csv('~/Data/Kaggle/real_or_not/cross_val_summary_1.csv', index=False)

    # Print best params
    print("Best Params: ")
    print(random_cv.best_params_)

    # Score the model
    best_model = random_cv.best_estimator_
    y_preds = best_model.predict(x_test)
    subm_dfx['target'] = y_preds

    # Save scores
    subm_dfx.to_csv('~/Data/Kaggle/real_or_not/submissions/subm_xgb_64.csv', index=False)
    pass


if __name__ == "__main__":
    main()
