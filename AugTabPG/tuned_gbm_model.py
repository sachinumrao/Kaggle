import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn import linear_model
import sklearn
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import StratifiedKFold
from functools import partial
import optuna


def optimize(trial, x, y):
    
    # define parameters
    boosting_type = trial.suggest_categorical(
        "boosting_type", ['gbdt', 'dart', 'goss'])
    num_leaves = trial.suggest_int("num_leaves", 32, 64)
    learning_rate = trial.suggest_uniform(
        "learning_rate", 0.01, 0.2)
    n_estimators = trial.suggest_int("n_estimators", 100, 500)
    reg_alpha = trial.suggest_uniform("reg_alpha", 0, 1)
    reg_lambda = trial.suggest_uniform("reg_lambda", 0, 1)
    max_depth = -1
    random_state = 42
    
    model = LGBMRegressor(boosting_type=boosting_type,
                          num_leaves=num_leaves,
                          learning_rate=learning_rate,
                          n_estimators=n_estimators,
                          reg_alpha=reg_alpha,
                          reg_lambda=reg_lambda,
                          max_depth=max_depth,
                          random_state=random_state)

    kf = StratifiedKFold(n_splits=5)

    accuracies = []
    for idx in kf.split(X=x, y=y):
        train_idx, test_idx = idx[0], idx[1]

        xtrain = x[train_idx]
        ytrain = y[train_idx]

        xtest = x[test_idx]
        ytest = y[test_idx]

        model.fit(xtrain, ytrain)
        preds = model.predict(xtest)
        
        rmse = np.sqrt(mean_squared_error(ytest, preds))
        accuracies.append(rmse)

    return np.mean(accuracies)

def main():
    # Load data
    file_folder = '~/Data/Kaggle/AugTabPG/'
    train_file = file_folder + 'train.csv'
    
    df = pd.read_csv(train_file)
    
    ycols = "loss"
    xcols = [col for col in df.columns if col.startswith("f")]
    
    y = df[ycols].values
    x = df[xcols].values   
    
    # Create optuna study
    optimization_function = partial(optimize, x=x, y=y)
    study = optuna.create_study(direction='minimize')
    study.optimize(optimization_function, n_trials=200)  
    

if __name__ == "__main__":
    main()