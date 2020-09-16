import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from functools import partial
import optuna
from xgboost import XGBClassifier


def optimize(trial, x, y):
    
    # define xgboost params to tune
    booster = trial.suggest_categorical("booster", ['gbtree', 'dart'])
    eta = trial.suggest_uniform("eta", 0.01, 0.3)
    gamma = trial.suggest_uniform("gamma", 0, 5)
    max_depth = trial.suggest_int("max_depth", 4, 10)
    min_child_weight = trial.suggest_uniform("min_child_weight", 0, 5)
    max_delta_step = trial.suggest_int("max_delta_step", 0, 5)
    subsample = trial.suggest_uniform("subsample", 0.5, 1)
    colsample_bytree = trial.suggest_uniform("colsample_bytree", 0.5, 1)
    reg_lambda = trial.suggest_uniform("reg_lambda", 0, 2)
    reg_alpha = trial.suggest_uniform("reg_alpha", 0, 2)
    tree_method = trial.suggest_categorical("tree_method", ['auto', 'exact', 'approx', 'hist'])
    
    threshold = trial.suggest_uniform("threshold", 0.3, 0.7)

    model = XGBClassifier(booster=booster,
                          verbosity=1,
                          nthread=-1,
                          eta=eta,
                          gamma=gamma,
                          max_depth=max_depth,
                          min_child_weight=min_child_weight,
                          max_delta_step=max_delta_step,
                          subsample=subsample,
                          colsample_bytree=colsample_bytree,
                          reg_lambda=reg_lambda,
                          reg_alpha=reg_alpha,
                          tree_method=tree_method,
                          scale_pos_weight=0.2/0.8,
                          objective='binary:logistic',
                          metrics='auc',
                          seed=42
                          
                          )
    
    kf = StratifiedKFold(n_splits=5)

    accuracies = []
    for idx in kf.split(X=x, y=y):
        train_idx, test_idx = idx[0], idx[1]

        xtrain = x[train_idx]
        ytrain = y[train_idx]

        xtest = x[test_idx]
        ytest = y[test_idx]

        model.fit(xtrain, ytrain)
        preds = model.predict_proba(xtest)
        
        preds = (preds[:,0] < threshold).astype(np.int)
        fold_acc = accuracy_score(ytest, preds)
        accuracies.append(fold_acc)

    return -1.0 * np.mean(accuracies)

def main():
    # Load data
    file_folder = '~/Data/Kaggle/Titanic/'
    train_file = file_folder + 'train_processed.csv'
    
    df = pd.read_csv(train_file)
    y = df['Survived'].values
    x = df.drop(['Survived'], axis=1).values   
    
    # Create optuna study
    num_trials=200
    optimization_function = partial(optimize, x=x, y=y)
    study = optuna.create_study(direction='minimize')
    study.optimize(optimization_function, n_trials=num_trials)  
    

if __name__ == "__main__":
    main()
    




