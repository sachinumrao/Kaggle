import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from functools import partial
import optuna
from xgboost import XGBClassifier


def optimize(trial, x, y):
    entropy = trial.suggest_categorical("criterion", ['gini', 'entropy'])
    n_estimators = trial.suggest_int("n_estimators", 200, 600)
    max_depth = trial.suggest_int("max_depth", 5, 10)
    max_features = trial.suggest_uniform("max_features", 0.01, 1.0)
    threshold = trial.suggest_uniform("threshold", 0.3, 0.7)

    model = XGBClassifier()
    
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
    optimization_function = partial(optimize, x=x, y=y)
    study = optuna.create_study(direction='minimize')
    study.optimize(optimization_function, n_trials=150)  
    

if __name__ == "__main__":
    main()
    



