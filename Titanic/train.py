import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from functools import partial
import optuna


def optimize(trial, x, y):
    entropy = trial.suggest_categorical("criterion", ['gini', 'entropy'])
    n_estimators = trial.suggest_int("n_estimators", 200, 600)
    max_depth = trial.suggest_int("max_depth", 5, 10)
    max_features = trial.suggest_uniform("max_features", 0.01, 1.0)
    threshold = trial.suggest_uniform("threshold", 0.3, 0.7)

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        max_features=max_features,
        criterion=entropy,
        n_jobs=-1,
        random_state=42
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
        
        yhat = (preds[:,0] < threshold).astype(np.int)
        # fold_acc = accuracy_score(ytest, preds)
        fold_auc = roc_auc_score(ytest, preds[:,0])
        accuracies.append(fold_auc)

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
    




