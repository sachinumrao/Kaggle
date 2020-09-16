import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier


def score_model(file_folder, model, threshold, scaler):
    # get scoring data
    test_file = file_folder + 'test_processed.csv'
    df = pd.read_csv(test_file)
    data = df.values
    
    data = scaler.transform(data)
    
    # score the model
    y_ = model.predict_proba(data)
    
    preds = (y_[:,0] < threshold).astype(np.int)
    
    # load submission file
    subm_file = file_folder + 'gender_submission.csv'
    subm = pd.read_csv(subm_file)
    
    # modify submission
    subm['Survived'] = preds
    
    # save submission file
    subm.to_csv(file_folder + 'xgb_subm_v4.csv', index=False)


def get_best_model():
    n_estimators = 411
    booster = 'gbtree'
    eta = 0.3523566
    gamma = 0.451301
    max_depth = 10
    min_child_weight = 3.057999
    max_delta_step = 4
    subsample = 0.8810433666
    colsample_bytree = 0.85334
    reg_lambda = 0.888965
    reg_alpha = 0.8341519
    tree_method = 'approx'

    model = model = XGBClassifier(booster=booster, n_estimators=n_estimators,
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
    return model

    
def train_best_model(file_folder):
    
    # load training data
    train_file = file_folder + 'train_processed.csv'
    df = pd.read_csv(train_file)
    y = df['Survived'].values
    x = df.drop(['Survived'], axis=1).values
    
    # scale the data
    scaler = MinMaxScaler()
    scaler.fit(x)
    x = scaler.transform(x)
    # create model instance with optimal params
    model = get_best_model()
      
    # retrain on full_data
    model.fit(x, y)
    
    # return trained model
    return model, scaler


def main():
    file_folder = '~/Data/Kaggle/Titanic/'
    
    threshold = 0.66090088097
    
    print("Training Model...")
    model, scaler = train_best_model(file_folder)
    
    print("Scoring Model...")
    score_model(file_folder, model, threshold, scaler)
    

if __name__ == "__main__":
    main()
    
