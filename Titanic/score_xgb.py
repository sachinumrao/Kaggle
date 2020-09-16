import numpy as np
import pandas as pd
from xgboost import XGBClassifier


def score_model(file_folder, model, threshold):
    # get scoring data
    test_file = file_folder + 'test_processed.csv'
    df = pd.read_csv(test_file)
    data = df.values
    
    # score the model
    y_ = model.predict_proba(data)
    
    preds = (y_[:,0] < threshold).astype(np.int)
    
    # load submission file
    subm_file = file_folder + 'gender_submission.csv'
    subm = pd.read_csv(subm_file)
    
    # modify submission
    subm['Survived'] = preds
    
    # save submission file
    subm.to_csv(file_folder + 'rf_subm_v1.csv', index=False)


def get_best_model():
    
    booster = None
    eta = None
    gamma = None
    max_depth = None
    min_child_weight = None
    max_delta_step = None
    subsample = None
    colsample_bytree = None
    reg_lambda = None
    reg_alpha = None
    tree_method = None

    model = model = XGBClassifier(booster=booster,
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
    
    # create model instance with optimal params
    model = get_best_model()
      
    # retrain on full_data
    model.fit(x, y)
    
    # return trained model
    return model


def main():
    file_folder = '~/Data/Kaggle/Titanic/'
    
    threshold = 0.4844103598537662
    
    print("Training Model...")
    model = train_best_model(file_folder)
    
    print("Scoring Model...")
    score_model(file_folder, model, threshold)
    

if __name__ == "__main__":
    main()
    
