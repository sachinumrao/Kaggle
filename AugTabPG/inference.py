import time
import scipy
import pandas as pd
import numpy as np
from lightgbm import LGBMRegressor
from sklearn import model_selection
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

def load_train_data(filename):
    df = pd.read_csv(filename)
    ycols = "loss"
    xcols = [col for col in df.columns if col.startswith("f")]
    return df[xcols], df[ycols]

def load_test_data(test_file):
    tdf = pd.read_csv(test_file)
    xcols = [col for col in tdf.columns if col.startswith("f")]
    return tdf[xcols]

def train_model(X_train, y_train):
    #iter-11 {'boosting_type': 'dart', 'num_leaves': 32, 'learning_rate': 0.06426629269440015, 
    # 'n_estimators': 499, 'reg_alpha': 0.8962770399146724, 'reg_lambda': 0.047391091546085624}
    
    # iter-17 {'boosting_type': 'gbdt', 'num_leaves': 52, 'learning_rate': 0.09630959938583214, 
    # 'n_estimators': 248, 'reg_alpha': 0.6960581589438712, 'reg_lambda': 0.2608984011941473}
    lgbm = LGBMRegressor(boosting_type="gbdt", 
                         num_leaves=64,
                         learning_rate=0.1,
                         n_estimators=1000, 
                         reg_alpha=0.7, 
                         reg_lambda=0.25,
                         max_depth=-1,
                         random_state=42,
                         silent=-1, 
                         verbose=-1)
    lgbm.fit(X_train, y_train)
    return lgbm

def score_model(model, xtest):
    y_preds = model.predict(xtest)
    return y_preds

def make_submission(submission_file, y_preds, output_folder):
    df = pd.read_csv(submission_file)
    df['loss'] = y_preds
    filepath = output_folder + "submission_" + str(time.time()) + ".csv"
    df.to_csv(filepath, index=False)
    
    
def main():
    train_file = "~/Data/Kaggle/AugTabPG/train.csv"
    test_file = "~/Data/Kaggle/AugTabPG/test.csv"
    subm_file = "~/Data/Kaggle/AugTabPG/sample_submission.csv"
    output_folder = "~/Data/Kaggle/AugTabPG/"
    
    x, y = load_train_data(train_file)
    
    model = train_model(x, y)
    xtest = load_test_data(test_file)
    y_test = score_model(model, xtest)
    
    make_submission(subm_file, y_test, output_folder)
    
## ToDo
# try transforms on target variable
# try pca
# try NN 
# try ensembling
    
if __name__ == "__main__":
    main()