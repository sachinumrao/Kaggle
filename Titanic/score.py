import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


def score_model(file_folder, model):
    # get scoring data
    test_file = file_folder + 'test_processed.csv'
    df = pd.read_csv(test_file)
    data = df.values
    
    # score the model
    y_ = model.predict(data)
    
    # load submission file
    subm_file = file_folder + 'submission.csv'
    subm = pd.read_csv(subm_file)
    
    # modify submission
    subm[''] = y_
    
    # save submission file
    subm.to_csv(file_folder + 'rf_subm.csv', index=False)


def get_best_model():
    model = None
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
    file_folder = '~/Data/Kaggle/Titanic'
    
    model = train_best_model(file_folder)
    score_model(file_folder, model)
    

if __name__ == "__main__":
    main()