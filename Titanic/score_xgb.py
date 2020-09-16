import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


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
    
    model = RandomForestClassifier(criterion='gini',
                                   n_estimators=402,
                                   max_depth=8,
                                   max_features=0.686454,
                                   random_state=42)
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
    
    
## 'criterion': 'gini', 'n_estimators': 402, 'max_depth': 8, 'max_features': 0.6864543886802769, 'threshold': 0.4844103598537662