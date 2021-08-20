import time
import pandas as pd
import numpy as np
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

def load_data(file_name):
    """
    Load dataset from csv file and split in train and test set
    """
    data = pd.read_csv(file_name)
    ycols = "loss"
    xcols = [x for x in data.columns if x.startswith("f")]
    # split dataset
    xtrain, xtest, ytrain, ytest = train_test_split(data[xcols], 
                                                    data[ycols], 
                                                    test_size=0.2, 
                                                    random_state=42)
    return (xtrain, xtest, ytrain, ytest)

def train_lightgbm_model(xtrain, ytrain, xtest, ytest):
    """
    Train lightgbm regressor model and evaluate it
    """
    # create model
    model = LGBMRegressor()
    
    # train model
    model.fit(xtrain, ytrain)
    
    # evaluate model
    ypred = model.predict(xtest)
    rmse = np.sqrt(mean_squared_error(ytest, ypred))
    return rmse

def train_randomforest_model(xtrain, ytrain, xtest, ytest):
    """
    Train lightgbm regressor model and evaluate it
    """
    # create model
    model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1, verbose=1)
    
    # train model
    model.fit(xtrain, ytrain)
    
    # evaluate model
    ypred = model.predict(xtest)
    rmse = np.sqrt(mean_squared_error(ytest, ypred))
    return rmse

def train_linear_model(xtrain, ytrain, xtest, ytest):
    """
    Train linear regression model
    """
    # create model
    model = LinearRegression()
    
    # train model
    model.fit(xtrain, ytrain)
    
    # evaluate model
    ypred = model.predict(xtest)
    rmse = np.sqrt(mean_squared_error(ytest, ypred))
    return rmse
    
    
def main():
    filename = "~/Data/Kaggle/AugTabPG/train.csv"
    xtrain, xtest, ytrain, ytest = load_data(filename)
    test_error = train_linear_model(xtrain, ytrain, xtest, ytest)
    print(f"Test Error: {test_error}")
    
    # get baseline performance using mean of train set
    ymean = np.mean(ytrain)
    y_pred = np.array([ymean]*len(ytrain))
    rmse = np.sqrt(mean_squared_error(ytrain, y_pred))
    print(f"Baseline Error: {rmse}")
    
if __name__ == "__main__":
    main()