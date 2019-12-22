import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


train_df = pd.read_csv('./data/train.csv')

remove_feats = ["MSSubClass", "MSZoning", "Street", "Alley",
                "LotShape", "LandContour", "Utilities", "LotConfig",
                "LandSlope", "Neighborhood", "Condition1", "Condition2",
                "BldgType", "HouseStyle", "BuildTypeCat", "YearBuilt",
                "YearRemodAdd", "RemodCat", "", ""]

train_df = pd.concat([train_df, pd.get_dummies(train_df['MSSubClass'], 
            prefix='mssubclass', dummy_na=True)], axis=1)

train_df = pd.concat([train_df, pd.get_dummies(train_df['MSZoning'], 
            prefix='mszoning', dummy_na=True)], axis=1)

train_df["LotFrontage"] = train_df["LotFrontage"].fillna(
                            train_df["LotFrontage"].mean())

train_df["LotFrontage"] = train_df["LotFrontage"].clip(upper=150)

train_df["LotArea"] = train_df["LotArea"].apply(np.log)
train_df["LotArea"] = train_df["LotArea"].clip(upper=11)

train_df = pd.concat([train_df, pd.get_dummies(train_df['Alley'], 
            prefix='alley', dummy_na=True)], axis=1)

train_df = pd.concat([train_df, pd.get_dummies(train_df['Street'], 
            prefix='street', dummy_na=True)], axis=1)

train_df = pd.concat([train_df, pd.get_dummies(train_df['LotShape'], 
            prefix='lotshape', dummy_na=True)], axis=1)

train_df = pd.concat([train_df, pd.get_dummies(train_df['LandContour'], 
            prefix='landcontour', dummy_na=True)], axis=1)

train_df = pd.concat([train_df, pd.get_dummies(train_df['LotConfig'], 
            prefix='lotconfig', dummy_na=True)], axis=1)

train_df = pd.concat([train_df, pd.get_dummies(train_df['LandSlope'], 
            prefix='landslope', dummy_na=True)], axis=1)

train_df = pd.concat([train_df, pd.get_dummies(train_df['Neighborhood'], 
            prefix='neighborhood', dummy_na=True)], axis=1)

train_df = pd.concat([train_df, pd.get_dummies(train_df['Condition1'], 
            prefix='condn1', dummy_na=True)], axis=1)

train_df = pd.concat([train_df, pd.get_dummies(train_df['Condition2'], 
            prefix='condn2', dummy_na=True)], axis=1)

train_df = pd.concat([train_df, pd.get_dummies(train_df['BldgType'], 
            prefix='condn1', dummy_na=True)], axis=1)

train_df = pd.concat([train_df, pd.get_dummies(train_df['HouseStyle'], 
            prefix='condn2', dummy_na=True)], axis=1)


def categorize_years(x):
    if x < 1950:
        return "old"
    elif x >= 1950 and x < 1995:
        return "medium"
    else:
        return "new"

train_df["BuildTypeCat"] = train_df["YearBuilt"].apply(categorize_years)

train_df = pd.concat([train_df, pd.get_dummies(train_df['BuildTypeCat'], 
            prefix='buildtypecat', dummy_na=True)], axis=1)

def categorize_remod(x):
    if x < 1980:
        return "old"
    elif x >= 1980 and x < 2000:
        return "medium"
    else:
        return "new"

train_df["RemodCat"] = train_df["YearRemodAdd"].apply(categorize_remod)

train_df = pd.concat([train_df, pd.get_dummies(train_df['RemodCat'], 
            prefix='remod', dummy_na=True)], axis=1)

train_df = pd.concat([train_df, pd.get_dummies(train_df['RoofStyle'], 
            prefix='roofstyle', dummy_na=True)], axis=1)

train_df = pd.concat([train_df, pd.get_dummies(train_df['RoofMatl'], 
            prefix='roofmatl', dummy_na=True)], axis=1)

train_df = pd.concat([train_df, pd.get_dummies(train_df['Exterior1st'], 
            prefix='exterior1', dummy_na=True)], axis=1)

train_df = pd.concat([train_df, pd.get_dummies(train_df['Exterior2nd'], 
            prefix='exterior2', dummy_na=True)], axis=1)

train_df = pd.concat([train_df, pd.get_dummies(train_df['MasVnrType'], 
            prefix='mvtype', dummy_na=True)], axis=1)