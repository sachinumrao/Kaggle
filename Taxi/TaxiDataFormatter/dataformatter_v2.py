import pandas as pd
import numpy as np
from datetime import datetime
from math import radians, cos, sin, asin, sqrt

def __init__():
    print("Using DataFormatter Class")


def weekday(x):
    """
    Figures out the day of the week. Outputs 1 for monday,2 for tuesday and so on.
    """
    return (x.weekday()+1)

def is_weekend(x):
    """
    Figures out if it was weekend. Outputs 0 or 1 for weekday or weekend.
    """
    z = x.weekday()+1
    return z//6

def hourly_info(x):
    """
    separates the hour from time stamp. Returns hour of time.
    """
    
    n1 = x.hour
    return n1

def minute_info(x):
    """
    separates the minutes from time stamp. Returns minute of time.
    """
    
    n2 = x.minute
    return n2/60


def haversine(x):
   
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians 
    lon1 = x['pickup_longitude']
    lat1 = x['pickup_latitude']
    lon2 = x['dropoff_longitude']
    lat2 = x['dropoff_latitude']
    
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # haversine formula 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    r = 3956 # Radius of earth in kilometers. Use 3956 for miles
    return c * r

def formatter(train):

    #convert vendor id into one-hot
    df_id = pd.DataFrame()
    df_id['vendor_id'] = train['vendor_id']//2
    df_id[['id']] = train[['id']].copy()
    #print(df_id.head())

    #convert flag into one-hot
    tmp_df_flag = pd.get_dummies(train['store_and_fwd_flag'])
    df_flag = pd.DataFrame()
    df_flag[['flag_n', 'flag_y']] = tmp_df_flag.copy()
    df_flag = df_flag.drop(['flag_y'],axis=1)
    df_flag[['id']] = train[['id']].copy()
    #print(df_flag.head())

    df_weekday = pd.DataFrame()
    n = train.shape[0]

    #well-format the pickup time
    df_weekday['pickup_time'] = pd.to_datetime(train['pickup_datetime'], format="%Y-%m-%d %H:%M:%S")

    df_weekday['pickup_weekday'] = df_weekday['pickup_time'].apply(weekday)
    df_weekday['is_weekend'] = df_weekday['pickup_time'].apply(is_weekend)
    
    #print(df_weekday['pickup_weekday'].head())

    df_weekday['p_hour'] = df_weekday['pickup_time'].apply(hourly_info)
    df_weekday['p_min'] = df_weekday['pickup_time'].apply(minute_info)
    #print(df_weekday['p_time'].head())

    #Convert pick-up hour into categorical variables
    df_pickup_time = pd.DataFrame()
    #df_pickup_time = pd.get_dummies(df_weekday['p_hour'],prefix='p_hour', prefix_sep='_')
    df_weekday['p_hour'] = df_weekday['p_hour']/24
    
    df_pickup_time[['p_hour', 'p_min', 'is_weekend']] = df_weekday[['p_hour', 'p_min','is_weekend']]
    df_pickup_time[['id']] = train[['id']].copy()

    #Convert pick-up weekday into categorical variables
    df_pickup_weekday = pd.DataFrame()
    #df_pickup_weekday = pd.get_dummies(df_weekday['pickup_weekday'],prefix='p_wd', prefix_sep='_')
    df_pickup_weekday[['id']] = train[['id']].copy()

    #find the haversine distance between the pickup and dropoff points
    df_dist = pd.DataFrame()
    df_dist['dist'] = train.apply(haversine, axis=1)
    df_dist[['id']] = train[['id']].copy()

    #drop unnecessary columns from train
    dalist = ['vendor_id', 'pickup_datetime', 'store_and_fwd_flag','dropoff_longitude','dropoff_latitude']
    plist = list(train.columns)
    dlist = list(set(dalist).intersection(set(plist)))
    
    train = train.drop(dlist, axis=1)

    train = train.merge(df_dist,on=['id'])
    train = train.merge(df_flag, on=['id'])
    train = train.merge(df_id, on=['id'])
    train = train.merge(df_pickup_time, on=['id'])
    #train = train.merge(df_pickup_weekday, on=['id'])

    return train