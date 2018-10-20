import pandas as pd
import numpy as np
from datetime import datetime


def __init__():
    print("Using DataFormatter Class")


def weekday(x):
    return (x.weekday()+1)

def hourly_info(x):
    n1 = x.hour
    n2 = x.minute/60
    n3 = n1+n2
    return n1

def distance(x):
    from math import sin, cos, sqrt, atan2, radians

    # approximate radius of earth in km
    R = 6373.0

    lat1 = radians(abs(x['pickup_latitude']))
    lon1 = radians(abs(x['pickup_longitude']))
    lat2 = radians(abs(x['dropoff_latitude']))
    lon2 = radians(abs(x['dropoff_longitude']))

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    dist = R * c
    
    return dist

def formatter(train):

    #convert vendor id into one-hot
    tmp_df_id = pd.get_dummies(train['vendor_id'])
    df_id = pd.DataFrame()
    df_id[['vendor_1', 'vendor_2']] = tmp_df_id.copy()
    df_id[['id']] = train[['id']].copy()
    #print(df_id.head())

    #convert flag into one-hot
    tmp_df_flag = pd.get_dummies(train['store_and_fwd_flag'])
    df_flag = pd.DataFrame()
    df_flag[['flag_n', 'flag_y']] = tmp_df_flag.copy()
    df_flag[['id']] = train[['id']].copy()
    #print(df_flag.head())

    df_weekday = pd.DataFrame()
    n = train.shape[0]

    df_weekday['pickup_time'] = pd.to_datetime(train['pickup_datetime'], format="%Y-%m-%d %H:%M:%S")

    df_weekday['pickup_weekday'] = df_weekday['pickup_time'].apply(weekday)
    #print(df_weekday['pickup_weekday'].head())

    df_weekday['p_time'] = df_weekday['pickup_time'].apply(hourly_info)
    #print(df_weekday['p_time'].head())


    #df_weekday['dropoff_time'] = pd.to_datetime(train['dropoff_datetime'], format="%Y-%m-%d %H:%M:%S")

    #df_weekday['dropoff_weekday'] = df_weekday['dropoff_time'].apply(weekday)
    #print(df_weekday['pickup_weekday'].head())

    #df_weekday['d_time'] = df_weekday['dropoff_time'].apply(hourly_info)

    #Convert pick-up hour into categorical variables
    #df_pickup_time = pd.get_dummies(df_weekday['p_time'],prefix='p_time', prefix_sep='_')
    df_pickup_time = pd.DataFrame()

    df_pickup_time[['p_time']] = df_weekday[['p_time']].copy()
    df_pickup_time[['id']] = train[['id']].copy()

    #Convert pick-up weekday into categorical variables
    #df_pickup_weekday = pd.get_dummies(df_weekday['pickup_weekday'],prefix='p_wd', prefix_sep='_')
    df_pickup_weekday = pd.DataFrame()
    df_pickup_weekday[['pickup_weekday']] = df_weekday[['pickup_weekday']].copy()
    df_pickup_weekday[['id']] = train[['id']].copy()

    #Convert drop-off hour into categorical variables
    #df_dropoff_time = pd.get_dummies(df_weekday['d_time'],prefix='d_time', prefix_sep='_')
    #df_dropoff_time[['id']] = train[['id']].copy()

    #Convert drop-off weekday into categorical variables
    #df_dropoff_weekday = pd.get_dummies(df_weekday['dropoff_weekday'],prefix='d_wd', prefix_sep='_')
    #df_dropoff_weekday[['id']] = train[['id']].copy()

    df_dist = pd.DataFrame()
    df_dist['dist'] = train.apply(distance, axis=1)
    df_dist[['id']] = train[['id']].copy()

    #drop unnecessary columns from train
    dlist = ['vendor_id', 'pickup_datetime', 'store_and_fwd_flag','pickup_longitude','dropoff_longitude','pickup_latitude','dropoff_latitude']
    train = train.drop(dlist, axis=1)

    train = train.merge(df_dist,on=['id'])
    #train = train.merge(df_dropoff_time, on=['id'])
    #train = train.merge(df_dropoff_weekday, on=['id'])
    train = train.merge(df_flag, on=['id'])
    train = train.merge(df_id, on=['id'])
    train = train.merge(df_pickup_time, on=['id'])
    train = train.merge(df_pickup_weekday, on=['id'])

    return train