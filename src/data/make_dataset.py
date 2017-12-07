import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import pickle
import statsmodels.formula.api as smf
import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose
import seaborn as sns
import pandas_profiling
import datetime
import sqlite3
import calendar
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

def createDf():

    # We willl load all the csv files into Pandas dataframes, properly parsing dates
    air_reserve = pd.read_csv('../data/raw/air_reserve.csv',parse_dates=['visit_datetime','reserve_datetime'])
    hpg_reserve = pd.read_csv('../data/raw/hpg_reserve.csv',parse_dates=['visit_datetime','reserve_datetime'])
    air_store_info = pd.read_csv('../data/raw/air_store_info.csv')
    hpg_store_info = pd.read_csv('../data/raw/hpg_store_info.csv')
    store_relation = pd.read_csv('../data/raw/store_id_relation.csv')
    date_info = pd.read_csv('../data/raw/date_info.csv',parse_dates=['calendar_date'])
    air_visit = pd.read_csv('../data/raw/air_visit_data.csv',parse_dates=['visit_date'])

    hpg_reserve['visit_year'] = hpg_reserve['visit_datetime'].dt.year
    hpg_reserve['visit_month'] = hpg_reserve['visit_datetime'].dt.month
    hpg_reserve['visit_day'] = hpg_reserve['visit_datetime'].dt.day
    hpg_reserve['reserve_year'] = hpg_reserve['reserve_datetime'].dt.year
    hpg_reserve['reserve_month'] = hpg_reserve['reserve_datetime'].dt.month
    hpg_reserve['reserve_day'] = hpg_reserve['reserve_datetime'].dt.day

    hpg_reserve.drop(['visit_datetime','reserve_datetime'], axis=1, inplace=True)
    hpg_reserve = hpg_reserve.groupby(['hpg_store_id', 'visit_year', 'visit_month','visit_day','reserve_year','reserve_month','reserve_day'], as_index=False).sum()

    # We should also prepair the rest of the files to get merged by visit day.
    air_reserve['visit_year'] = air_reserve['visit_datetime'].dt.year
    air_reserve['visit_month'] = air_reserve['visit_datetime'].dt.month
    air_reserve['visit_day'] = air_reserve['visit_datetime'].dt.day
    air_reserve['reserve_year'] = air_reserve['reserve_datetime'].dt.year
    air_reserve['reserve_month'] = air_reserve['reserve_datetime'].dt.month
    air_reserve['reserve_day'] = air_reserve['reserve_datetime'].dt.day

    air_reserve.drop(['visit_datetime','reserve_datetime'], axis=1, inplace=True)

    date_info['calendar_year'] = date_info['calendar_date'].dt.year
    date_info['calendar_month'] = date_info['calendar_date'].dt.month
    date_info['calendar_day'] = date_info['calendar_date'].dt.day

    date_info.drop(['calendar_date'], axis=1, inplace=True)

    air_visit['visit_year'] = air_visit['visit_date'].dt.year
    air_visit['visit_month'] = air_visit['visit_date'].dt.month
    air_visit['visit_day'] = air_visit['visit_date'].dt.day

    air_visit.drop(['visit_date'], axis=1, inplace=True)

    # Merging the data
    # Now that the data is prepared to be merged, we need to add all the columns to the air_reserve file, as it is the file connected to all the rest of them by one or other way.
    # First, we merge all the reserves from both systems into the air_reserve file.

    hpg_reserve = pd.merge(hpg_reserve, store_relation, on='hpg_store_id', how='inner')
    hpg_reserve.drop(['hpg_store_id'], axis=1, inplace=True)

    air_reserve = pd.concat([air_reserve, hpg_reserve])

    # Now we can downsalmple to daily visit days, adding all the reserves made for that specific date.
    air_reserve = air_reserve.groupby(['air_store_id', 'visit_year', 'visit_month','visit_day'], as_index=False).sum().drop(['reserve_day','reserve_month','reserve_year'], axis=1)

    # We can easily add the holiday info to our dataframe.
    air_reserve = pd.merge(air_reserve, date_info, left_on=['visit_year','visit_month','visit_day'], right_on=['calendar_year','calendar_month','calendar_day'], how='left')
    air_reserve.drop(['calendar_year','calendar_month','calendar_day'], axis=1, inplace=True)

    # And merge also the store information and the restaurant visits per day. At this point, we're going to create a new dataframe, df, to mark the moment where we have all the data together.
    air_reserve = pd.merge(air_reserve, air_store_info, on='air_store_id', how='left')

    df = pd.merge(air_reserve, air_visit, on=['air_store_id','visit_year','visit_month','visit_day'], how='left')

    df.air_genre_name = df.air_genre_name.replace(' ', '_', regex=True)
    df.air_genre_name = df.air_genre_name.replace('/', '_', regex=True)
    df=df.rename(columns = {'air_genre_name':'genre','day_of_week':'dow'})

    df.sort_values(by=['visit_year','visit_month','visit_day','air_store_id'], ascending=[True,True,True,True], inplace=True)

    return df
