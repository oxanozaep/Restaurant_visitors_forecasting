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

def submission_format(row):
    return row['air_store_id']+"_"+str(row['visit_year'])+"-"+str(row['visit_month']).zfill(2)+"-"+str(row['visit_day']).zfill(2)


def prediction_creation(data_train, data_test, models_dict, half_models_dict,nodata_model):

    # Generation of predictions and submission file
    #
    # With all the models created, we can finally predict the visitors for the test dataset.
    # We'll start by filtering the test dataset by the first predicted date.

    data_test.set_index(pd.to_datetime(data_test.visit_year*10000+data_test.visit_month*100+data_test.visit_day,format='%Y%m%d'), inplace=True)
    data_test = data_test[data_test.index > '2017-04-22']

    # We need to complete the forecasting days, as not all days had visitors and only those were the ones in our datasets.
    sample_submission = pd.read_csv('../data/raw/sample_submission.csv')
    restaurants = sample_submission.id.str[:20].unique()

    start_date = datetime.datetime.strptime('2017-04-23', "%Y-%m-%d")
    end_date = datetime.datetime.strptime('2017-05-31', "%Y-%m-%d")

    #We'll use a new dataframe to store the prediction data and the new dates
    predict_df = pd.DataFrame(columns=data_test.columns)

    for restaurant in restaurants:
        while start_date <= end_date:

            if len(data_test[(data_test['air_store_id']==restaurant) & (data_test['visit_month']==start_date.month)  & (data_test['visit_day']==start_date.day)]):
                predict_df = predict_df.append(data_test[(data_test['air_store_id']==restaurant)  & (data_test['visit_month']==start_date.month)                                                      & (data_test['visit_day']==start_date.day)], ignore_index=True)
            else:
                position = len(predict_df)
                predict_df.loc[position,"air_store_id"] = restaurant
                predict_df.loc[position,"visit_year"] = start_date.year
                predict_df.loc[position,"visit_month"] = start_date.month
                predict_df.loc[position,"visit_day"] = start_date.day
                predict_df.loc[position,"dow"] = calendar.day_name[start_date.weekday()]

            start_date += datetime.timedelta(days=1)
        start_date = datetime.datetime.strptime('2017-04-23', "%Y-%m-%d")

    # We now have a complete test dataset, for all ids and dates to be forecasted.
    # Lets format now the dataframe in order to be able to use it in the previously obtained models.
    date_info = pd.read_csv('../data/raw/date_info.csv',parse_dates=['calendar_date'])
    date_info['calendar_year'] = date_info['calendar_date'].dt.year
    date_info['calendar_month'] = date_info['calendar_date'].dt.month
    date_info['calendar_day'] = date_info['calendar_date'].dt.day
    date_info.drop(['calendar_date'], axis=1, inplace=True)
    predict_df = pd.merge(predict_df, date_info, left_on=['visit_year','visit_month','visit_day'], right_on=['calendar_year','calendar_month','calendar_day'], how='left')
    predict_df = pd.get_dummies(predict_df, columns=['dow'])
    predict_df = pd.merge(predict_df, data_train[['air_store_id','visitors_rest_mean']].drop_duplicates(),  on='air_store_id', how='left')

    predict_df.drop(['holiday_flg_x','day_of_week','calendar_year','calendar_month','calendar_day','latitude', 'longitude','air_area_name','genre'], axis=1, inplace=True)
    predict_df=predict_df.rename(columns = {'holiday_flg_y':'holiday_flg'})
    predict_df.sort_values(by=['reserve_visitors','visitors_rest_mean'], ascending=[True,True], inplace=True)

    predict_df.reserve_visitors = pd.to_numeric(predict_df.reserve_visitors)
    predict_df.visit_year = pd.to_numeric(predict_df.visit_year)
    predict_df.visit_month = pd.to_numeric(predict_df.visit_month)
    predict_df.visit_day = pd.to_numeric(predict_df.visit_day)

    # Let's check, how many models for restaurants from the train data do we have, and how many restaurants in total are there in the predict dataframe?
    print("Different restaurants in test set:",len(predict_df.air_store_id.unique()))
    print("Different restaurants in train set:",len(data_train.air_store_id.unique()))

    # Finally, let's make the prediction for each restaurant and date.
    predict_df.reset_index(drop=True, inplace=True)
    predict_df = predict_df[['dow_Friday','dow_Monday','dow_Saturday','dow_Sunday','dow_Thursday','dow_Tuesday','dow_Wednesday','holiday_flg','reserve_visitors','visit_day','visit_month','visit_year','visitors_rest_mean','air_store_id','visitors']]

    for index, row in predict_df.iterrows():
        if row['air_store_id'] in models_dict and row['reserve_visitors'] > 0:
            visitors = models_dict[row['air_store_id']].predict(predict_df.loc[index:index,'dow_Friday':'visitors_rest_mean'])
            predict_df.visitors[index] = visitors
        elif row['air_store_id'] in half_models_dict and row['visitors_rest_mean'] > 0:
            visitors = half_models_dict[row['air_store_id']].predict(predict_df.loc[index:index,'dow_Friday':'visitors_rest_mean'])
            predict_df.visitors[index] = visitors
        else:
            visitors = nodata_model.predict(predict_df.loc[index:index,'dow_Friday':'visitors_rest_mean'])
            predict_df.visitors[index] = visitors

        if index%5000 == 0 or index==(len(predict_df)-1):
            print("Prediction {} of {}".format(index+1,len(predict_df)))

    # Let's see what kind of predictions did we get.
    print(predict_df.visitors.describe())

    # There are a few negative numbers and a few to big to be true. Let's crop that numbers.
    predict_df.visitors[predict_df['visitors']<data_train.visitors.min()] = data_train.visitors.mean()
    predict_df.visitors[predict_df['visitors']>data_train.visitors.max()] = data_train.visitors.mean()

    # Finally, let's create the file with the appropriate format to be submitted to the Kaggle challenge.
    predict_df['id'] = predict_df.apply(submission_format, axis=1)
    predict_df['visitors'] = predict_df['visitors'].round(0).astype(int)

    return predict_df
