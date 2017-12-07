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

from models.regression import *

def time_series(data_train):
    # We will use a single restaurant id (air_6b15edd1b4fbb96a) to evaluate it's time evolution data and use that to forecast the visitors.
    # Let's first explore the chosen id creating a time index.

    df_time = data_train[data_train.air_store_id == 'air_6b15edd1b4fbb96a']
    df_time.set_index(pd.to_datetime(df_time.visit_year*10000+df_time.visit_month*100+df_time.visit_day,format='%Y%m%d'), inplace=True)
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(16, 7))

    axes[0].plot(df_time.visitors, color='navy', linewidth=4)
    axes[1].plot(df_time.visitors[df_time.visit_month > 10], color='navy', linewidth=4)

    # Let's try now to forecast this time series using several methods.
    # First, let's see how the first model from the previous section would behave, the ones with the total average visitors and the specific restaurant average visitors.
    model_mean_RMSLE = RMSLE(df_time.visitors_mean, df_time.visitors)
    model_rest_mean_RMSLE = RMSLE(df_time.visitors.mean(), df_time.visitors)

    results_df_time = pd.DataFrame(columns=["Model", "RMSLE"])
    results_df_time.loc[0,"Model"] = "Total Mean"
    results_df_time.loc[0,"RMSLE"] = model_mean_RMSLE
    results_df_time.loc[1,"Model"] = "Restaurant Mean"
    results_df_time.loc[1,"RMSLE"] = model_rest_mean_RMSLE

    # We will decompose the time series into trend and seasonality
    decomposition = seasonal_decompose(df_time.log_visitors, model="additive", freq=6)
    decomposition.plot();

    # Let's store this information into the dataframe and predict the visitors using them.
    trend = decomposition.trend
    seasonal = decomposition.seasonal
    residual = decomposition.resid

    df_time['power_decomp'] = np.exp(trend + seasonal)

    model_Decomp_RMSLE = RMSLE(df_time.power_decomp, df_time.visitors)

    results_df_time.loc[2,"Model"] = "Time Decomposition"
    results_df_time.loc[2,"RMSLE"] = model_Decomp_RMSLE

    # Let's start by forward subsetting the predictors.
    models_time = pd.DataFrame(columns=["RSS", "model"])

    predictors = []
    y=df_time.visitors
    X = df_time[['visit_year', 'visit_month', 'visit_day', 'reserve_visitors','holiday_flg','latitude','longitude', 'dow_Friday','dow_Monday','dow_Tuesday','dow_Wednesday','dow_Thursday','dow_Saturday','dow_Sunday', 'visitors_rest_mean','past_dow_visitors']].astype('float64')

    for i in range(1, len(X.columns) + 1):
        models_time.loc[i] = forward(predictors, X, y)
        predictors = models_time.loc[i]["model"].model.exog_names

    # Again, let's plot them to choose an appropriate number of predictors.
    plt.figure(figsize=(20,10))
    plt.rcParams.update({'font.size': 18, 'lines.markersize': 10})
    plt.subplot(4, 1, 1)

    plt.plot(models_time["RSS"])
    plt.xlabel('# Predictors')
    plt.ylabel('RSS')

    rsquared_adj = models_time.apply(lambda row: row[1].rsquared_adj, axis=1)

    plt.subplot(4, 1, 2)
    plt.plot(rsquared_adj)
    plt.plot(rsquared_adj.argmax(), rsquared_adj.max(), "ob")
    plt.xlabel('# Predictors')
    plt.ylabel('adjusted rsquared')

    aic = models_time.apply(lambda row: row[1].aic, axis=1)

    plt.subplot(4, 1, 3)
    plt.plot(aic)
    plt.plot(aic.argmin(), aic.min(), "ob")
    plt.xlabel('# Predictors')
    plt.ylabel('AIC')

    bic = models_time.apply(lambda row: row[1].bic, axis=1)

    plt.subplot(4, 1, 4)
    plt.plot(bic)
    plt.plot(bic.argmin(), bic.min(), "ob")
    plt.xlabel('# Predictors')
    plt.ylabel('BIC')

    df_time["subset_selection"] = models_time.loc[10, "model"].predict()
    model_subset_RMSLE = RMSLE(df_time.subset_selection, df_time.visitors)

    results_df_time.loc[3,"Model"] = "Subset selection"
    results_df_time.loc[3,"RMSLE"] = model_subset_RMSLE

    # We get rid of the genres, as they do not help making a better model
    df_time.drop(list(df_time.filter(regex = 'genre_')), axis = 1, inplace = True)
    df_time.dropna(axis=0,how='any',inplace=True)

    model = sm.OLS.from_formula('visitors ~ ' + '+'.join(df_time.columns.difference(['visitors', 'log_visitors','air_store_id','visitors_mean', 'subset_selection','past_dow_predict','power_decomp','poly_regr'])), df_time)

    result = model.fit()
    print(result.summary())

    df_time["linear_regr"] = result.predict()

    # RMSLE for linear regressor
    model_lin_RMSLE = RMSLE(df_time.linear_regr, df_time.visitors)
    results_df_time.loc[4,"Model"] = "Linear Regressor"
    results_df_time.loc[4,"RMSLE"] = model_lin_RMSLE

    return results_df_time
