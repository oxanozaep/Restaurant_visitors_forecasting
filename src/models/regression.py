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

#Definition of the formula that will show the goodness of the model.
def RMSLE(predicted, actual):
    msle = (np.log(predicted+1) - np.log(actual+1))**2
    rmsle = np.sqrt(msle.sum()/msle.count())
    return rmsle

def save_model(obj, name):
    with open('../models/'+ name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
        
def forward(predictors, X, y):
    remaining_predictors = [p for p in X.columns if p not in predictors]
    results = []

    for p in remaining_predictors:
        results.append(processSubset(predictors + [p], X, y))

    models = pd.DataFrame(results)
    print("Processed ", models.shape[0], "models on", len(predictors)+1, "predictors.")
    return models.loc[models['RSS'].argmin()]

def processSubset(feature_set, X, y):
    model = sm.OLS(y, X[list(feature_set)])
    regr = model.fit()
    RSS = ((regr.predict(X[list(feature_set)]) - y) ** 2).sum()
    return {"model":regr, "RSS":RSS}

def regression(data_train):
    data_train = pd.get_dummies(data_train, columns=['genre','dow'])

    #We will use the log of the visitors to get a more useful mean.
    model_mean_pred = data_train.log_visitors.mean()

    # And we'll store this value in the dataframe
    data_train['visitors_mean'] = np.exp(model_mean_pred)

    data_train.loc[:, ['visitors','visitors_mean']].plot(color=['#bbbbbb','r'], figsize=(16,8));

    model_mean_RMSLE = RMSLE(data_train.visitors_mean, data_train.visitors)

    results_df = pd.DataFrame(columns=["Model", "RMSLE"])

    results_df.loc[0,"Model"] = "Mean"
    results_df.loc[0,"RMSLE"] = model_mean_RMSLE

    # Let's now see if and how much the model would enhance if we predicted always the mean number of visitors of the restaurant being predicted.
    data_train = pd.merge(data_train, data_train[['air_store_id','visitors']].groupby(['air_store_id'], as_index=False).mean(), on='air_store_id', how='left')
    data_train=data_train.rename(columns = {'visitors_y':'visitors_rest_mean','visitors_x':'visitors'})
    model_mean_rest_RMSLE = RMSLE(data_train.visitors_rest_mean, data_train.visitors)

    results_df.loc[1,"Model"] = "Mean_by_rest"
    results_df.loc[1,"RMSLE"] = model_mean_rest_RMSLE

    # Let's start creating the models with linear and polynomial regression. Starting with a model with multiple linear regressors, one for each variable in the data.
    model = sm.OLS.from_formula('visitors ~ ' + '+'.join(data_train.columns.difference(['visitors', 'log_visitors', 'air_store_id','visitors_mean'])), data_train)
    result = model.fit()
    print(result.summary())

    data_train["linear_regr"] = result.predict()
    model_lin_RMSLE = RMSLE(data_train.linear_regr, data_train.visitors)
    results_df.loc[2,"Model"] = "Multiple linear regressors"
    results_df.loc[2,"RMSLE"] = model_lin_RMSLE

    # We'll try and perform now some sort of random walk model: We'll just take the visitors of the restaurant from the previous similar day of the week, as this could be a good fit that includes seasonality for each restaurant. For that, we'll create 7 new columns containing the value of previous similar dow visitors and then create a new column, "past_dow_visitors", with the appropriate number for the specific day.
    dows = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
    for dow in dows:
        data_train['past_'+dow]= 0

    data_train.sort_values(by=['air_store_id','visit_year','visit_month','visit_day'], ascending=[True,True,True,True], inplace=True)
    data_train['store_change'] = (data_train.air_store_id!=data_train.air_store_id.shift())
    data_train['past_dow_visitors'] = data_train['visitors_rest_mean']
    data_train.reset_index(drop=True, inplace=True)

    for index, row in data_train.iterrows():
        if not row.store_change:
            for dow in dows:
                if data_train.iloc[index-1, data_train.columns.get_loc('dow_'+dow)]:
                    data_train.set_value(index,'past_'+dow,data_train.iloc[index-1, data_train.columns.get_loc('visitors')])
                else:
                    data_train.set_value(index,'past_'+dow,data_train.iloc[index-1, data_train.columns.get_loc('past_'+dow)])

    for index, row in data_train.iterrows():
        for dow in dows:
            if row['dow_'+dow] and row['past_'+dow]>0:
                data_train.set_value(index,'past_dow_visitors', row['past_'+dow])

    for dow in dows:
        data_train.drop(['past_'+dow], axis=1, inplace=True)

    # The "random walk" model will include this new variable and the two other most powerful ones, the reserve visitors and wether if it's a holiday or not. We'll also include the intercept between the variables this time.
    model = sm.OLS.from_formula('visitors ~ past_dow_visitors * reserve_visitors * holiday_flg',data_train)
    result = model.fit()
    print(result.summary())

    # This time, all the variables have strong predictive power, being the newly created column of past day of week visitors the one with a higher t statistic (>100)
    model_pred = result.predict()
    data_train['past_dow_predict'] = model_pred
    model_past_dow_RMSLE = RMSLE(data_train.past_dow_predict, data_train.visitors)
    results_df.loc[3,"Model"] = "Past_DoW"
    results_df.loc[3,"RMSLE"] = model_past_dow_RMSLE

    # Residuals:
    s_residuals = pd.Series(result.resid_pearson, name="S. Residuals")
    fitted_values = pd.Series(result.fittedvalues, name="Fitted Values")
    sns.regplot(fitted_values, s_residuals,  fit_reg=False)

    models = pd.DataFrame(columns=["RSS", "model"])
    predictors = []
    y=data_train.visitors
    X = data_train[['visit_year', 'visit_month', 'visit_day', 'reserve_visitors','holiday_flg','latitude','longitude', 'dow_Friday','dow_Monday','dow_Tuesday','dow_Wednesday','dow_Thursday','dow_Saturday','dow_Sunday', 'visitors_rest_mean','past_dow_visitors']].astype('float64')

    for i in range(1, len(X.columns) + 1):
        models.loc[i] = forward(predictors, X, y)
        predictors = models.loc[i]["model"].model.exog_names

    # Let's inspect the correlation coefficient for each of the best possible models with the different number of predictors.
    models.apply(lambda row: row[1].rsquared, axis=1)

    # Let's show some graphs to see how these models compare to each other.
    plt.figure(figsize=(20,10))
    plt.rcParams.update({'font.size': 18, 'lines.markersize': 10})
    plt.subplot(4, 1, 1)

    plt.plot(models["RSS"])
    plt.xlabel('# Predictors')
    plt.ylabel('RSS')

    rsquared_adj = models.apply(lambda row: row[1].rsquared_adj, axis=1)

    plt.subplot(4, 1, 2)
    plt.plot(rsquared_adj)
    plt.plot(rsquared_adj.argmax(), rsquared_adj.max(), "ob")
    plt.xlabel('# Predictors')
    plt.ylabel('adjusted rsquared')

    aic = models.apply(lambda row: row[1].aic, axis=1)

    plt.subplot(4, 1, 3)
    plt.plot(aic)
    plt.plot(aic.argmin(), aic.min(), "ob")
    plt.xlabel('# Predictors')
    plt.ylabel('AIC')

    bic = models.apply(lambda row: row[1].bic, axis=1)

    plt.subplot(4, 1, 4)
    plt.plot(bic)
    plt.plot(bic.argmin(), bic.min(), "ob")
    plt.xlabel('# Predictors')
    plt.ylabel('BIC')

    # We'll chose the model with 8 predictors to try to keep it simple, as after this point, the models only preform slightly better.
    data_train["subset_selection"] = models.loc[8, "model"].predict()
    model_subset_RMSLE = RMSLE(data_train.subset_selection, data_train.visitors)

    results_df.loc[4,"Model"] = "Subset selection"
    results_df.loc[4,"RMSLE"] = model_subset_RMSLE

    # Let's try a polynomial regression model with the past dow visitors variable, as it is the one with the highest t statistic, up to a 5th degree polynomial.
    poly_1 = smf.ols(formula='visitors ~ 1 + past_dow_visitors', data=data_train).fit()
    poly_2 = smf.ols(formula='visitors ~ 1 + past_dow_visitors + I(past_dow_visitors ** 2.0)', data=data_train).fit()
    poly_3 = smf.ols(formula='visitors ~ 1 + past_dow_visitors + I(past_dow_visitors ** 2.0) + I(past_dow_visitors ** 3.0)', data=data_train).fit()
    poly_4 = smf.ols(formula='visitors ~ 1 + past_dow_visitors + I(past_dow_visitors ** 2.0) + I(past_dow_visitors ** 3.0) + I(past_dow_visitors ** 4.0)', data=data_train).fit()
    poly_5 = smf.ols(formula='visitors ~ 1 + past_dow_visitors + I(past_dow_visitors ** 2.0) + I(past_dow_visitors ** 3.0) + I(past_dow_visitors ** 4.0) + I(past_dow_visitors ** 5.0)', data=data_train).fit()

    print(sm.stats.anova_lm(poly_1, poly_2, poly_3, poly_4, poly_5, typ=1))

    plt.figure(figsize=(6 * 1.618, 6))
    plt.scatter(data_train.past_dow_visitors, data_train.visitors, s=10, alpha=0.3)
    plt.xlabel('past_dow_visitors')
    plt.ylabel('visitors')

    x = pd.DataFrame({'past_dow_visitors': np.linspace(data_train.past_dow_visitors.min(), data_train.past_dow_visitors.max(), 100)})
    plt.plot(x.past_dow_visitors, poly_1.predict(x), 'b-', label='Poly n=1 $R^2$=%.2f' % poly_1.rsquared, alpha=0.9)
    plt.plot(x.past_dow_visitors, poly_2.predict(x), 'g-', label='Poly n=2 $R^2$=%.2f' % poly_2.rsquared, alpha=0.9)
    plt.plot(x.past_dow_visitors, poly_3.predict(x), 'r-', alpha=0.9,label='Poly n=3 $R^2$=%.2f' % poly_3.rsquared)
    plt.plot(x.past_dow_visitors, poly_4.predict(x), 'y-', alpha=0.9,label='Poly n=4 $R^2$=%.2f' % poly_4.rsquared)
    plt.plot(x.past_dow_visitors, poly_5.predict(x), 'k-', alpha=0.9,label='Poly n=5 $R^2$=%.2f' % poly_5.rsquared)

    plt.legend()

    data_train["poly_regr"] = poly_5.predict()
    model_poly_RMSLE = RMSLE(data_train.poly_regr, data_train.visitors)

    results_df.loc[5,"Model"] = "Polynomial Regressor"
    results_df.loc[5,"RMSLE"] = model_poly_RMSLE
    results_df

    return data_train,results_df

def multiple_model_creation(data_train,results_df):
    # We'll start by creating a multiple linear regression model for each restaurant in the train data.
    #Let's get rid of the columns that won't be used in the final predictions.
    data_train.drop(data_train[['air_area_name', 'latitude','past_dow_visitors','longitude','visitors_mean','linear_regr','store_change','past_dow_predict','subset_selection','poly_regr','log_visitors']], axis=1, inplace=True)
    data_train.drop(list(data_train.filter(regex = 'genre_')), axis = 1, inplace = True)

    restaurants = data_train.air_store_id.unique()
    RMSLEs = []
    models_dict = {}

    for i,restaurant in enumerate(restaurants):
        if i%100 == 0 or i==(len(restaurants)-1):
            print("Model {} of {}".format(i+1,len(restaurants)))

        df_temp = data_train[data_train.air_store_id == restaurant]
        df_temp.dropna(axis=0,how='any',inplace=True)
        model = sm.OLS.from_formula('visitors ~ ' + '+'.join(df_temp.columns.difference(['visitors', 'air_store_id'])), df_temp).fit()
        RMSLEs.append(RMSLE(model.predict(), df_temp.visitors))
        models_dict[restaurant] = model

    # We'll create now the models for the restaurants with no reserved visitors info, as this data is not complete for the forecasted weeks.
    RMSLEhalf = []
    half_models_dict = {}

    for i,restaurant in enumerate(restaurants):
        if i%100 == 0 or i==(len(restaurants)-1):
            print("Model {} of {}".format(i+1,len(restaurants)))

        df_temp = data_train[data_train.air_store_id == restaurant]
        df_temp.dropna(axis=0,how='any',inplace=True)
        model = sm.OLS.from_formula('visitors ~ ' + '+'.join(df_temp.columns.difference(['visitors', 'air_store_id','reserve_visitors'])), df_temp).fit()
        RMSLEhalf.append(RMSLE(model.predict(), df_temp.visitors))
        half_models_dict[restaurant] = model

    # And finally, a last model for those restaurants that are new in the test dataframe.
    nodata_model = sm.OLS.from_formula('visitors ~ ' + '+'.join(data_train.columns.difference(['visitors', 'air_store_id','reserve_visitors','visitors_rest_mean'])), data_train).fit()
    RMSLE_rest = RMSLE(nodata_model.predict(), data_train.visitors)

    # Let's see how these newly created models compare with the ones obtained in the modeling section.
    results_df.loc[6,"Model"] = "Regressor per id"
    results_df.loc[6,"RMSLE"] = np.mean(RMSLEs)
    results_df.loc[7,"Model"] = "Regressor per id w/o reserves"
    results_df.loc[7,"RMSLE"] = np.mean(RMSLEs)
    results_df.loc[8,"Model"] = "New id model"
    results_df.loc[8,"RMSLE"] = RMSLE_rest

    results_df

    # We'll store all the created models
    save_model(models_dict,'full_models')
    save_model(half_models_dict,'half_models')
    save_model(nodata_model,'no_data_model')

    return data_train, models_dict, half_models_dict, nodata_model
