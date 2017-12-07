
# coding: utf-8

# # Restaurant visitors forecasting
# 
# ## Data
# 
# All the datasets are provided by the Kaggle competition "Recruit Restaurant Visitor Forecasting" (https://www.kaggle.com/c/recruit-restaurant-visitor-forecasting).These are:
# 
# - air_reserve.csv.csv
# - air_store_info.csv
# - air_visit_data.csv
# - date_info.csv
# - hpg_reserve.csv
# - hpg_store_info.csv
# - store_id_relation.csv
# 
# There's a final dataset, included for the purpose of defining a common format for prediction submission:
# 
# - sample_submission.csv
# 
# 
# ## Goal
# 
# We will try to perform an EDA of the provided data by loading, cleaning and merging the files, and finally visualizing the final dataframes. We'll also take advantage of the time series data provided to visualize an example, and finally we'll use different regression techniques to see how could we predict the visitors of a restaurant using only the regression information.
# 
# 
# ## Index
# 
# 1- Importing libraries and loading the datasets
# 
# 2- Data preparation: Exploring, cleaning and merging datasets
# 
# 3- Visualizing the data
# 
# 4- Modeling the data
# 
# 5- Time series forecasting
# 
# 6- Generation of predictions and submission file

# ### 1- Importing libraries and loading the datasets
# 
# - Importing the necessary libraries

# In[1]:


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

# - Loading the data files

# In[2]:


#We willl load all the csv files into Pandas dataframes, properly parsing dates

air_reserve = pd.read_csv('../data/raw/air_reserve.csv',parse_dates=['visit_datetime','reserve_datetime'])
hpg_reserve = pd.read_csv('../data/raw/hpg_reserve.csv',parse_dates=['visit_datetime','reserve_datetime'])
air_store_info = pd.read_csv('../data/raw/air_store_info.csv')
hpg_store_info = pd.read_csv('../data/raw/hpg_store_info.csv')
store_relation = pd.read_csv('../data/raw/store_id_relation.csv')
date_info = pd.read_csv('../data/raw/date_info.csv',parse_dates=['calendar_date'])
air_visit = pd.read_csv('../data/raw/air_visit_data.csv',parse_dates=['visit_date'])
sample_submission = pd.read_csv('../data/raw/sample_submission.csv')


# ### Data preparation: Exploring, cleaning and merging datasets
# 
# - Exploring the data
# 
# We will use here Pandas Profiling, as it is a nice way to understand the content of each file and all their important characteristics. It's going to be a little painful, since there are 7 files to profile, but it will definitely worth it.

# In[3]:


pandas_profiling.ProfileReport(air_reserve)


# In[4]:


pandas_profiling.ProfileReport(hpg_reserve)


# In[5]:


pandas_profiling.ProfileReport(air_store_info)


# In[6]:


pandas_profiling.ProfileReport(hpg_store_info)


# In[7]:


pandas_profiling.ProfileReport(store_relation)


# In[8]:


pandas_profiling.ProfileReport(date_info)


# In[9]:


pandas_profiling.ProfileReport(air_visit)


# - Cleaning the data
# 
# As we can see from the Pandas profiles above, the data is so clean that fortunately it will take low to no work cleaning it up. None of the files have missing data and only two files have duplicates (air_reserve & hpg_reserve), which are considered to be valid duplicates since there can be multiple reserves records for the same restaurant, day and hour, made in the same hour range. We'll just need to downsample those to daily reserves to get rid of the duplicates.
# 
# One thing we can do is check is if there are reserves made after the visit time, which shouldn't be possible.

# In[10]:


print((air_reserve['reserve_datetime']>air_reserve['visit_datetime']).value_counts())

print((hpg_reserve['reserve_datetime']>hpg_reserve['visit_datetime']).value_counts())


# We can see that there are no trues in the above statement, so no errors were found.
# 
# Now, it would be usefull to translate all the hpg ids to air ids, to have a common ground before merging dataframes. To get that, we need to downsample the reserves in the hpg system in order to have a single row when we do the merge with the air system.

# In[11]:


hpg_reserve['visit_year'] = hpg_reserve['visit_datetime'].dt.year
hpg_reserve['visit_month'] = hpg_reserve['visit_datetime'].dt.month
hpg_reserve['visit_day'] = hpg_reserve['visit_datetime'].dt.day
hpg_reserve['reserve_year'] = hpg_reserve['reserve_datetime'].dt.year
hpg_reserve['reserve_month'] = hpg_reserve['reserve_datetime'].dt.month
hpg_reserve['reserve_day'] = hpg_reserve['reserve_datetime'].dt.day

hpg_reserve.drop(['visit_datetime','reserve_datetime'], axis=1, inplace=True)

hpg_reserve = hpg_reserve.groupby(['hpg_store_id', 'visit_year', 'visit_month',                                   'visit_day','reserve_year','reserve_month','reserve_day'], as_index=False).sum()


# We should also prepair the rest of the files to get merged by visit day.

# In[12]:


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


# - Merging the data
# 
# Now that the data is prepared to be merged, we need to add all the columns to the air_reserve file, as it is the file connected to all the rest of them by one or other way.
# 
# First, we merge all the reserves from both systems into the air_reserve file.

# In[13]:


hpg_reserve = pd.merge(hpg_reserve, store_relation, on='hpg_store_id', how='inner')
hpg_reserve.drop(['hpg_store_id'], axis=1, inplace=True)

air_reserve = pd.concat([air_reserve, hpg_reserve])


# Now we can downsalmple to daily visit days, adding all the reserves made for that specific date.

# In[14]:


air_reserve = air_reserve.groupby(['air_store_id', 'visit_year', 'visit_month','visit_day'],                as_index=False).sum().drop(['reserve_day','reserve_month','reserve_year'], axis=1)


# We can easily add the holiday info to our dataframe.

# In[15]:


air_reserve = pd.merge(air_reserve, date_info, left_on=['visit_year','visit_month','visit_day'], right_on=['calendar_year','calendar_month','calendar_day'], how='left')
air_reserve.drop(['calendar_year','calendar_month','calendar_day'], axis=1, inplace=True)


# And merge also the store information and the restaurant visits per day. At this point, we're going to create a new dataframe, df, to mark the moment where we have all the data together.

# In[16]:


air_reserve = pd.merge(air_reserve, air_store_info, on='air_store_id', how='left')

df = pd.merge(air_reserve, air_visit, on=['air_store_id','visit_year','visit_month','visit_day'], how='left')


# Let's see what is the result with this complete dataframe with a Pandas profile.

# In[17]:


pandas_profiling.ProfileReport(df)


# There missings values found in the visits variable will make our test dataset, as that is the variable that we want to predict. Hence, the rest of rows will be our train data. Let's create both train and test dataframes.

# In[18]:


df.air_genre_name = df.air_genre_name.replace(' ', '_', regex=True)
df.air_genre_name = df.air_genre_name.replace('/', '_', regex=True)
df=df.rename(columns = {'air_genre_name':'genre','day_of_week':'dow'})

df.sort_values(by=['visit_year','visit_month','visit_day','air_store_id'],               ascending=[True,True,True,True], inplace=True)

data_train = df[df.visitors.notnull()]
data_test = df[df.visitors.isnull()]


# We will create a new column with the natural logarithm of the visitor numbers, in case that helps us in the forecasting section later.

# In[19]:


data_train['log_visitors'] = data_train.visitors.apply(lambda x: np.log(x))


# We're going to store these two dataframes into a database for easy use in future projects.

# In[20]:


conn = sqlite3.connect("../data/processed/restaurants.db")
data_train.to_sql("data_train", conn, if_exists="replace")
data_test.to_sql("data_test", conn, if_exists="replace")


# ### 3- Visualizing the data
# 
# - We can start by visualizing violin plots of the visitors distribution for each day of the week, differentiating if it is a holiday or not.

# In[21]:


fig, ax = plt.subplots(figsize=(14,12));
ax = sns.violinplot(x='dow', y="visitors", hue='holiday_flg',data=df, palette="muted", split=True)


# We can see that, as expected, Monday through Thursday, the distribution of visitors is much lower than Friday to Sunday. Also, the holiday flag plays a big role in the visitor number, but that role seems to have a bigger effect on weekdays.
# 
# - We should explore now the relationship between the reserve visitors and the actual visitors.

# In[22]:


sns.jointplot(x='visitors', y='reserve_visitors', data=data_train, color='navy',              size=10, space=0, kind='reg',marginal_kws={'hist_kws': {'log': True}})


# There seems to be a strong relationship, with a p of 0, so we can reject the null hypothesis of both variables being independent, and a Pearson correlation coefficient of 0.42.
# 
# - We should also inspect the visitors affluence to the restaurants depending on the month of the year.

# In[23]:


data_train_month = data_train[['visit_month','visitors','visit_year']].groupby(['visit_year','visit_month']).sum()

data_train_month.plot(kind ="bar", y='visitors')


# In the previous graph we can see every month in the dataset, as there are not so many months. We can see how this kind of graph could be skewed, as the visitors data has a big jump in November 2016 and it stays high until the end of the graph. This could be that there are more restaurants added to the database, or multiple other reasons. Let's check the month average.

# In[24]:


data_train_month_av = data_train[['visit_month','visitors','visit_year']].groupby(['visit_month']).mean()

data_train_month_av.plot(kind ="bar", y='visitors')


# As we can see, the average of restaurant visitors for each month doesn't show those big jumps.
# 
# - We've seen the strong relationship between reserve visitors and visitors. Is there any other strong correlation in the dataset?

# In[25]:


cor = data_train.corr()
plt.figure(figsize=(14,3))
sns.heatmap(cor.loc[['visitors'], list(df)[:-1]]);


# We can see that the reserve visitors variable is the strongest with a great difference. After it, we could use holiday flag, visit day and visit month. We'll check on those later.
# 
# - Finally, what is the evolution of visitors per day of the week for each month?

# In[26]:


data_train_Pivot = pd.pivot_table(data_train, values='visitors', columns='dow', index='visit_month')
data_train_Pivot.plot();
plt.legend(bbox_to_anchor=(1,1), loc="upper left")


# ### 4- Modeling the data
# 
# Now that we have a clean dataframe and that we've inspected the variables and their relationship, let's start trying out some models to find out their behaviour.
# 
# - First, we'll start by predicting always the average number of visitors of all restaurants. Any future model behaving worse than this one will be useless.

# In[27]:


#Definition of the formula that will show the goodness of the model.

def RMSLE(predicted, actual):
    msle = (np.log(predicted+1) - np.log(actual+1))**2
    rmsle = np.sqrt(msle.sum()/msle.count())
    return rmsle


# In[28]:


data_train = pd.get_dummies(data_train, columns=['genre','dow'])

#We will use the log of the visitors to get a more useful mean.
model_mean_pred = data_train.log_visitors.mean()

# And we'll store this value in the dataframe
data_train['visitors_mean'] = np.exp(model_mean_pred)

data_train.loc[:, ['visitors','visitors_mean']].plot(color=['#bbbbbb','r'], figsize=(16,8));


# In[29]:


model_mean_RMSLE = RMSLE(data_train.visitors_mean, data_train.visitors)

results_df = pd.DataFrame(columns=["Model", "RMSLE"])

results_df.loc[0,"Model"] = "Mean"
results_df.loc[0,"RMSLE"] = model_mean_RMSLE
results_df.head()


# - Let's now see if and how much the model would enhance if we predicted always the mean number of visitors of the restaurant being predicted.

# In[30]:


data_train = pd.merge(data_train, data_train[['air_store_id','visitors']].groupby(['air_store_id'], as_index=False).mean(), on='air_store_id', how='left')

data_train=data_train.rename(columns = {'visitors_y':'visitors_rest_mean','visitors_x':'visitors'})

model_mean_rest_RMSLE = RMSLE(data_train.visitors_rest_mean, data_train.visitors)

results_df.loc[1,"Model"] = "Mean_by_rest"
results_df.loc[1,"RMSLE"] = model_mean_rest_RMSLE
results_df.head()


# - Let's start creating the models with linear and polynomial regression. Starting with a model with multiple linear regressors, one for each variable in the data.

# In[31]:


model = sm.OLS.from_formula('visitors ~ ' + '+'.join(data_train.columns.difference(['visitors',                             'log_visitors', 'air_store_id','visitors_mean'])), data_train)
result = model.fit()
print(result.summary())


# We can see how the null hypothesis of independence can't be rejected for none of the dummy variables (genres, areas and day of week), as can't be for latitude and longitude. However, the holiday flag, the reserve visitors and the visit date, as well as the own mean visitors number for the restaurant, help to get a better prediction.

# In[32]:


data_train["linear_regr"] = result.predict()

model_lin_RMSLE = RMSLE(data_train.linear_regr, data_train.visitors)

results_df.loc[2,"Model"] = "Multiple linear regressors"
results_df.loc[2,"RMSLE"] = model_lin_RMSLE
results_df


# - We'll try and perform now some sort of random walk model: We'll just take the visitors of the restaurant from the previous similar day of the week, as this could be a good fit that includes seasonality for each restaurant. For that, we'll create 7 new columns containing the value of previous similar dow visitors and then create a new column, "past_dow_visitors", with the appropriate number for the specific day.

# In[33]:


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


# In[34]:


for index, row in data_train.iterrows():
    for dow in dows:
        if row['dow_'+dow] and row['past_'+dow]>0:
            data_train.set_value(index,'past_dow_visitors', row['past_'+dow])

for dow in dows:
    data_train.drop(['past_'+dow], axis=1, inplace=True)


# The "random walk" model will include this new variable and the two other most powerful ones, the reserve visitors and wether if it's a holiday or not. We'll also include the intercept between the variables this time.

# In[35]:


model = sm.OLS.from_formula('visitors ~ past_dow_visitors * reserve_visitors * holiday_flg',data_train)
result = model.fit()
print(result.summary())


# This time, all the variables have strong predictive power, being the newly created column of past day of week visitors the one with a higher t statistic (>100)

# In[36]:


model_pred = result.predict()
data_train['past_dow_predict'] = model_pred

model_past_dow_RMSLE = RMSLE(data_train.past_dow_predict, data_train.visitors)

results_df.loc[3,"Model"] = "Past_DoW"
results_df.loc[3,"RMSLE"] = model_past_dow_RMSLE
results_df


# Nevertheless, this model does not outperform the multiple linear regressors obtained previously.
# 
# Residuals:

# In[37]:


s_residuals = pd.Series(result.resid_pearson, name="S. Residuals")
fitted_values = pd.Series(result.fittedvalues, name="Fitted Values")
sns.regplot(fitted_values, s_residuals,  fit_reg=False)


# The residuals seem to be in a random distribution, and we can't observe a curvature in the data nor heteroskedasticity.
# 
# - Let's create a more efficient model by using forward subsetting, using all the variables in the dataframe, including the newly created past dow visitors. Let's start by defining the needed functions.

# In[38]:


def forward(predictors):
    remaining_predictors = [p for p in X.columns if p not in predictors]    
    results = []
    
    for p in remaining_predictors:
        results.append(processSubset(predictors + [p]))
    
    models = pd.DataFrame(results)
    print("Processed ", models.shape[0], "models on", len(predictors)+1, "predictors.")
    return models.loc[models['RSS'].argmin()]

def processSubset(feature_set):
    model = sm.OLS(y, X[list(feature_set)])
    regr = model.fit()
    RSS = ((regr.predict(X[list(feature_set)]) - y) ** 2).sum()
    return {"model":regr, "RSS":RSS}


# In[39]:


models = pd.DataFrame(columns=["RSS", "model"])

predictors = []
y=data_train.visitors
X = data_train[['visit_year', 'visit_month', 'visit_day', 'reserve_visitors','holiday_flg','latitude','longitude',                'dow_Friday','dow_Monday','dow_Tuesday','dow_Wednesday','dow_Thursday','dow_Saturday','dow_Sunday',                 'visitors_rest_mean','past_dow_visitors']].astype('float64')

for i in range(1, len(X.columns) + 1):    
    models.loc[i] = forward(predictors)
    predictors = models.loc[i]["model"].model.exog_names


# Let's inspect the correlation coefficient for each of the best possible models with the different number of predictors.

# In[40]:


models.apply(lambda row: row[1].rsquared, axis=1)


# Let's show some graphs to see how these models compare to each other.

# In[41]:


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


# We can see from the first graph that the RSS of the models is decreasing as the number of predictors increase. Also, from the second graph, we can see that the adjusted r sqared increases only up to a point of around 0.85, as we saw in the previous list, but it decreases sharply with all the predictors. The maximum adjusted r squared is marked with a point, as is the model with the lowest AIC and BIC.
# 
# We'll chose the model with 8 predictors to try to keep it simple, as after this point, the models only preform slightly better.

# In[42]:


data_train["subset_selection"] = models.loc[8, "model"].predict()
model_subset_RMSLE = RMSLE(data_train.subset_selection, data_train.visitors)

results_df.loc[4,"Model"] = "Subset selection"
results_df.loc[4,"RMSLE"] = model_subset_RMSLE
results_df


# This last model is the best up until now by RMSLE standards.
# 
# - Let's try a polynomial regression model with the past dow visitors variable, as it is the one with the highest t statistic, up to a 5th degree polynomial.

# In[43]:


poly_1 = smf.ols(formula='visitors ~ 1 + past_dow_visitors', data=data_train).fit()

poly_2 = smf.ols(formula='visitors ~ 1 + past_dow_visitors + I(past_dow_visitors ** 2.0)', data=data_train).fit()

poly_3 = smf.ols(formula='visitors ~ 1 + past_dow_visitors + I(past_dow_visitors ** 2.0) + I(past_dow_visitors ** 3.0)', data=data_train).fit()

poly_4 = smf.ols(formula='visitors ~ 1 + past_dow_visitors + I(past_dow_visitors ** 2.0) + I(past_dow_visitors ** 3.0) + I(past_dow_visitors ** 4.0)', data=data_train).fit()

poly_5 = smf.ols(formula='visitors ~ 1 + past_dow_visitors + I(past_dow_visitors ** 2.0) + I(past_dow_visitors ** 3.0) + I(past_dow_visitors ** 4.0) + I(past_dow_visitors ** 5.0)', data=data_train).fit()


# In[44]:


print(sm.stats.anova_lm(poly_1, poly_2, poly_3, poly_4, poly_5, typ=1))


# The model is ever increasing in goodness of fit, but actually doing so by just a little. Let's see it plotted to better understand it.

# In[45]:


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


# In[46]:


data_train["poly_regr"] = poly_5.predict()
model_poly_RMSLE = RMSLE(data_train.poly_regr, data_train.visitors)

results_df.loc[5,"Model"] = "Polynomial Regressor"
results_df.loc[5,"RMSLE"] = model_poly_RMSLE
results_df


# The polynomial regression wasn't actually an improvement over the linear regression.

# ### 5- Time series forecasting
# 
# We will use a single restaurant id (air_6b15edd1b4fbb96a) to evaluate it's time evolution data and use that to forecast the visitors.
# 
# - Let's first explore the chosen id creating a time index.

# In[47]:


df_time = data_train[data_train.air_store_id == 'air_6b15edd1b4fbb96a']

df_time.set_index(pd.to_datetime(df_time.visit_year*10000+df_time.visit_month*100                                 +df_time.visit_day,format='%Y%m%d'), inplace=True)

fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(16, 7))

axes[0].plot(df_time.visitors, color='navy', linewidth=4)
axes[1].plot(df_time.visitors[df_time.visit_month > 10], color='navy', linewidth=4)


# We can see that there is a spike for each Saturday in the time series, and after the spike come the low values for Mondays through Wednesdays. The frequency of the series is hence weekly.
# Let's try now to forecast this time series using several methods.
# 
# - First, let's see how the first model from the previous section would behave, the ones with the total average visitors and the specific restaurant average visitors.

# In[50]:


model_mean_RMSLE = RMSLE(df_time.visitors_mean, df_time.visitors)
model_rest_mean_RMSLE = RMSLE(df_time.visitors.mean(), df_time.visitors)

results_df_time = pd.DataFrame(columns=["Model", "RMSLE"])
results_df_time.loc[0,"Model"] = "Total Mean"
results_df_time.loc[0,"RMSLE"] = model_mean_RMSLE
results_df_time.loc[1,"Model"] = "Restaurant Mean"
results_df_time.loc[1,"RMSLE"] = model_rest_mean_RMSLE

results_df_time


# This is in line with the numbers obtained in the previous section, so let's now do something new: Time Series Decomposition.
# 
# - We will decompose the time series into trend and seasonality

# In[51]:


decomposition = seasonal_decompose(df_time.log_visitors, model="additive", freq=6)
decomposition.plot();


# Let's store this information into the dataframe and predict the visitors using them.

# In[52]:


trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid

df_time['power_decomp'] = np.exp(trend + seasonal)


# In[53]:


model_Decomp_RMSLE = RMSLE(df_time.power_decomp, df_time.visitors)

results_df_time.loc[2,"Model"] = "Time Decomposition"
results_df_time.loc[2,"RMSLE"] = model_Decomp_RMSLE
results_df_time


# OK, this is not a bad number having seen the numbers from the previous section, but we still have to create the multiple linear regressors for this specific restaurant.
# 
# - Let's start by forward subsetting the predictors.

# In[54]:


models_time = pd.DataFrame(columns=["RSS", "model"])

predictors = []
y=df_time.visitors
X = df_time[['visit_year', 'visit_month', 'visit_day', 'reserve_visitors','holiday_flg','latitude','longitude',                'dow_Friday','dow_Monday','dow_Tuesday','dow_Wednesday','dow_Thursday','dow_Saturday','dow_Sunday',                 'visitors_rest_mean','past_dow_visitors']].astype('float64')

for i in range(1, len(X.columns) + 1):    
    models_time.loc[i] = forward(predictors)
    predictors = models_time.loc[i]["model"].model.exog_names


# Again, let's plot them to choose an appropriate number of predictors.

# In[55]:


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


# Although the best adjusted r squared is obtained with 2 models, the RSS is still too high, so we'll chose 10 predictors as a compromise solution.

# In[56]:


df_time["subset_selection"] = models_time.loc[10, "model"].predict()
model_subset_RMSLE = RMSLE(df_time.subset_selection, df_time.visitors)

results_df_time.loc[3,"Model"] = "Subset selection"
results_df_time.loc[3,"RMSLE"] = model_subset_RMSLE
results_df_time


# This is a great improvement from the previous models.
# 
# - Let's see now how a multiple linear regression model would perform.

# In[57]:


#We get rid of the genres, as they do not help making a better model
df_time.drop(list(df_time.filter(regex = 'genre_')), axis = 1, inplace = True)
df_time.dropna(axis=0,how='any',inplace=True)

model = sm.OLS.from_formula('visitors ~ ' + '+'.join(df_time.columns.difference(['visitors', 'log_visitors','air_store_id','visitors_mean', 'subset_selection','past_dow_predict','power_decomp','poly_regr'])), df_time)

result = model.fit()
print(result.summary())


# In[58]:


df_time["linear_regr"] = result.predict()

# RMSLE for linear regressor
model_lin_RMSLE = RMSLE(df_time.linear_regr, df_time.visitors)

results_df_time.loc[4,"Model"] = "Linear Regressor"
results_df_time.loc[4,"RMSLE"] = model_lin_RMSLE
results_df_time


# So we have a winner. Compairing the multiple linear regressor RMSLE with the RMSLEs obtained in the previous section, we can conclude that having a model for each restaurant will improve the prediction of the visitors for that restaurant. 
# 
# The only problem now is that not all restaurants in the test data have enough information in the train data, there are restaurants that are not even included in the train data, so we'll have to make just the best possible model for each group of them.
# 
# - We'll start by creating a multiple linear regression model for each restaurant in the train data.

# In[59]:


#Let's get rid of the columns that won't be used in the final predictions.
data_train.drop(data_train[['air_area_name', 'latitude','past_dow_visitors','longitude','visitors_mean','linear_regr','store_change','past_dow_predict','subset_selection','poly_regr','log_visitors']], axis=1, inplace=True)
data_train.drop(list(data_train.filter(regex = 'genre_')), axis = 1, inplace = True)


# In[60]:


restaurants = data_train.air_store_id.unique()
RMSLEs = []
models_dict = {}

for i,restaurant in enumerate(restaurants):
    if i%100 == 0 or i==(len(restaurants)-1):
        print("Model {} of {}".format(i+1,len(restaurants)))
        
    df_temp = data_train[data_train.air_store_id == restaurant]
    df_temp.dropna(axis=0,how='any',inplace=True)
    model = sm.OLS.from_formula('visitors ~ ' + '+'.join(df_temp.columns.difference(['visitors',                                'air_store_id'])), df_temp).fit()
    RMSLEs.append(RMSLE(model.predict(), df_temp.visitors))
    models_dict[restaurant] = model


# We'll create now the models for the restaurants with no reserved visitors info, as this data is not complete for the forecasted weeks.

# In[61]:


RMSLEhalf = []
half_models_dict = {}

for i,restaurant in enumerate(restaurants):
    if i%100 == 0 or i==(len(restaurants)-1):
        print("Model {} of {}".format(i+1,len(restaurants)))
        
    df_temp = data_train[data_train.air_store_id == restaurant]
    df_temp.dropna(axis=0,how='any',inplace=True)
    model = sm.OLS.from_formula('visitors ~ ' + '+'.join(df_temp.columns.difference(['visitors',                                'air_store_id','reserve_visitors'])), df_temp).fit()
    RMSLEhalf.append(RMSLE(model.predict(), df_temp.visitors))
    half_models_dict[restaurant] = model


# And finally, a last model for those restaurants that are new in the test dataframe.

# In[62]:


nodata_model = sm.OLS.from_formula('visitors ~ ' + '+'.join(data_train.columns.difference(['visitors',                                   'air_store_id','reserve_visitors','visitors_rest_mean'])), data_train).fit()
RMSLE_rest = RMSLE(nodata_model.predict(), data_train.visitors)


# - Let's see how these newly created models compare with the ones obtained in the modeling section.

# In[63]:


results_df.loc[6,"Model"] = "Regressor per id"
results_df.loc[6,"RMSLE"] = np.mean(RMSLEs)
results_df.loc[7,"Model"] = "Regressor per id w/o reserves"
results_df.loc[7,"RMSLE"] = np.mean(RMSLEs)
results_df.loc[8,"Model"] = "New id model"
results_df.loc[8,"RMSLE"] = RMSLE_rest

results_df


# We can see that the models for the ids in the train data will perform much better than the model for the new restaurants that will appear in the test dataset.
# 
# - We'll store all the created models

# In[64]:


def save_model(obj, name):
    with open('../models/'+ name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
        
save_model(models_dict,'full_models')
save_model(half_models_dict,'half_models')
save_model(nodata_model,'no_data_model')


# ### 6- Generation of predictions and submission file
# 
# With all the models created, we can finally predict the visitors for the test dataset.
# 
# - We'll start by filtering the test dataset by the first predicted date.

# In[65]:


data_test.set_index(pd.to_datetime(data_test.visit_year*10000+data_test.visit_month*100+                                   data_test.visit_day,format='%Y%m%d'), inplace=True)
data_test = data_test[data_test.index > '2017-04-22']


# - We need to complete the forecasting days, as not all days had visitors and only those were the ones in our datasets.

# In[66]:


restaurants = sample_submission.id.str[:20].unique()

start_date = datetime.datetime.strptime('2017-04-23', "%Y-%m-%d")
end_date = datetime.datetime.strptime('2017-05-31', "%Y-%m-%d")

#We'll use a new dataframe to store the prediction data and the new dates
predict_df = pd.DataFrame(columns=data_test.columns)

for restaurant in restaurants:
    while start_date <= end_date: 

        if len(data_test[(data_test['air_store_id']==restaurant) & (data_test['visit_month']==start_date.month)                         & (data_test['visit_day']==start_date.day)]):
            predict_df = predict_df.append(data_test[(data_test['air_store_id']==restaurant)                                                     & (data_test['visit_month']==start_date.month)                                                      & (data_test['visit_day']==start_date.day)], ignore_index=True)
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
# 
# - Lets format now the dataframe in order to be able to use it in the previously obtained models.

# In[67]:


predict_df = pd.merge(predict_df, date_info, left_on=['visit_year','visit_month','visit_day'],                       right_on=['calendar_year','calendar_month','calendar_day'], how='left')
predict_df = pd.get_dummies(predict_df, columns=['dow'])
predict_df = pd.merge(predict_df, data_train[['air_store_id','visitors_rest_mean']].drop_duplicates(),                       on='air_store_id', how='left')

predict_df.drop(['holiday_flg_x','day_of_week','calendar_year','calendar_month','calendar_day','latitude',                 'longitude','air_area_name','genre'], axis=1, inplace=True)

predict_df=predict_df.rename(columns = {'holiday_flg_y':'holiday_flg'})

predict_df.sort_values(by=['reserve_visitors','visitors_rest_mean'], ascending=[True,True], inplace=True)

predict_df.reserve_visitors = pd.to_numeric(predict_df.reserve_visitors)
predict_df.visit_year = pd.to_numeric(predict_df.visit_year)
predict_df.visit_month = pd.to_numeric(predict_df.visit_month)
predict_df.visit_day = pd.to_numeric(predict_df.visit_day)


# - Let's check, how many models for restaurants from the train data do we have, and how many restaurants in total are there in the predict dataframe?

# In[68]:


print("Different restaurants in test set:",len(predict_df.air_store_id.unique()))
print("Different restaurants in train set:",len(data_train.air_store_id.unique()))


# Sadly, we see that there is a big number of restaurants for which we had not calculated models, this will affect the result.
# 
# - Finally, let's make the prediction for each restaurant and date.

# In[69]:


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

    if index%5000 == 0 or i==(len(predict_df)-1):
        print("Prediction {} of {}".format(index+1,len(predict_df)))


# Let's see what kind of predictions did we get.

# In[70]:


print(predict_df.visitors.describe())


# There are a few negative numbers and a few to big to be true. Let's crop that numbers.

# In[71]:


predict_df.visitors[predict_df['visitors']<data_train.visitors.min()] = data_train.visitors.mean()
predict_df.visitors[predict_df['visitors']>data_train.visitors.max()] = data_train.visitors.mean()


# - Finally, let's create the file with the appropriate format to be submitted to the Kaggle challenge.

# In[72]:


def submission_format(row):
    return row['air_store_id']+"_"+str(row['visit_year'])+"-"+str(row['visit_month']).zfill(2)+"-"+str(row['visit_day']).zfill(2)

predict_df['id'] = predict_df.apply(submission_format, axis=1)
predict_df['visitors'] = predict_df['visitors'].round(0).astype(int)

predict_df[['id','visitors']].to_csv('../data/processed/submission.csv', index=False)


# We will store these predictions in the DB so we can compare it in the future if necessary.

# In[73]:


predict_df.to_sql("regression_predictions", conn, if_exists="replace")
conn.close()

