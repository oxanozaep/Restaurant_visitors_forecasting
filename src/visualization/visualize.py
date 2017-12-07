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

def visualize(df,data_train):
    fig, ax = plt.subplots(figsize=(14,12));
    ax = sns.violinplot(x='dow', y="visitors", hue='holiday_flg',data=df, palette="muted", split=True)

    sns.jointplot(x='visitors', y='reserve_visitors', data=data_train, color='navy',size=10, space=0, kind='reg',marginal_kws={'hist_kws': {'log': True}})

    data_train_month = data_train[['visit_month','visitors','visit_year']].groupby(['visit_year','visit_month']).sum()
    data_train_month.plot(kind ="bar", y='visitors')

    data_train_month_av = data_train[['visit_month','visitors','visit_year']].groupby(['visit_month']).mean()
    data_train_month_av.plot(kind ="bar", y='visitors')

    cor = data_train.corr()
    plt.figure(figsize=(14,3))
    sns.heatmap(cor.loc[['visitors'], list(df)[:-1]]);

    data_train_Pivot = pd.pivot_table(data_train, values='visitors', columns='dow', index='visit_month')
    data_train_Pivot.plot();
    plt.legend(bbox_to_anchor=(1,1), loc="upper left")
