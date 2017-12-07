from data.make_dataset import *
from visualization.visualize import *
from models.regression import *
from models.time_series import *
from models.predictions import *

df = createDf()

data_train = df[df.visitors.notnull()]
data_test = df[df.visitors.isnull()]

# We will create a new column with the natural logarithm of the visitor numbers, in case that helps us in the forecasting section later.
data_train['log_visitors'] = data_train.visitors.apply(lambda x: np.log(x))

#Let's visualize the data
visualize(df,data_train)

#Call to the regression analysis
data_train, results_df = regression(data_train)

#Time to do the time series analysis
print(time_series(data_train))

#Finally, let's create all the models that will be used for predictions
data_train, models_dict, half_models_dict, nodata_model = multiple_model_creation(data_train, results_df)

#Let's store the predictions in a new dataframe
predict_df = prediction_creation(data_train, data_test, models_dict, half_models_dict, nodata_model)

#Finally, let's create the csv file to be sent to the kaggle competition with their required format
predict_df[['id','visitors']].to_csv('../data/processed/submission_script.csv', index=False)
