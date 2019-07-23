import sys
### change to the current script folder
sys.path.insert(0, 'E:\\Work\\aspirantura\\kafedra\\upwork\\RtoP\\PyModels\\LinearRegression')
import h2o
import pandas as pd
import numpy as np
import json
import R2P_linregr_dataprep
import seaborn as sns
import matplotlib.pyplot as plt
from h2o.estimators.glm import H2OGeneralizedLinearEstimator
from h2o.estimators.gbm import H2OGradientBoostingEstimator
from h2o.estimators.kmeans import H2OKMeansEstimator
from h2o.estimators.pca import H2OPrincipalComponentAnalysisEstimator
from statsmodels.stats.outliers_influence import variance_inflation_factor
import scipy
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

from scipy import stats
from statsmodels.tsa.holtwinters import Holt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima_model import ARIMA
from numpy import array

import keras
from keras.models import Sequential
from keras.layers import Activation, Dropout, Flatten, Dense

from keras.optimizers import Adam

from R2P_linregr_dataprep import columns_data_type, ch_sq_test, correlations, transformation, variable_importance_h2o, transformation_inv 


"""
  Split method a univariate sequence into samples 
  for futher neural net training 
 
 input:
    sequence - time series for splitting
    n_steps - number of prev. values of the time series that takes into account
 output:
     x - prev. values
     y - predicted value

"""
def split_sequence(sequence, n_steps):
	X, y = list(), list()
	for i in range(len(sequence)):
		# find the end of this pattern
		end_ix = i + n_steps
		# check if we are beyond the sequence
		if end_ix > len(sequence)-1:
			break
		# gather input and output parts of the pattern
		seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
		X.append(seq_x)
		y.append(seq_y)
	return array(X), array(y)


"""
  MLP forecasting method a univariate sequence 
   
 input:
    time_ser - time series
    n_steps - number of prev. values in time series
    no_of_periods_to_forecast - number of forvard values
 output:
     forecasted values
"""
def nnetar(time_ser, n_steps, no_of_periods_to_forecast):
    x, y = split_sequence(time_ser, n_steps)
    
    # define model
    model = Sequential()
    model.add(Dense(100, activation='relu', input_dim=n_steps))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    # fit model
    model.fit(x, y, epochs=2000, verbose=0)
    
    # predictions model
    forecasted = []
    x_input_index = list(time_ser.index[-n_steps:])
    x_input = array(list(time_ser[x_input_index]))
    
    for j in range(no_of_periods_to_forecast):
    #    print(j)
        x_input = x_input.reshape((1, n_steps))
        pred = model.predict(x_input, verbose=0)
    #    print(pred)
        forecasted.append(pred)
        x_input = np.append(x_input, pred, axis=1)
        x_input = x_input[0][-n_steps:]
    return forecasted 


"""
 Prediction models for time series forecasting  
 
 input:
     data_initial - data frame after preprocessing 
     variable - observed time series
     date - date of the time series observations
     model - (HoltWinters, ARIMA, NNETAR(MLP) )
     no_of_periods_to_forecast
     independentVariables - variables for Multivariate Time Series Forecasting using MLP
     test - data for testing
 output:
     decomp_overall
     slope_text
     Percentage_Variance_Explained_by_trend_text
     Percentage_Variance_Explained_by_seasonality_text
     Percentage_Variance_Explained_by_randomness_text
     seasonal_text
     seasonal_component
     model_text
     forecasted
     ts_test
     MAPE
     MSE
     ME
     MAE
     
"""
def forecasting_model(data_initial,variable_col,date_col,model,independentVariables,test_split):
    
    if date_col == '':
        index = data_initial.index
    else:
        index = pd.DatetimeIndex(data_initial[date_col])
    
    ts = pd.Series(data_initial[variable_col])
    ts.index = index 
    ts = ts.sort_index()
    test_ind = round(len(ts)*test_split)
    ts_train = ts[:len(ts)-test_ind]
    ts_test = ts[-test_ind:]
    no_of_periods_to_forecast = len(ts_test)
    
    #ts.plot()
    
    ########## resampling with time frame monthly, quarterly and annual          
    try:
        # Resampling to monthly frequency
        ts_monthly = ts_train.resample('M').mean()
        # Resampling to quarterly frequency
        ts_quarterly = ts_train.resample('Q-DEC').mean()
        #Resampling to annual frequency
        ts_annual = ts_train.resample('A-DEC').mean()
        
    #    ts_monthly.plot()
    #    ts_quarterly.plot()
    #    ts_annual.plot()
    
    except:
        print('You need specify the date column')
    
    days_available = len(data_initial.dropna()) 
    months_available = len(ts_monthly.dropna())
    years_available= len(ts_annual.dropna())
    
    if  days_available+months_available+years_available == 0:
        error = "Insufficient data for time series analysis with "+days_available+" Days, "+months_available+" Months and "+years_available+" Years"
        print(error)
    #    return 1
    else:
        error = ''
    
    try:
        result = seasonal_decompose(ts_train, model='additive',freq=1)
        #result.plot()
        decomp_overall = pd.DataFrame([list(result.observed), list(result.trend), list(result.seasonal), list(result.resid), list(ts_train.index)])
        decomp_overall = decomp_overall.transpose()
        decomp_overall.columns = ["actual","trend","seasonality","randomness","date"]
    except:
        error = "Unable to Decompose time series"
        print(error)
        
       
    Percentage_Variance_Explained_by_trend = (decomp_overall['trend'].dropna().var())/(decomp_overall['actual'].dropna().var())*100
    Percentage_Variance_Explained_by_seasonality = (decomp_overall['seasonality'].dropna().var())/(decomp_overall['actual'].dropna().var())*100
    Percentage_Variance_Explained_by_randomness = (decomp_overall['randomness'].dropna().var())/(decomp_overall['actual'].dropna().var())*100
    
    gradient, intercept, r_value, p_value, std_err = stats.linregress(list(decomp_overall['actual']), range(len(decomp_overall)))
    slope_with_time = gradient
    slope_text = "<li class=\"nlg-item\"><span class=\"nlgtext\"> The slope of time for <font style=\"color: #078ea0;\">"+variable_col+" </font> is <b>"+str(round(slope_with_time,4))+"</b> , For every one unit change in time <font style=\"color: #078ea0;\">"+variable_col+" </font> is effected by <b>"+str(round(slope_with_time,4))+"</b> units </span></li>"
    
    model = 'NNETAR'
        
    if (model == "Holtwinters"):
        forecast_model = Holt(np.asarray(decomp_overall['actual'])).fit()
        forecasted = forecast_model.predict(start=len(ts_train)-1, end=len(ts_train)-1+no_of_periods_to_forecast-1)
        forecasted = pd.Series(forecasted)
    elif (model == "ARIMA"):
        forecast_model = ARIMA(ts_train, order=(7,0,1))
        model_fit = forecast_model.fit(disp=0)
        forecasted = model_fit.predict(start=len(ts_train)-1, end=len(ts_train)-1+no_of_periods_to_forecast-1)
    elif (model == "NNETAR"):
        forecasted = nnetar(ts_train,3,no_of_periods_to_forecast)    
        forecasted = pd.Series(forecasted)
    else:
        print('Model method not specified')    
    #### Error metrics evaluate
    forecasted.index = ts_test.index
        
    MAPE = np.mean((np.abs(forecasted - ts_test)/np.abs(ts_test))*100)
    MSE = np.mean((forecasted - ts_test)**2)
    ME = np.mean(forecasted - ts_test)
    MAE = np.mean(np.abs(forecasted - ts_test))
    
    # NLG with model description
    
    Percentage_Variance_Explained_by_trend_text = "<li class=\"nlg-item\"><span class=\"nlgtext\"> Variance Explained By <font style=\"color: #078ea0;\"> Trend </font> is <b>"+str(round(Percentage_Variance_Explained_by_trend,4))+"</b> </span></li>"
    Percentage_Variance_Explained_by_seasonality_text = "<li class=\"nlg-item\"><span class=\"nlgtext\"> Variance Explained By <font style=\"color: #078ea0;\"> Seasonality </font>is <b>"+str(round(Percentage_Variance_Explained_by_seasonality,4))+"</b> </span></li>"
    Percentage_Variance_Explained_by_randomness_text  = "<li class=\"nlg-item\"><span class=\"nlgtext\"> Variance Explained By <font style=\"color: #078ea0;\"> Randomness </font> is <b>"+str(round(Percentage_Variance_Explained_by_randomness,4))+"</b> </span></li>"
    
    seasonal_text = "The amount of "+variable_col+" is effected due to seasonality with time"
    seasonal_component = pd.concat([decomp_overall["date"],decomp_overall["seasonality"]], axis=1)
    model_text = "Model has Forecasted the "+ variable_col+" for next "+str(no_of_periods_to_forecast)+" periods"
    
    output = [decomp_overall,slope_text,Percentage_Variance_Explained_by_trend_text,Percentage_Variance_Explained_by_seasonality_text,Percentage_Variance_Explained_by_randomness_text,seasonal_text,seasonal_component,model_text,forecasted,ts_test,MAPE,MSE,ME,MAE]  
    
    return output



"""
 Methods for forecasting problem 
 
 input:
    #* @param dbHost 
    #* @param dbPort 
    #* @param userName 
    #* @param password
    #* @param dbName 
    #* @param query
    #* @param yValue  e.g. date column  
    #* @param xValues e.g. time series column
    #* @param parametersObj
    #* @param columnsArray
 output:
     text description of the model

"""

def forecasting(dbHost="",dbPort="",userName="",password="",dbName="",query="",xValues="",yValue="",parametersObj="",columnsArray=""):
    ##############
    # connecting to BD
    ##############

    data = pd.read_csv("E:/Work/aspirantura/kafedra/upwork/RtoP/Models_sep/Datasets/50000_Sales_Records_Dataset_e.csv")
    data, columnsArray_e = columns_data_type(data[0:100], columnsArray)
    ent_cor,chisq_dependency,data,rm_cols, miss_cols, obj_t = correlations(data, columnsArray=columnsArray_e, method = 'impute', no_rem_col =['Order Date'])

    output = forecasting_model(data,variable_col = xValues,date_col = yValue, model = parametersObj, independentVariables='',test_split = 0.4)
   
    return output

    
############## testing forecasting

### example of columnsArray

columnsArray = '[{"columnDisplayName":"S@#@#no","tableDisplayType":"string","columnName":"S#@#@no"},{"columnDisplayName":"Region","tableDisplayType":"string","columnName":"Region"},{"columnDisplayName":"Country","tableDisplayType":"string","columnName":"Country"},{"columnDisplayName":"Item Type","tableDisplayType":"string","columnName":"Item Type"},{"columnDisplayName":"Sales Channel","tableDisplayType":"string","columnName":"Sales Channel"},{"columnDisplayName":"Order Priority","tableDisplayType":"string","columnName":"Order Priority"},{"columnDisplayName":"Order Date","tableDisplayType":"string","columnName":"Order Date"},{"columnDisplayName":"Order ID","tableDisplayType":"string","columnName":"Order ID"},{"columnDisplayName":"Ship Date","tableDisplayType":"string","columnName":"Ship Date"},{"columnDisplayName":"Units Sold","tableDisplayType":"number","columnName":"Units Sold"},{"columnDisplayName":"Unit Price","tableDisplayType":"number","columnName":"Unit Price"},{"columnDisplayName":"Unit Cost","tableDisplayType":"number","columnName":"Unit Cost"},{"columnDisplayName":"Total Revenue","tableDisplayType":"number","columnName":"Total Revenue"},{"columnDisplayName":"Total Cost","tableDisplayType":"number","columnName":"Total Cost"},{"columnDisplayName":"target_var","tableDisplayType":"number","columnName":"target_var"}]'    
#columnsArray = '[{"columnDisplayName":"Region","tableDisplayType":"string","columnName":"Region"},{"columnDisplayName":"Country","tableDisplayType":"string","columnName":"Country"},{"columnDisplayName":"Item Type","tableDisplayType":"string","columnName":"Item Type"},{"columnDisplayName":"Sales Channel","tableDisplayType":"string","columnName":"Sales Channel"},{"columnDisplayName":"Order Priority","tableDisplayType":"string","columnName":"Order Priority"},{"columnDisplayName":"Order Date","tableDisplayType":"string","columnName":"Order Date"},{"columnDisplayName":"Order ID","tableDisplayType":"string","columnName":"Order ID"},{"columnDisplayName":"Ship Date","tableDisplayType":"string","columnName":"Ship Date"},{"columnDisplayName":"Units Sold","tableDisplayType":"number","columnName":"Units Sold"},{"columnDisplayName":"Unit Price","tableDisplayType":"number","columnName":"Unit Price"},{"columnDisplayName":"Unit Cost","tableDisplayType":"number","columnName":"Unit Cost"},{"columnDisplayName":"Total Revenue","tableDisplayType":"number","columnName":"Total Revenue"},{"columnDisplayName":"Total Cost","tableDisplayType":"number","columnName":"Total Cost"},{"columnDisplayName":"target_var","tableDisplayType":"number","columnName":"target_var"}]'    
try:
    columnsArray = json.loads(columnsArray)
except:
    print('json format not valid')
   
forecast_model = forecasting(xValues='targetvar',yValue='Order Date', parametersObj='NNETAR', columnsArray=columnsArray)
    


