import h2o
h2o.init()
from h2o.estimators.glm import H2OGeneralizedLinearEstimator
from h2o.estimators.gbm import H2OGradientBoostingEstimator

import numpy as np
import pandas as pd # must be 0.24.0
import sklearn

from pandas.io.json import json_normalize
from scipy.stats import chi2_contingency
from scipy.stats import chi2
import itertools
from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import LabelEncoder
import statistics
from dateutil.parser import parse
import datetime
import calendar
####################################################################
#import json
#
#
#columnsArray = '[{"columnDisplayName":"S@#@#no","tableDisplayType":"string","columnName":"S#@#@no"},{"columnDisplayName":"Region","tableDisplayType":"string","columnName":"Region"},{"columnDisplayName":"Country","tableDisplayType":"string","columnName":"Country"},{"columnDisplayName":"Item Type","tableDisplayType":"string","columnName":"Item Type"},{"columnDisplayName":"Sales Channel","tableDisplayType":"string","columnName":"Sales Channel"},{"columnDisplayName":"Order Priority","tableDisplayType":"string","columnName":"Order Priority"},{"columnDisplayName":"Order Date","tableDisplayType":"string","columnName":"Order Date"},{"columnDisplayName":"Order ID","tableDisplayType":"string","columnName":"Order ID"},{"columnDisplayName":"Ship Date","tableDisplayType":"string","columnName":"Ship Date"},{"columnDisplayName":"Units Sold","tableDisplayType":"number","columnName":"Units Sold"},{"columnDisplayName":"Unit Price","tableDisplayType":"number","columnName":"Unit Price"},{"columnDisplayName":"Unit Cost","tableDisplayType":"number","columnName":"Unit Cost"},{"columnDisplayName":"Total Revenue","tableDisplayType":"number","columnName":"Total Revenue"},{"columnDisplayName":"Total Cost","tableDisplayType":"number","columnName":"Total Cost"},{"columnDisplayName":"target_var","tableDisplayType":"number","columnName":"target_var"}]'
##columnsArray = '[{"columnDisplayName":"Region","tableDisplayType":"string","columnName":"Region"},{"columnDisplayName":"Country","tableDisplayType":"string","columnName":"Country"},{"columnDisplayName":"Item Type","tableDisplayType":"string","columnName":"Item Type"},{"columnDisplayName":"Sales Channel","tableDisplayType":"string","columnName":"Sales Channel"},{"columnDisplayName":"Order Priority","tableDisplayType":"string","columnName":"Order Priority"},{"columnDisplayName":"Order Date","tableDisplayType":"string","columnName":"Order Date"},{"columnDisplayName":"Order ID","tableDisplayType":"string","columnName":"Order ID"},{"columnDisplayName":"Ship Date","tableDisplayType":"string","columnName":"Ship Date"},{"columnDisplayName":"Units Sold","tableDisplayType":"number","columnName":"Units Sold"},{"columnDisplayName":"Unit Price","tableDisplayType":"number","columnName":"Unit Price"},{"columnDisplayName":"Unit Cost","tableDisplayType":"number","columnName":"Unit Cost"},{"columnDisplayName":"Total Revenue","tableDisplayType":"number","columnName":"Total Revenue"},{"columnDisplayName":"Total Cost","tableDisplayType":"number","columnName":"Total Cost"},{"columnDisplayName":"target_var","tableDisplayType":"number","columnName":"target_var"}]'    
#try:
#    columnsArray = json.loads(columnsArray)
#except:
#    print('json format not valid')
#columnsArray = pd.DataFrame(columnsArray)
#columnsArray = columnsArray.replace(r'[^a-zA-Z0-9 -]', "", regex=True)
#
#data = pd.read_csv("E:/Work/aspirantura/kafedra/upwork/RtoP/Models_sep/Datasets/50000_Sales_Records_Dataset_e.csv",dtype=str)
##data = pd.DataFrame(np.genfromtxt('E:/Work/aspirantura/kafedra/upwork/RtoP/Models_sep/Datasets/50000_Sales_Records_Dataset_e.csv', dtype=str))
#data.info()
#
#
#data, columnsArray_e = columns_data_type(data[0:100], columnsArray)
#
#ent_cor,chisq_dependency,data,rm_cols, miss_cols, obj_t = correlations(data, columnsArray=columnsArray_e, method='predict')



##GBM_impute(data, columnsArray, rm_cols)
#
#columnsArray_ind = []
#for i in columnsArray['columnName']:
#    if i in rm_cols:
#        columnsArray_ind.append(list(columnsArray[columnsArray['columnName']==i].index)[0])
#columnsArray_ind1 = set(columnsArray.index)-set(columnsArray_ind)
#print(columnsArray_ind1)
#columnsArray_edit = columnsArray.iloc[list(columnsArray_ind1)]        
#
## select observations without NA's
#data_clean = data.dropna()
## creating H2O Frame and splitting for model train
#data_clean.info()
#hf = h2o.H2OFrame(data_clean)
#train, valid, test = hf.split_frame(ratios=[.8, .1])
#
## select observations with NA's
#data_na_index = [i for i in (set(list(data.index)) - set(list(data_clean.index))) ]
#data_na = data.iloc[data_na_index]
#
#model_accuracy = []
#for i in range(len(data_na)):
#    # select features with NA's in current row
#    print('Index in data_na_index')
#    print(i)
#    y_set = set(data_na.iloc[i].index) - set(data_na.iloc[i].dropna().index)
#    gbm = H2OGradientBoostingEstimator()
#    xValues = set(data_na.columns)-y_set
#    print(xValues)
#    
#    for yValue in y_set:
#        print('yValue from y_set for current index')
#        print(yValue)
#        print('GBM model training')
#        gbm.train(xValues, yValue, training_frame=train, validation_frame=valid)
#        model_accuracy.append(gbm.r2())
##        test_na = data_na.iloc[i].drop(y_set)
#        test_na = data_na.iloc[i]
#        test_na = pd.DataFrame(test_na).transpose()
#        
#        print('Missing value prediction with GBM model')
#        
#        test_na, columnsArray_edit = columns_data_type(test_na, columnsArray_edit)
#        print(test_na)
#        
##        test_na = test_na.drop(xValues,axis=1)
#        test_na = test_na.drop(yValue,axis=1)
#        print(test_na.info())
#        test_na = h2o.H2OFrame(test_na)
#        predicted = gbm.predict(test_na)
#        predicted = predicted.as_data_frame()
#        predicted_val = list(predicted['predict'])[0]
#        print("Predicted value")
#        print(predicted_val)
#        data_na[yValue].iloc[i] = predicted_val
#
#acc = np.mean(model_accuracy)
#frames = [data_clean, data_na]
#df = pd.concat(frames, axis=0)
#df.info()
##return df, acc 




####################################################################

"""
Return whether the string can be interpreted as a date.

 input:
     string - value to check on date format
 output:
     boolean - true - date, false - not date
"""

def is_date(string, fuzzy=False):
    try:
        float(string)
        return False
    except ValueError:
#        return False
        if (str(string)=='nan'):
            return False
        if str(string).isnumeric():
            return False
        try: 
            parse(str(string), fuzzy=fuzzy)
            return True
    
        except:
            return False

"""
Converting date feature to timestamp format

 input:
     data - raw data frame before columns_data_type()
 output:
     data - changed data frame with timestamp instead date string
"""

def date2stamp(data):
    for i in list(data.columns):
       
        for j in range(len(data)):
    #        print(j)
            if is_date(data[i].iloc[j]):
                
#                date = data[i].iloc[j]
#                date = datetime.datetime.strptime(date, "%d/%m/%y")
#                data[i].iloc[j] = str(date)
                try:
                    data[i].iloc[j] = str(datetime.datetime.strptime(data[i].iloc[j], "%d/%m/%y"))
                except:
                    continue
#                data[i].iloc[j] = calendar.timegm(date.utctimetuple())
    return data


"""
 Changing names of the features.
 Set types of features: numeric or categorical.
 Replacing special characters in column names.
 input:
     df - data frame
     columnsArray - data frame with names and types of columns
 output:
     df - changed data frame
"""
def columns_data_type(df, columnsArray = ""):
    
    df = date2stamp(df)
    columnsArray = pd.DataFrame(columnsArray)
    columnsArray = columnsArray.replace(r'[^a-zA-Z0-9 -]', "", regex=True)
    df.columns = columnsArray['columnName'] 
    df = df.replace(r'^\s*$', np.nan, regex=True)

    for i in columnsArray['columnName']:
        sub = columnsArray[columnsArray['columnName'] == i]
        if (sub['tableDisplayType'].values[0] == 'number'):
            df[i] = pd.to_numeric(df[i].values)
        if (sub['tableDisplayType'].values[0] == 'string'):
#            df[i] = df[i].astype(str)
            df[i] = df[i].astype('category')
            
            
    return df,columnsArray

"""
 Imputation or removing of missing values using prediction model.
 Replacing blanks with NA's.
 input:
     df - data frame after preprocessing
     columnsArray - 
     rm_cols - removed columns by remove_col() 
 output:
     df - changed data frame
"""
def GBM_impute(data, columnsArray, rm_cols):

    columnsArray_ind = []
    for i in columnsArray['columnName']:
        if i in rm_cols:
            columnsArray_ind.append(list(columnsArray[columnsArray['columnName']==i].index)[0])
    columnsArray_ind1 = set(columnsArray.index)-set(columnsArray_ind)
#    print(columnsArray_ind1)
    columnsArray_edit = columnsArray.iloc[list(columnsArray_ind1)]        
    
    # select observations without NA's
    data_clean = data.dropna()
    # creating H2O Frame and splitting for model train
    data_clean.info()
    hf = h2o.H2OFrame(data_clean)
    train, valid, test = hf.split_frame(ratios=[.8, .1])
    
    valid.columns
    # select observations with NA's
    data_na_index = [i for i in (set(list(data.index)) - set(list(data_clean.index))) ]
    data_na = data.iloc[data_na_index]

    model_accuracy = []
    for i in range(len(data_na)):
        # select features with NA's in current row
#        print('Index in data_na_index')
#        print(i)
        y_set = set(data_na.iloc[i].index) - set(data_na.iloc[i].dropna().index)
        gbm = H2OGradientBoostingEstimator()
        xValues = set(data_na.columns)-y_set
#        print(xValues)
        
        for yValue in y_set:
#            print('yValue from y_set for current index')
#            print(yValue)
#            print('GBM model training')
            gbm.train(xValues, yValue, training_frame=train, validation_frame=valid)
            model_accuracy.append(gbm.r2())
    #        test_na = data_na.iloc[i].drop(y_set)
            test_na = data_na.iloc[i]
            test_na = pd.DataFrame(test_na).transpose()
            
#            print('Missing value prediction with GBM model')
            
            test_na, columnsArray_edit = columns_data_type(test_na, columnsArray_edit)
#            print(test_na)
            
    #        test_na = test_na.drop(xValues,axis=1)
            test_na = test_na.drop(yValue,axis=1)
            print(test_na.info())
            test_na = h2o.H2OFrame(test_na)
            predicted = gbm.predict(test_na)
            predicted = predicted.as_data_frame()
            predicted_val = list(predicted['predict'])[0]
#            print("Predicted value")
#            print(predicted_val)
            data_na[yValue].iloc[i] = predicted_val
    
    acc = np.mean(model_accuracy)
    frames = [data_clean, data_na]
    df = pd.concat(frames, axis=0)
    return df, acc 



"""
 Imputation or removing of missing values using mean an mode.
 Replacing blanks with NA's.
 input:
     df - data frame
     method
         drop - drop all NA's
         impute - mean for numeric and mode - for categorical
         predict - impute missing val. with prediction model
     columnsArray - 
     rm_cols - removed columns by remove_col()
 output:
     df - changed data frame
"""
def missing_val_impute(df, method, columnsArray, rm_cols):
    try:

        miss_count = pd.DataFrame(df.isna().sum())
        miss_count.columns = ['miss_count']
        
        cat_data = df.select_dtypes(include=['category']).copy()
        num_data = df.select_dtypes(include=['number']).copy()

        if (method == 'drop'):
            df = df.dropna()
        elif (method == 'impute'):
            num_data = num_data.fillna(num_data.mean())
            for i in list(cat_data.columns):
                cat_data[i] = cat_data[i].fillna((cat_data[i].mode(dropna=True))[0])
            frames = [cat_data,num_data]
            df = pd.concat(frames, axis=1)
        elif (method == 'predict'):
            df, acc = GBM_impute(df, columnsArray, rm_cols)
            
        else:
            print("Imputation method not specify")        
    except:
        print("Imputation method doesn't meet the data")
        df = df.dropna(axis=0)
    return df, miss_count

"""
 Removing columns that contains huge number of levels
 Removing zero variance column
 input:
     df - data frame
     ratio - ratio observations to levels
 output:
     df - changed data frame
     removed_cols - list of removed columns
"""
def remove_col(df, ratio):
    try:
        cat_data = df.select_dtypes(include=['category']).copy()
        num_data = df.select_dtypes(include=['number']).copy()

        num_level_cat = []
        removed_cols = []
        for i in list(cat_data.columns):
            cat_list = list(cat_data[i].unique())
            num_obs = cat_data[i].count() 
            for j in cat_list:
                num_level_cat.append([i,j,cat_data[i][cat_data[i]== j].count(),num_obs])
        num_level_cat = pd.DataFrame(num_level_cat)
        num_level_cat.columns = ['category','level','count_level','count_observ']
        
        for i in list(num_level_cat['category'].unique()):
            if (len(cat_data) / num_level_cat['level'][num_level_cat['category']==i].count() < ratio):
                cat_data = cat_data.drop(i, 1)
                removed_cols.append(i)
#Removing zero variance column
        var = pd.DataFrame(num_data.var())
        for i in list(var.index):
            if list(var[var.index==i][0])[0] == 0:
                num_data = num_data.drop(i, 1)
                removed_cols.append(i)
        frames = [cat_data,num_data]
        transformed_data = pd.concat(frames, axis=1)
           
    except:
        print("Exception in removing columns")
    return transformed_data,removed_cols




"""
 chi square test for correlation between categorical features
 input:
     cat_data - data frame with categorical features, encoded
 output:
     data frame with chi square metrics
"""
def ch_sq_test(cat_data):
    
    ncol = len(cat_data.columns)
    test = []
    if ncol>1:
        combos = list(itertools.combinations(range(0,ncol), 2))
        combos = pd.DataFrame(combos)
        ind1 = list(combos[0])
        ind2 = list(combos[1])
        print(list(cat_data[cat_data.columns[ind1[1]]]))
        for i in range(len(ind1)):
            print(i)

            try:
                test.append(chi2_contingency([list(cat_data[cat_data.columns[ind1[i]]]),list(cat_data[cat_data.columns[ind2[i]]])]))
            except:
                continue
        test = pd.DataFrame(test)
        #test.columns = ['stat', 'p-val', 'dof', 'expected']
        return test    
    elif ncol == 1:
        print("There is only one category field exists")
    else:
        print("No category field exists")
     

"""
 Transformation method implements 
 YeoJohnson transformation for numeric features 
 (power transform featurewise to make data more Gaussian-like)
 input:
     df - data frame 
 output:
     transformed data frame
     PowerTransformer() object for invert transformation
"""
def transformation(data):
    cat_data = data.select_dtypes(include=['category']).copy()
    num_data = data.select_dtypes(include=['number']).copy()
    pt = PowerTransformer() 

    transformed = pt.fit(num_data).transform(num_data)
    transformed = pd.DataFrame(transformed)
    transformed.columns = num_data.columns

    frames = [cat_data,transformed]
    transformed_data = pd.concat(frames, axis=1)
    return transformed_data, pt     

"""
 Transformation invert method get the origin values back after
 YeoJohnson transformation
 input:
     df - data frame 
 output:
     original values data frame
"""
def transformation_inv(data, obj):
#    cat_data = data.select_dtypes(include=['category']).copy()
#    num_data = data.select_dtypes(include=['number']).copy() 
    num_data = data
    transformed = obj.inverse_transform(num_data)
    transformed = pd.DataFrame(transformed)
    transformed.columns = num_data.columns

#    frames = [cat_data,transformed]
#    transformed_data = pd.concat(frames, axis=1)
    return transformed


"""
 Correlations in the data
 Pearson test for numerical
 Chi squared test for categorical
 input:
     data - data frame outputed be columns_data_type()
     columnsArray - 
     method - method for missing val. imput (drop, impute, predict) 
 output:
     corr matrix between numeric features
     chi squared method result for categorical features
     preprocessed data
     list of exclude columns
     list of missing values amount for each columns 
"""
def correlations(data, columnsArray, method):
    # data type conversion and deleting missing values 
    data, rm_cols = remove_col(data, ratio=3)
    data, miss_cols = missing_val_impute(data, method=method, columnsArray=columnsArray, rm_cols=rm_cols)
    print("Data info after missing_val_impute()")
    print(data.info())
    data, obj_t = transformation(data)
    print("Data info after missing_val_impute()")
    print(data.info())
    cat_data = data.select_dtypes(include=['category']).copy()
    num_data = data.select_dtypes(include=['number']).copy()
    if (len(num_data.columns)>1):
        ent_cor = num_data.corr()
    else:
        print("There is only one feature exists. You need at least two to analyse")
    if (len(cat_data.columns)>1):
        chisq_dependency = ch_sq_test(cat_data)
    else:
        print("There is only one feature exists. You need at least two to analyse")
#    frames = [cat_data, num_data]
#    data = pd.concat(frames, axis=1)
        
    return ent_cor,chisq_dependency,data,rm_cols, miss_cols, obj_t  

"""
 Determine variable importance method implements 
 h2o.glm and h2o.gbm models for further using h2o.varimp 
 
 input:
     df - data frame 
     variable - dependent variable (y)
 output:
     matrix of variable importance
"""
def variable_importance_h2o(data, predictors, response_col):
    #cat_data = data.select_dtypes(include=['category']).copy()
    num_data = data.select_dtypes(include=['number']).copy()

    if(data[response_col].dtypes == 'float') or (data[response_col].dtypes == 'int'):
        print("Finding variable importance by taking given numeric variable as a dependent variable")
        
        hf = h2o.H2OFrame(num_data)
    
        train, valid, test = hf.split_frame(ratios=[.8, .1])    
        
        
        glm_model = H2OGeneralizedLinearEstimator(family = 'gaussian')

        glm_model.train(predictors, response_col, training_frame= train, validation_frame=valid)

        
        var_imp1 = glm_model.varimp()

        
        gbm = H2OGradientBoostingEstimator()
        gbm.train(predictors, response_col, training_frame= train, validation_frame=valid)

        var_imp2 = gbm.varimp()
        
        
        Fin_imp_var = [var_imp1, var_imp2]
        return Fin_imp_var
    else:
        print("Finding variable importance by taking categorical variables as dependent variable")
        hf = h2o.H2OFrame(data)
    
        train, valid, test = hf.split_frame(ratios=[.8, .1])    

        gbm = H2OGradientBoostingEstimator()
        gbm.train(predictors, response_col, training_frame= train, validation_frame=valid)
#        print(gbm)
        var_imp2 = gbm.varimp()
        
        Fin_imp_var = [[],var_imp2]
        return Fin_imp_var
        
        
    
         
    
    
