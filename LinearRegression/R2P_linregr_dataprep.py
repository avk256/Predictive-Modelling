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
####################################################################

####################################################################


"""
 Changing names of the featrures.
 Set types of features: numeric or categorical.
 Replacing special characters in column names.
 input:
     df - data frame
     columnsArray - data frame with names and types of columns
 output:
     df - changed data frame
"""
def columns_data_type(df, columnsArray = ""):
    columnsArray = pd.DataFrame(columnsArray)
    columnsArray = columnsArray.replace(r'[^a-zA-Z0-9 -]', "", regex=True)
    df.columns = columnsArray['columnName'] 
    df = df.replace(r'^\s*$', np.nan, regex=True)

    for i in columnsArray['columnName']:
        sub = columnsArray[columnsArray['columnName'] == i]
        if (sub['tableDisplayType'].values[0] == 'number'):
            df[i] = pd.to_numeric(df[i].values)
        if (sub['tableDisplayType'].values[0] == 'string'):
            #df[i] = str(df[i])
            df[i] = df[i].astype('category')
            
    return df

"""
 Imputation or removing of missing values using mean an mode.
 Replacing blanks with NA's.
 input:
     df - data frame
     method
         drop - drop all NA's
         mean - mean for numeric and 
         mode - for categorical
 output:
     df - changed data frame
"""
def missing_val_impute(df, method):
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
                    
        else:
            print("Imputation method not specify")        
        frames = [cat_data,num_data]
        df = pd.concat(frames, axis=1)
    except:
        print("Imputation method doesn't meet the data")
        df = df.dropna()
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
 output:
     corr matrix between numeric features
     chi squared method result for categorical features
     preprocessed data
     list of exclude columns
     list of missing values amount for each columns 
"""
def correlations(data):
    # data type conversion and deleting missing values 
    data, miss_cols = missing_val_impute(data, method='impute')
    data, rm_cols = remove_col(data, ratio=3)
    data, obj_t = transformation(data)
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
        gbm = H2OGradientBoostingEstimator()
        gbm.train(predictors, response_col, training_frame= train, validation_frame=valid)
#        print(gbm)
        var_imp2 = gbm.varimp()
        Fin_imp_var = [var_imp2]
        return Fin_imp_var
        
        
    
         
    
    
