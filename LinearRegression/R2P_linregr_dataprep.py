import h2o
h2o.init()
from h2o.estimators.glm import H2OGeneralizedLinearEstimator
from h2o.estimators.gbm import H2OGradientBoostingEstimator

import numpy as np
import pandas as pd
import sklearn

from pandas.io.json import json_normalize
from scipy.stats import chi2_contingency
from scipy.stats import chi2
import itertools
from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import LabelEncoder

"""
 change names of the featrures
 set types of features: numeric or categorical
 input:
     df - data frame
     columnsArray - data frame with names and types of columns
 output:
     df - changed data frame
"""
def columns_data_type(df, columnsArray = ""):
    columnsArray = pd.DataFrame(columnsArray)
    df = df.dropna()
    df.columns = columnsArray['columnName'] 
    for i in columnsArray['columnName']:
        sub = columnsArray[columnsArray['columnName'] == i]
        if (sub['tableDisplayType'].values[0] == 'number'):
            df[i] = pd.to_numeric(df[i].values)
        if (sub['tableDisplayType'].values[0] == 'string'):
            df[i] = df[i].astype('category')
            
    return df

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

        for i in range(len(ind1)):
            
            try:
                test.append(chi2_contingency([list(cat_data[cat_data.columns[ind1[i]]]),list(cat_data[cat_data.columns[ind2[i]]])]))
            except:
                continue
        test = pd.DataFrame(test)
        test.columns = ['stat', 'p-val', 'dof', 'expected']
        return test    
    elif ncol == 1:
        print("There is only one category field exists")
    else:
        print("No category field exists")
        
    
"""
 correlations in the data
 Pearson test for numerical
 chi squared test for categorical
 input:
     df - data frame 
 output:
     corr matrix between numeric features
     chi squared method result for categorical features
"""

def correlations(data):
    cat_data = data.select_dtypes(include=['category']).copy()
    num_data = data.select_dtypes(include=['number']).copy()
    ent_cor = num_data.corr()
    chisq_dependency = ch_sq_test(cat_data)
    return ent_cor,chisq_dependency 

"""
 transformation method implements 
 YeoJohnson transformation for numeric features 
 (power transform featurewise to make data more Gaussian-like)
 input:
     df - data frame 
 output:
     transformed data frame
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
    return transformed_data     

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
        
        
    
         
    
    
