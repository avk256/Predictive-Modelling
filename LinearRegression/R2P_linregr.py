import sys
### change to the current script folder
sys.path.insert(0, 'E:\\Work\\aspirantura\\kafedra\\upwork\\RtoP\\PyModels\\LinearRegression')
import h2o
import pandas as pd
import json
import R2P_linregr_dataprep
import seaborn as sns
import matplotlib.pyplot as plt
from h2o.estimators.glm import H2OGeneralizedLinearEstimator
from h2o.estimators.gbm import H2OGradientBoostingEstimator
from statsmodels.stats.outliers_influence import variance_inflation_factor
import scipy
from R2P_linregr_dataprep import columns_data_type, ch_sq_test, correlations, transformation, variable_importance_h2o, transformation_inv 

############################################################################

############################################################################


"""
 Variance Inflation Factor (VIF) Explained method implements 
 variance_inflation_factor from statsmodels  
 
 input:
     daframe with Independent variables
     VIF threshold value
 output:
     List of multicollinear variables
"""
def vif(num_data, y, thresh):
    num_data = num_data.drop([y], axis=1)
    vif = pd.DataFrame()
    vif["VIF Factor"] = [variance_inflation_factor(num_data.values, i) for i in range(num_data.shape[1])]
    vif["features"] = num_data.columns
    return list(vif["features"][vif["VIF Factor"]>thresh])

    

"""
 Linear regression method implements 
 h2o.glm and h2o.gbm models  
 
 input:
    #* @param dbHost 
    #* @param dbPort 
    #* @param userName 
    #* @param password
    #* @param dbName 
    #* @param query
    #* @param yValue  e.g. response_col = 'target_var' 
    #* @param xValues e.g. predictors = ['Units Sold', 'Unit Price', 'Unit Cost', 'Total Revenue', 'Total Cost']
    #* @param parametersObj
    #* @param columnsArray
 output:
     R square value,
     Test data with predictions
     RMSE
     Model coefficients
     Correlation matrix
     list of variable importance
     list of multicollinear variables
"""
def linear_regression(dbHost="",dbPort="",userName="",password="",dbName="",query="",yValue="",xValues="",parametersObj="",columnsArray=""):
    ##############
    # connecting to BD
    ##############

    data = pd.read_csv("E:/Work/aspirantura/kafedra/upwork/RtoP/Models_sep/Datasets/50000_Sales_Records_Dataset_e.csv")
    data, columnsArray_e = columns_data_type(data[0:100], columnsArray)
    
    ent_cor,chisq_dependency,data,rm_cols, miss_cols, obj_t = correlations(data, columnsArray=columnsArray_e, method='predict')
    #print(ent_cor,chisq_dependency, rm_cols, miss_cols)
    
    num_data = data.select_dtypes(include=['number']).copy()
    variable_imp = variable_importance_h2o(data, list(num_data.columns), yValue)
    vif_var = vif(num_data, yValue, 10)
    #print(variable_imp, vif_var)
    num_data_inv = transformation_inv(num_data,obj_t)
    x_mean = num_data_inv[xValues].mean()
    x_min = num_data_inv[xValues].min()
    x_max = num_data_inv[xValues].max()
    x_quant25 = num_data_inv[xValues].quantile(0.25)
    x_quant50 = num_data_inv[xValues].quantile(0.5)
    x_quant75 = num_data_inv[xValues].quantile(0.75)
    x_skew = scipy.stats.skew(num_data_inv[xValues])

    
    if(data[yValue].dtypes == 'float') or (data[yValue].dtypes == 'int'):
        print("Finding variable importance by taking given numeric variable as a dependent variable")
        hf = h2o.H2OFrame(num_data)
        #hf.col_names
        train, valid, test = hf.split_frame(ratios=[.8, .1])
        
        glm_model = H2OGeneralizedLinearEstimator(family = 'gaussian')
        glm_model.train(xValues, yValue, training_frame= train, validation_frame=valid)
        print(glm_model)
        predicted = glm_model.predict(test_data=test)
        
        test_inv = transformation_inv(test.as_data_frame(),obj_t)
        true_y = test_inv[yValue]
        test[yValue] =  predicted
        test_inv = transformation_inv(test.as_data_frame(),obj_t)
        pred_y = test_inv[yValue]
        
        linear_regr = [pred_y, true_y ,glm_model.r2(),glm_model.rmse(),ent_cor,glm_model.coef(),variable_imp,vif_var,[x_mean,x_min,x_max,x_quant25,x_quant50,x_quant75,x_skew]]
        
    else:
        print("Finding variable importance by taking categorical variables as dependent variable")
        gbm = H2OGradientBoostingEstimator()
        gbm.train(xValues, yValue, training_frame= train, validation_frame=valid)
#        print(gbm)
        predicted2 = gbm.predict(test_data=test)

        test_inv = transformation_inv(test.as_data_frame(),obj_t)
        true_y = test_inv[yValue]
        test[yValue] =  predicted2
        test_inv = transformation_inv(test.as_data_frame(),obj_t)
        pred2_y = test_inv[yValue]


        linear_regr = [pred2_y, true_y ,glm_model.r2(),glm_model.rmse(),ent_cor,glm_model.coef(),variable_imp,vif_var]

    
    return linear_regr

"""
 Generate text description of the data and model 
 
 input:
     linear_regr - data structure with results of linear regression model
 output:
     lists with text description 
"""

def NLG(linear_regr, yValue="",xValues=""):
    # correlated variables
    corr = linear_regr[4]
    corr = corr.drop(yValue)
    mask1 = corr[yValue] >= 0.3
    mask2 = corr[yValue] <= -0.2
    mask3 = (corr[yValue] >= 0.2) & (corr[yValue] <= 0.3)
    corr_val1 = "<li class=\"nlg-item\"><span class=\"nlgtext\">There is a mutual relationship between <strong>" + (corr[yValue][mask1].index) + "</strong><strong>" + yValue + "</strong>, With every unit of increase in"+ (corr[yValue][mask1].index) + " there is an increase in " + yValue + "</span></li>"
    corr_val2 = "<li class=\"nlg-item\"><span class=\"nlgtext\">There is a mutual relationship between<strong> " + (corr[yValue][mask2].index) + "</strong><strong>" + yValue + "</strong></span>, With every unit of decrease in "+ (corr[yValue][mask2].index) + " there is an increase in " + yValue + "</span></li>"
    corr_val3 = "<li class=\"nlg-item\"><span class=\"nlgtext\">There is a little or no relationship between <strong> " + (corr[yValue][mask3].index) + "</strong> and <strong>" + yValue + "</strong></span></li>"
    # quantile and min, max values
    #linear_regr[8] : [x_mean,x_min,x_max,x_quant25,x_quant50,x_quant75,x_skew]    
    var_val = "<li class=\"nlg-item\"><span class=\"nlgtext\">The average value of <strong>"+str(xValues)+"</strong> is <strong>"+str(list(linear_regr[8][0]))+"</strong></span>","</li>"
    quantile_val = "<li class=\"nlg-item\"><span class=\"nlgtext\">The minimum value of <strong>"+str(xValues)+"</strong> is <strong>" +str(list(linear_regr[8][1]))+"</strong> whereas <strong>25%</strong> of data lies below the value <strong>"+str(list(linear_regr[8][3]))+ "</strong> the median of <strong>"+str(xValues)+ "</strong> is <strong>"+str(list(linear_regr[8][4]))+"</strong> and the <strong>75%</strong> of data lies below the value <strong>"+str(list(linear_regr[8][5]))+"</strong> and the max value is <strong>"+str(list(linear_regr[8][2]))+"</strong></span></li>"
    
    return [corr_val1, corr_val2, corr_val3, var_val, quantile_val] 
    
    
"""
 Plotting method 
 
 input:
     linear_regr - data structure with results of linear regression model
 output:
     plots: R^2, Predicted-Actual Value, Variable importance
"""

def plotting(linear_regr):
 
    pred_val = linear_regr[0]
    act_val = linear_regr[1]
    
    r2 = linear_regr[2]
    
    var_imp = pd.DataFrame(linear_regr[6][0])
    var_imp.columns = ['variable', 'relative_importance', 'scaled_importance', 'percentage']
    
    sns.distplot(var_imp['relative_importance'],kde = False, axlabel = list(var_imp['variable']))
    fig=plt.figure()
    ax1=fig.add_subplot(1,1,1)
    ax1.scatter(pred_val.values,act_val.values)
    ax1.set_title('Scatterplot')
    ax1.set_xlabel('Predicted value')
    ax1.set_ylabel('Actual value')
    print("Linear regression accuracy: R^2")
    print(r2)


    
############## testing linear regression

### example of columnsArray

columnsArray = '[{"columnDisplayName":"S@#@#no","tableDisplayType":"string","columnName":"S#@#@no"},{"columnDisplayName":"Region","tableDisplayType":"string","columnName":"Region"},{"columnDisplayName":"Country","tableDisplayType":"string","columnName":"Country"},{"columnDisplayName":"Item Type","tableDisplayType":"string","columnName":"Item Type"},{"columnDisplayName":"Sales Channel","tableDisplayType":"string","columnName":"Sales Channel"},{"columnDisplayName":"Order Priority","tableDisplayType":"string","columnName":"Order Priority"},{"columnDisplayName":"Order Date","tableDisplayType":"string","columnName":"Order Date"},{"columnDisplayName":"Order ID","tableDisplayType":"string","columnName":"Order ID"},{"columnDisplayName":"Ship Date","tableDisplayType":"string","columnName":"Ship Date"},{"columnDisplayName":"Units Sold","tableDisplayType":"number","columnName":"Units Sold"},{"columnDisplayName":"Unit Price","tableDisplayType":"number","columnName":"Unit Price"},{"columnDisplayName":"Unit Cost","tableDisplayType":"number","columnName":"Unit Cost"},{"columnDisplayName":"Total Revenue","tableDisplayType":"number","columnName":"Total Revenue"},{"columnDisplayName":"Total Cost","tableDisplayType":"number","columnName":"Total Cost"},{"columnDisplayName":"target_var","tableDisplayType":"number","columnName":"target_var"}]'    
#columnsArray = '[{"columnDisplayName":"Region","tableDisplayType":"string","columnName":"Region"},{"columnDisplayName":"Country","tableDisplayType":"string","columnName":"Country"},{"columnDisplayName":"Item Type","tableDisplayType":"string","columnName":"Item Type"},{"columnDisplayName":"Sales Channel","tableDisplayType":"string","columnName":"Sales Channel"},{"columnDisplayName":"Order Priority","tableDisplayType":"string","columnName":"Order Priority"},{"columnDisplayName":"Order Date","tableDisplayType":"string","columnName":"Order Date"},{"columnDisplayName":"Order ID","tableDisplayType":"string","columnName":"Order ID"},{"columnDisplayName":"Ship Date","tableDisplayType":"string","columnName":"Ship Date"},{"columnDisplayName":"Units Sold","tableDisplayType":"number","columnName":"Units Sold"},{"columnDisplayName":"Unit Price","tableDisplayType":"number","columnName":"Unit Price"},{"columnDisplayName":"Unit Cost","tableDisplayType":"number","columnName":"Unit Cost"},{"columnDisplayName":"Total Revenue","tableDisplayType":"number","columnName":"Total Revenue"},{"columnDisplayName":"Total Cost","tableDisplayType":"number","columnName":"Total Cost"},{"columnDisplayName":"target_var","tableDisplayType":"number","columnName":"target_var"}]'    
try:
    columnsArray = json.loads(columnsArray)
except:
    print('json format not valid')



linear_regr = linear_regression(yValue='Total Revenue',xValues=['Units Sold', 'Total Cost'],columnsArray=columnsArray)

NLG(linear_regr, yValue='Total Revenue',xValues=['Units Sold', 'Total Cost'])

plotting(linear_regr)

