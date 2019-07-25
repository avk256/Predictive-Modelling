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
from h2o.estimators.kmeans import H2OKMeansEstimator
from h2o.estimators.pca import H2OPrincipalComponentAnalysisEstimator
from statsmodels.stats.outliers_influence import variance_inflation_factor
import scipy
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score


from R2P_linregr_dataprep import columns_data_type, ch_sq_test, correlations, transformation, variable_importance_h2o, transformation_inv 

############################################################

############################################################

"""
 k-means model for clustering problem 
 
 input:
     df - preprocessed data frame
 output:
     cluster column
     principle components columns
     principle components metrics
     centroid
     inter cluster similarity value for each cluster
     intra cluster similarity value
     size of each cluster

"""
def kmeans_model(df, xValues):
    
    hf = h2o.H2OFrame(df)
    train, valid, test = hf.split_frame(ratios=[.8, .1])    
    # kmeans model
    kmeans = H2OKMeansEstimator(k=3,max_iterations=5,seed = 10,categorical_encoding = "AUTO",max_runtime_secs=10)
    kmeans.train(xValues, training_frame= hf)
    
    # pca model, generate Principal Components for further modelling or plotting
    
    pca = H2OPrincipalComponentAnalysisEstimator(k=4)
#    pca.train(xValues, training_frame= hf)
    pca.train(list(df.columns), training_frame= hf)
    pca_features = pca.predict(hf).as_data_frame()
    pca_metric = pca.summary().as_data_frame()
    
    # model metrics
    cluster_column = kmeans.predict(hf).as_data_frame()
    # The Between Cluster Sum-of-Square Error
    inter_cluster_error = kmeans.betweenss()
    # Within Cluster Sum-of-Square Error
    intra_cluster_error = kmeans.withinss()
    # Centroids
    centroids = kmeans.centers()
    # Size of clusters
    cluster_size = kmeans.size()
# 
    cluster_column.columns = ['cluster']
    frames = [df,cluster_column]
    transformed_data = pd.concat(frames, axis=1)
    
    output = [transformed_data, pca_features, pca_metric, centroids, inter_cluster_error, intra_cluster_error, cluster_size]
    return output

"""
 Cluster profiling method 
 
 input:
     data - data frame after kmeans_model()
     xValues  
 output:
     important variables based on cluster column
     cluster wise mean and median
     cluster wise mode
"""
def cluster_profiling(data, xValues):
    ######### cluster_profiling
    
    # variable importance for cluster - categorical feature
    variable_imp = variable_importance_h2o(data, xValues, 'cluster')[1]
    variable_imp = pd.DataFrame(variable_imp)
    variable_imp.columns = ['variable', 'relative_importance', 'scaled_importance', 'percentage']
    
    ### cluster wise mean and median for numeric data, mode for categorical
    
    clusters = list(data['cluster'].unique())
    
    cat_data = data.select_dtypes(include=['category']).copy()
    cat_data_cols = list(cat_data.columns)
    cat_data = pd.concat([cat_data,data['cluster']], axis=1)
    num_data = data.select_dtypes(include=['number']).copy()
    num_data_cols = list(num_data.drop(['cluster'],axis=1).columns)
    
    mean_clust = pd.DataFrame(columns=num_data_cols, index=clusters)
    median_clust = pd.DataFrame(columns=num_data_cols, index=clusters)
    mode_clust = pd.DataFrame(columns=cat_data_cols, index=clusters)
    
    for i in clusters:
        for j in num_data_cols:
            mean_clust[j].iloc[i] = num_data[num_data['cluster'] == i][j].mean()
            median_clust[j].iloc[i] = num_data[num_data['cluster'] == i][j].median()
    
    for i in clusters:
        for j in cat_data_cols:
            mode_clust[j].iloc[i] = cat_data[cat_data['cluster'] == i][j].mode()[0]
            
    output = [variable_imp, mean_clust, median_clust, mode_clust]
    
    return output


"""
 k-means method for clustering problem 
 
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
     cluster column
     principle components
     centroid
     inter cluster similarity value for each cluster
     intra cluster similarity value
     size of each cluster

"""

def clustering(dbHost="",dbPort="",userName="",password="",dbName="",query="",xValues="",parametersObj="",columnsArray=""):
    ##############
    # connecting to BD
    ##############

#    data = pd.read_csv("E:/Work/aspirantura/kafedra/upwork/RtoP/Models_sep/Datasets/50000_Sales_Records_Dataset_e.csv")
#    data = pd.read_excel("E:/Work/aspirantura/kafedra/upwork/RtoP/Models_sep/Datasets/quine.xlsx")
#    data = pd.read_excel("E:/Work/aspirantura/kafedra/upwork/RtoP/Models_sep/Datasets/airfares.xlsx")
    data = pd.read_csv("E:/Work/aspirantura/kafedra/upwork/RtoP/Models_sep/Datasets/AAPL.csv")
    data, columnsArray_e = columns_data_type(data[0:100], columnsArray)
    ent_cor,chisq_dependency,data,rm_cols, miss_cols, obj_t = correlations(data, columnsArray=columnsArray_e, method='predict', no_rem_col='none')
    print('Data info before clustering')
    print(data.info())
    
    output = kmeans_model(data[xValues],xValues)
   
    return output


"""
 Description of the clusters similarity
 based on the inter and intra metrics 
 input:
     inter_cluster_error, 
     intra_cluster_error  
 output:
     list of text descriptions
"""
def cluster_nlg(inter_cluster_error, intra_cluster_error):
    
    min_withiness = intra_cluster_error.index(min(intra_cluster_error))    
    withiness_desc = "<li class=\"nlg-item\"><span class=\"nlgtext\"> <strong> Cluster"+str(min_withiness)+"</strong> has lowest value of withiness. The points within the <strong> Cluster"+str(min_withiness)+"</strong> are more homogenous </span></li>"
    betweeness = inter_cluster_error
    if(betweeness>=10000):
        betweeness_desc = "<li class=\"nlg-item\"><span class=\"nlgtext\"> Clusters have reasonably high value of betweeness  <strong>"+str(round(betweeness))+"  </strong> indicating that the heterogeneity among them is high </span></li>"
    elif (betweeness>5000 and betweeness<10000):
        betweeness_desc = "<li class=\"nlg-item\"><span class=\"nlgtext\"> Clusters have medium level of betweeness  <strong>"+str(round(betweeness))+" </strong>indicating that the heterogeneity among them is not very high </span></li>"
    elif (betweeness<5000): 
        betweeness_desc = "<li class=\"nlg-item\"><span class=\"nlgtext\"> Clusters have low level of betweeness <strong>"+str(round(betweeness))+"  </strong> indicating that the heterogeneity among them is low </span></li>"
    
    return [withiness_desc, betweeness_desc]

"""
 Description of the statistics by clusters  
 input:
     variable_imp - important variables based on cluster column
     mean_clust, median_clust - cluster wise mean and median
     mode_clust output - cluster wise mode
 output:
     list of text descriptions
"""
def profiling_nlg(variable_imp, mean_clust, median_clust, mode_clust):
    max_mean = []
    min_mean = []
    freq_info = []
    if len(list(mean_clust.columns))>0:
        for i in list(mean_clust.columns):
            
            max_mean.append("<li class=\"nlg-item\"><span class=\"nlgtext\"> The mean value of  <strong> "+ str(i) +"</strong> in <strong> cluster"+str(list(mean_clust[i]).index(max(list(mean_clust[i]))))+"</strong> is highest across other <strong>"+str(len(cluster_prof[1].index)-1)+"</strong> clusters"+"</span></li>")
            min_mean.append("<li class=\"nlg-item\"><span class=\"nlgtext\"> The mean value of  <strong>"+ str(i)+ "</strong> in <strong> cluster"+str(list(mean_clust[i]).index(min(list(mean_clust[i]))))+"</strong> is the lowest across other <strong>"+str(len(cluster_prof[1].index)-1)+"</strong> clusters"+"</span></li>")
    
    if len(list(mode_clust.columns))>0:
        for i in list(mode_clust.columns):
            for j in list(mode_clust.index): 
                freq_info.append("<li class=\"nlg-item\"><span class=\"nlgtext\"> The frequency of <strong>"+ str(i)+"</strong> - <strong> "+str(mode_clust[i].iloc[j])+" </strong> is the most repeated level in <strong> Cluster "+str(j)+"</strong></span></li>")
    
    return [max_mean, min_mean, freq_info]
    
############## testing clustering

### example of columnsArray

columnsArray = '[{"columnDisplayName":"S@#@#no","tableDisplayType":"string","columnName":"S#@#@no"},{"columnDisplayName":"Region","tableDisplayType":"string","columnName":"Region"},{"columnDisplayName":"Country","tableDisplayType":"string","columnName":"Country"},{"columnDisplayName":"Item Type","tableDisplayType":"string","columnName":"Item Type"},{"columnDisplayName":"Sales Channel","tableDisplayType":"string","columnName":"Sales Channel"},{"columnDisplayName":"Order Priority","tableDisplayType":"string","columnName":"Order Priority"},{"columnDisplayName":"Order Date","tableDisplayType":"string","columnName":"Order Date"},{"columnDisplayName":"Order ID","tableDisplayType":"string","columnName":"Order ID"},{"columnDisplayName":"Ship Date","tableDisplayType":"string","columnName":"Ship Date"},{"columnDisplayName":"Units Sold","tableDisplayType":"number","columnName":"Units Sold"},{"columnDisplayName":"Unit Price","tableDisplayType":"number","columnName":"Unit Price"},{"columnDisplayName":"Unit Cost","tableDisplayType":"number","columnName":"Unit Cost"},{"columnDisplayName":"Total Revenue","tableDisplayType":"number","columnName":"Total Revenue"},{"columnDisplayName":"Total Cost","tableDisplayType":"number","columnName":"Total Cost"},{"columnDisplayName":"target_var","tableDisplayType":"number","columnName":"target_var"}]'    
columnsArray_quine = '[{"columnDisplayName":"Id","tableDisplayType":"number","columnName":"Id"},{"columnDisplayName":"Days","tableDisplayType":"number","columnName":"Days"},{"columnDisplayName":"Age","tableDisplayType":"number","columnName":"Age"},{"columnDisplayName":"Sex","tableDisplayType":"number","columnName":"Sex"},{"columnDisplayName":"Eth","tableDisplayType":"number","columnName":"Eth"},{"columnDisplayName":"Lrn","tableDisplayType":"number","columnName":"Lrn"}]'    
columnsArray_airfares = '[{"columnDisplayName":"COUPON","tableDisplayType":"number","columnName":"COUPON"},{"columnDisplayName":"NEW","tableDisplayType":"number","columnName":"NEW"},{"columnDisplayName":"HI","tableDisplayType":"number","columnName":"HI"},{"columnDisplayName":"S_INCOME","tableDisplayType":"number","columnName":"S_INCOME"},{"columnDisplayName":"E_INCOME","tableDisplayType":"number","columnName":"E_INCOME"},{"columnDisplayName":"S_POP","tableDisplayType":"number","columnName":"S_POP"},{"columnDisplayName":"E_POP","tableDisplayType":"number","columnName":"E_POP"},{"columnDisplayName":"DISTANCE","tableDisplayType":"number","columnName":"DISTANCE"},{"columnDisplayName":"PAX","tableDisplayType":"number","columnName":"PAX"},{"columnDisplayName":"FARE","tableDisplayType":"number","columnName":"FARE"}]'    
columnsArray_aapl = '[{"columnDisplayName":"date","tableDisplayType":"string","columnName":"date"},{"columnDisplayName":"open","tableDisplayType":"number","columnName":"open"},{"columnDisplayName":"high","tableDisplayType":"number","columnName":"high"},{"columnDisplayName":"low","tableDisplayType":"number","columnName":"low"},{"columnDisplayName":"close","tableDisplayType":"number","columnName":"close"},{"columnDisplayName":"volume","tableDisplayType":"number","columnName":"volume"},{"columnDisplayName":"unadjustedVolume","tableDisplayType":"number","columnName":"unadjustedVolume"},{"columnDisplayName":"change","tableDisplayType":"number","columnName":"change"},{"columnDisplayName":"changePercent","tableDisplayType":"number","columnName":"changePercent"},{"columnDisplayName":"vwap","tableDisplayType":"number","columnName":"vwap"},{"columnDisplayName":"label","tableDisplayType":"string","columnName":"label"},{"columnDisplayName":"changeOverTime","tableDisplayType":"number","columnName":"changeOverTime"}]'    

try:
    columnsArray = json.loads(columnsArray_aapl)
except:
    print('json format not valid')
   
cluster_model = clustering(xValues=['open', 'high', 'low', 'close'],columnsArray=columnsArray)

cluster_prof = cluster_profiling(cluster_model[0], xValues=['open', 'high', 'low', 'close'])

    
cluster_nlg(cluster_model[4], cluster_model[5])

profiling_nlg(cluster_prof[0],cluster_prof[1],cluster_prof[2],cluster_prof[3])




