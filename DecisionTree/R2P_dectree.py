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
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score


from R2P_linregr_dataprep import columns_data_type, ch_sq_test, correlations, transformation, variable_importance_h2o, transformation_inv 

############################################################

############################################################

"""
 Recursive method of walking through nodes 
 
 input:
     node - id of H2oTree node 
 output:
     lists of dictionaries with node description 
"""
def go2node(node):
    pass

"""
 Convert H2oTree object to json 
 
 input:
     h2otree - H2oTree object 
 output:
     tree in json 
"""
def tree2json(h2otree):
    pass
#    len(tH2oTree)
#    print(tH2oTree)
#    len(tH2oTree)
#    tH2oTree.node_ids
#    tH2oTree.root_node.left_child



"""
 Decision tree method for classification problem 
 
 input:
     h2otree - H2oTree object 
 output:
      
"""
def decisiontree(dbHost="",dbPort="",userName="",password="",dbName="",query="",yValue="",xValues="",parametersObj="",columnsArray=""):
    ##############
    # connecting to BD
    ##############

    data = pd.read_csv("E:/Work/aspirantura/kafedra/upwork/RtoP/Models_sep/Datasets/50000_Sales_Records_Dataset_e.csv")
    data, columnsArray_e = columns_data_type(data[0:100], columnsArray)
    ent_cor,chisq_dependency,data,rm_cols, miss_cols, obj_t = correlations(data, columnsArray=columnsArray_e)
    
    feature = xValues
    feature.append(yValue) 
    
    df = data[feature]
    
    hf = h2o.H2OFrame(df)
    
    train, valid, test = hf.split_frame(ratios=[.8, .1])    
    # GBM model
    gbm = H2OGradientBoostingEstimator()
    gbm.train(xValues, yValue, training_frame= train, validation_frame=valid)
    var_imp = pd.DataFrame(gbm.varimp())
    var_imp.columns = ['variable','relative_importance','scaled_importance','percentage']
    # metrics
    variable_importance = var_imp[var_imp['relative_importance']>=0.1]
    variable_importance = variable_importance['variable']
    gbm_cm = gbm.confusion_matrix()
    gbm_cm = gbm_cm.table.as_data_frame()
    gbm_prec = gbm.precision()[0][0]
    gbm_f1 = gbm.F1()[0][0]
    gbm_acc = gbm.accuracy()[0][0]
    gbm_rec = gbm_f1*gbm_prec/(2*gbm_prec-gbm_f1)
    # Tree info
    from h2o.tree import H2OTree
    tree = H2OTree(model = gbm, tree_number = 1, tree_class = None)
    nlg_tree = tree.descriptions
    tree_json = tree2json(tree)
    
    output = [tree_json, gbm_prec, gbm_rec, gbm_acc, gbm_cm, variable_importance, nlg_tree] 
    return output

    
############## testing linear regression

### example of columnsArray

columnsArray = '[{"columnDisplayName":"S@#@#no","tableDisplayType":"string","columnName":"S#@#@no"},{"columnDisplayName":"Region","tableDisplayType":"string","columnName":"Region"},{"columnDisplayName":"Country","tableDisplayType":"string","columnName":"Country"},{"columnDisplayName":"Item Type","tableDisplayType":"string","columnName":"Item Type"},{"columnDisplayName":"Sales Channel","tableDisplayType":"string","columnName":"Sales Channel"},{"columnDisplayName":"Order Priority","tableDisplayType":"string","columnName":"Order Priority"},{"columnDisplayName":"Order Date","tableDisplayType":"string","columnName":"Order Date"},{"columnDisplayName":"Order ID","tableDisplayType":"string","columnName":"Order ID"},{"columnDisplayName":"Ship Date","tableDisplayType":"string","columnName":"Ship Date"},{"columnDisplayName":"Units Sold","tableDisplayType":"number","columnName":"Units Sold"},{"columnDisplayName":"Unit Price","tableDisplayType":"number","columnName":"Unit Price"},{"columnDisplayName":"Unit Cost","tableDisplayType":"number","columnName":"Unit Cost"},{"columnDisplayName":"Total Revenue","tableDisplayType":"number","columnName":"Total Revenue"},{"columnDisplayName":"Total Cost","tableDisplayType":"number","columnName":"Total Cost"},{"columnDisplayName":"target_var","tableDisplayType":"number","columnName":"target_var"}]'    
#columnsArray = '[{"columnDisplayName":"Region","tableDisplayType":"string","columnName":"Region"},{"columnDisplayName":"Country","tableDisplayType":"string","columnName":"Country"},{"columnDisplayName":"Item Type","tableDisplayType":"string","columnName":"Item Type"},{"columnDisplayName":"Sales Channel","tableDisplayType":"string","columnName":"Sales Channel"},{"columnDisplayName":"Order Priority","tableDisplayType":"string","columnName":"Order Priority"},{"columnDisplayName":"Order Date","tableDisplayType":"string","columnName":"Order Date"},{"columnDisplayName":"Order ID","tableDisplayType":"string","columnName":"Order ID"},{"columnDisplayName":"Ship Date","tableDisplayType":"string","columnName":"Ship Date"},{"columnDisplayName":"Units Sold","tableDisplayType":"number","columnName":"Units Sold"},{"columnDisplayName":"Unit Price","tableDisplayType":"number","columnName":"Unit Price"},{"columnDisplayName":"Unit Cost","tableDisplayType":"number","columnName":"Unit Cost"},{"columnDisplayName":"Total Revenue","tableDisplayType":"number","columnName":"Total Revenue"},{"columnDisplayName":"Total Cost","tableDisplayType":"number","columnName":"Total Cost"},{"columnDisplayName":"target_var","tableDisplayType":"number","columnName":"target_var"}]'    
try:
    columnsArray = json.loads(columnsArray)
except:
    print('json format not valid')

   
tree = decisiontree(yValue='Sales Channel',xValues=['Units Sold','Unit Price','Total Revenue'],columnsArray=columnsArray)

tree[6]
    