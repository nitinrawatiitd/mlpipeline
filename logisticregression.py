import pandas as pd
import numpy as np
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error
from sklearn import linear_model
from sklearn.externals import joblib
root = '/Users/nrawat/Documents/customer_value_modelling/'
scaler = joblib.load(root+'feature_selection/DV_scaler.pkl')


# In[2]:


data = pd.read_pickle(root+'feature_selection/feats_all_LI_dv_ALL_inv_capped_scaled_preprocessed_c50.pkl')


# In[3]:


feat_groups = 'months,MI,YLB,PAH,YL,PPB,PP,EA,RV,YAH,CR,KH,FICOCLV8,SC_FICOScore'
feat_groups = feat_groups.split(',')


# In[4]:


tuned_parameters = [{'alpha':[0.0005,0.001,0.005,0.01,0.1,0.5]}]
scores=['r2']


# In[5]:


feature_names = set()
for group in feat_groups:
    group_columns = data.filter(regex='^'+group).columns
    feature_names.update(group_columns)


# In[6]:


dv = 'DV_rev_inv'
X_dev = data[data['sample']=='dev'][list(feature_names)]
Y_dev = data[data['sample']=='dev'][dv]
X_val = data[data['sample']=='val'][list(feature_names)]
Y_val = data[data['sample']=='val'][dv]


# In[ ]:


import time
start_time = time.time()
for score in scores:
    print("# Tuning hyper-parameters for %s" % score)
    lasso = linear_model.Lasso()
    clf = GridSearchCV(lasso, tuned_parameters, cv=4, scoring=score,n_jobs=4,verbose=1)
    print("The model is trained on the full development set.")
    clf.fit(X_dev, Y_dev)
    for params, mean_score, scores in clf.grid_scores_:
        print("%0.3f (+/-%0.03f) for %r"% (mean_score, scores.std() / 2, params))

    print("The scores are computed on the full evaluation set.")
    y_true_dev, y_pred_dev = Y_dev, clf.predict(X_dev)
    y_true_val, y_pred_val = Y_val, clf.predict(X_val)
    print 'dev R2 :',r2_score(y_true_dev, y_pred_dev),'val R2 :',r2_score(y_true_val, y_pred_val)
    print 'dev RMSE :',mean_squared_error(y_true_dev, y_pred_dev)**0.5,'val RMSE :',mean_squared_error(y_true_val, y_pred_val)**0.5

print 'time taken : ', time.time() - start_time
