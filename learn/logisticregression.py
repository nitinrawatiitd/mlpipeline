import pandas as pd
import numpy as np
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error
from sklearn import linear_model
from sklearn.externals import joblib
import time

'''
Load data and declare parameters
Improvements
1. Load all kinds of data - csv, pickle, text extract
2. Declare parameters in a seperate main.py file where the ml modules are called
3. Declare the parameters in CLI way
'''

# declaring parameters and inputs
data = pd.read_csv('../data/'+'train_preprocessed.csv')
feat_groups = 'RAW'
tuned_parameters = [{'alpha':[0.0005,0.001,0.005,0.01,0.1,0.5]}]
dv_col = 'SalePrice'
sample_col = 'sample'
scores=['r2']
'''
# model evaluation options - http://scikit-learn.org/stable/modules/model_evaluation.html#implementing-your-own-scoring-object
#scores for regression
['explained_variance','neg_mean_absolute_error','neg_mean_squared_error','neg_mean_squared_log_error','neg_median_absolute_error','r2']
# scores for classification
['accuracy','average_precision','f1','f1_micro','f1_macro','f1_weighted','f1_samples','neg_log_loss','precision','recall','roc_auc']
'''
# Decorator function for calculating time it takes for a function to run
def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        print 'func:%r args:[%r, %r] took: %2.4f sec' % \
          (method.__name__, args, kw, te-ts)
        return result
    return timed

@timeit
def lr(data,feat_groups,dv_col,sample_col,tuned_parameters,scores):

    # Find the columns with prefixes as per the feat_groups
    feat_groups = feat_groups.split(',')
    feature_names = set()
    for i in range(len(feat_groups)):
        group = feat_groups[i]
        group_columns = data.filter(regex='^'+group).columns
        feature_names.update(group_columns)

    # Break the dataset into dev, val and X and Y parts
    X_dev = data[data[sample_col]=='dev'][list(feature_names)]
    Y_dev = data[data[sample_col]=='dev'][dv_col]
    X_val = data[data[sample_col]=='val'][list(feature_names)]
    Y_val = data[data[sample_col]=='val'][dv_col]

    # Run grid search for the score you want to optimise for
    for score in scores:
        print("# Tuning hyper-parameters for %s" % score)
        lasso = linear_model.Lasso()
        clf = GridSearchCV(lasso, tuned_parameters, cv=4, scoring=score,n_jobs=4,verbose=1)
        #print("The model is trained on the full development set.")
        clf.fit(X_dev, Y_dev)
        df_scores = pd.DataFrame(columns=['param','mean_score','std_score'])
        for i, grid_params in enumerate(clf.grid_scores_):
            params, mean_score, scores = grid_params.parameters, grid_params.mean_validation_score, grid_params.cv_validation_scores
            print("%0.3f (+/-%0.03f) for %r"% (mean_score, scores.std() / 2, params))
            # save the results in a csv file
            df_scores.loc[i] = [params,mean_score,scores.std() / 2]

        df_scores.to_csv('scores'+str(score)+'.csv',index=False)

        print("The scores are computed on the full evaluation set.")
        y_true_dev, y_pred_dev = Y_dev, clf.predict(X_dev)
        y_true_val, y_pred_val = Y_val, clf.predict(X_val)
        print 'dev R2 :',r2_score(y_true_dev, y_pred_dev),'val R2 :',r2_score(y_true_val, y_pred_val)
        print 'dev RMSE :',mean_squared_error(y_true_dev, y_pred_dev)**0.5,'val RMSE :',mean_squared_error(y_true_val, y_pred_val)**0.5
