import sys
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
#%matplotlib inline

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import BayesianRidge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import LassoCV
from sklearn.neural_network import MLPRegressor
import xgboost
from sklearn.model_selection import train_test_split

from  aux_functions import *
from elapsed_time import *
from feature_selection import *
from plots import *

start_time = time.time()


#Comparing some 'simple' linear models, linear regression
#Ridge regression and Lasso to see what effect regularisation has
#on the final solution of these techniques
lr = LinearRegression()
br = BayesianRidge(alpha_1=1e-06, alpha_2=1e-06, compute_score=False,
        copy_X=True, fit_intercept=True, lambda_1=1e-06, lambda_2=1e-06,
        n_iter=300, normalize=False, tol=0.001, verbose=False)
rfr = RandomForestRegressor(n_estimators=20, n_jobs=-1)
rng = np.random.RandomState(1)
bdt = AdaBoostRegressor(DecisionTreeRegressor(max_depth=4), n_estimators=10, random_state=rng)
nn = MLPRegressor(hidden_layer_sizes=(100, ), activation='relu', solver='adam', alpha=0.0001, batch_size='auto', learning_rate='constant', learning_rate_init=0.001, power_t=0.5, max_iter=200, shuffle=True, random_state=None, tol=0.0001, verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True, early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08)


frame = pd.read_csv("./data/train.csv")
frame_test = pd.read_csv("./data/test.csv")

grpdict = group_by_attr_type(frame)
categorical = grpdict['object']

#explore_data(frame)
data_train, data_test, categorical_feats, num_feats, target_data = prepare_data(frame, frame_test, categorical)

all_feats = data_train.columns.values

categorical_feats_frame = data_train[categorical_feats]
selected_features, coef = feature_importance(categorical_feats_frame, target_data)
final_feats = num_feats + selected_features

#splitting train data in test and train subsets
X,x,Y,y = train_test_split( data_train ,target_data ,test_size = 0.3 ,random_state = 23 )

trainlr, testlr, testlr_abs, lr = feattest(lr, final_feats, X, x, Y, y)

lastFeat = np.argmin(testlr)
finalFeats = final_feats[0:lastFeat]
minscrs = min(testlr)


#setting final frame to feed azure and scikit
final_frame = data_train[finalFeats]
final_frame['SalePrice'] = target_data
final_frame.to_csv("./data/finalframe.csv")


#submiting kaggle file
lr.fit(data_train[finalFeats], target_data)
predlr = lr.predict(data_test[finalFeats])

file = 'lr_15012017_1.csv'
SUBMISSION_FILE = './data/sample_submission.csv'
submission = pd.read_csv(SUBMISSION_FILE)
submission.iloc[:, 1] = predlr
saleprice = np.exp(submission['SalePrice'])-1
submission['SalePrice'] = saleprice
submission.to_csv("./data/"+file, index=None)

