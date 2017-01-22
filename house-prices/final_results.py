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


def feattest(model, Xtrain, xtest, Ytrain, ytest):
    """
    Testing the features one by one and scoring the
    models on the test and training set.

    """
    
    X_t = Xtrain
    x_t = xtest
    model.fit(X_t,Ytrain)
    trainscrs = np.sqrt(-cross_val_score(model, X_t, Ytrain, scoring="neg_mean_squared_error", cv = 5)).mean()              
    testscrs = np.sqrt(mean_squared_error(ytest, model.predict(x_t)))
    testscrs_abs =mean_absolute_error(ytest, model.predict(x_t))

    return trainscrs, testscrs, testscrs_abs, model


#Comparing some 'simple' linear models, linear regression
#Ridge regression and Lasso to see what effect regularisation has
#on the final solution of these techniques
lr = LinearRegression()
br = BayesianRidge(alpha_1=1e-06, alpha_2=1e-06, compute_score=False,
        copy_X=True, fit_intercept=True, lambda_1=1e-06, lambda_2=1e-06,
        n_iter=300, normalize=False, tol=0.001, verbose=False)
rfr = RandomForestRegressor(n_estimators=100, n_jobs=-1)
rng = np.random.RandomState(1)
bdt = AdaBoostRegressor(DecisionTreeRegressor(max_depth=4), n_estimators=300, random_state=rng)
nn = MLPRegressor(hidden_layer_sizes=(100, ), activation='relu', solver='adam', alpha=0.0001, batch_size='auto', learning_rate='constant', learning_rate_init=0.001, power_t=0.5, max_iter=200, shuffle=True, random_state=None, tol=0.0001, verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True, early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08)


frame = pd.read_csv("./data/finalframe.csv")

TARGET = 'SalePrice'
target_data = frame[TARGET]
frame.drop([TARGET], axis=1, inplace=True)


#splitting train data in test and train subsets
X,x,Y,y = train_test_split(frame, target_data ,test_size = 0.3 ,random_state = 23 )

trainlr, testlr, testlr_abs, nn = feattest(nn, X, x, Y, y)

print 'MAE '+ str(testlr_abs)
print 'RMSE '+ str(testlr)


