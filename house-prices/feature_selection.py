import sys
import pandas
import numpy

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
def univariate_selection(X, Y):

    #print X.values.shape[1]
    #nfeatures = X.values.shape[1] - X.shape[1]/4
    nfeatures = X.values.shape[1] / 6
    #print 'Initial n features ' + str(X.values.shape[1]) + '.\nFinal n features ' + str(nfeatures)
    Y = Y.astype(int)
    test = SelectKBest(score_func=chi2, k=nfeatures)
    fit = test.fit(X.values, Y)

    return X.columns.values[fit.get_support()]


from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import LassoCV
def recursive_feature_elimination(X, Y):
    nfeatures = X.values.shape[1] / 5
    model = LassoCV()
    test = RFE(model, nfeatures)
    Y = Y.astype(int)
    fit = test.fit(X.values, Y)
    return X.columns.values[fit.get_support()]
    #return fit

from sklearn import preprocessing
from sklearn.feature_selection import RFECV
def recursive_feature_elimination2(X, Y):
    nfeatures = X.values.shape[1] / 5
    #lm = RidgeCV()
    lm = LinearRegression()
    rfecv = RFECV(estimator=lm, step=1, cv=5, scoring='r2')
    Y = Y.astype(int)
    fit = rfecv.fit(X.values, Y)
    print("Optimal number of features : %d" % rfecv.n_features_)
    #print X.columns.values[fit.get_support()]
    return X.columns.values[fit.get_support()]
    #return fit

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.tree import DecisionTreeRegressor
import numpy as np
import pandas as pd
def feature_importance(X, Y):

    nfeatures = X.values.shape[1] / 6
    params = {'n_estimators': 500, 'max_depth': 4, 'min_samples_split': 2,
              'learning_rate': 0.01, 'loss': 'ls'}
    params2 = {'n_estimators': 500,
              'learning_rate': 0.01, 'loss': 'linear'}

    #model = RandomForestRegressor(n_estimators=100)
    #model = RandomForestRegressor(**params)
    model = GradientBoostingRegressor(**params)
    #model = AdaBoostRegressor(**params2)


    fit = model.fit(X.values, Y.apply(np.log1p))
    frameFeats = pd.DataFrame(data=None, columns=['Feature', 'Categorical Feature Importance'])
    frameFeats['Feature'] = X.columns.values
    frameFeats['Categorical Feature Importance'] = fit.feature_importances_
    coef = frameFeats.sort_values('Categorical Feature Importance', ascending=False)
    #draw_cat_features_importance(coef)

    top = frameFeats.sort_values('Categorical Feature Importance', ascending=False)
    dummyFeats = top['Feature'].values.tolist()
    return dummyFeats[0:nfeatures], coef
    #return X.columns.values[fit.get_support()]

from sklearn.decomposition import PCA
def pca(X):
    nfeatures = X.values.shape[1] / 5
    pca = PCA(nfeatures)
    fit = pca.fit(X.values)
    print(fit.components_)



