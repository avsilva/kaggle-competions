import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import skew

from elapsed_time import *
from plots import *

@profile
def heatmap(df, labels):
    sns.set(font_scale = 0.6)
    cm = np.corrcoef(df[labels].dropna().values.T)
    hm = sns.heatmap( cm, cbar = True, annot = True, square = True, fmt = '.2f')
    hm.set_xticklabels(labels, rotation = 90)
    hm.set_yticklabels(labels[::-1],rotation = 0)

    return hm, cm


@profile
def feattest(model, tfeats, Xtrain, xtest, Ytrain, ytest):
    """
    Testing the features one by one and scoring the
    models on the test and training set.

    """
    trainscrs = []
    testscrs = []
    testscrs_abs = []
    for i in range(1,len(tfeats)+1):
        X_t = Xtrain[tfeats[0:i]]
        x_t = xtest[tfeats[0:i]]
        model.fit(X_t,Ytrain)
        trainscrs.append(
                         np.sqrt(
                                 -cross_val_score(
                                                  model,
                                                  X_t,
                                                  Ytrain,
                                                  scoring="neg_mean_squared_error",
                                                  cv = 5
                                                 )
                                ).mean()
                        )
        testscrs.append(np.sqrt(mean_squared_error(ytest, model.predict(x_t))))
        testscrs_abs.append(mean_absolute_error(ytest, model.predict(x_t)))

    return trainscrs, testscrs, testscrs_abs, model


from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.cross_validation import cross_val_score, KFold
from sklearn.model_selection import train_test_split
from feature_selection import *
def feattest2(model, categorical_feats, num_feats, data_train ,target_data):
    """
    Testing the features one by one and scoring the
    models on the test and training set.

    """
    #print num_feats
    scores = []

    kf = KFold(len(target_data), 5, shuffle=False)
    for train_indices, test_indices in kf:
        Xtrain = data_train.ix[train_indices, :]
        Ytrain = target_data[train_indices]
        xtest = data_train.ix[test_indices, :]
        ytest = target_data[test_indices]

        univariate_selected_features = univariate_selection(Xtrain[categorical_feats], Ytrain)
        #print (len(univariate_selected_features.tolist()))
        print str(univariate_selected_features)
        final_feats = num_feats + univariate_selected_features.tolist()
        xtrain = Xtrain[final_feats]
        xtest = xtest[final_feats]
        model.fit(xtrain, Ytrain)
        score = np.sqrt(mean_squared_error(ytest, model.predict(xtest)))
        print score
        scores.append(score)

    print("CV Score is ", np.mean(scores))
    print("CV Score is ", np.min(scores))
    return univariate_selected_features.tolist()


@profile
def dataprep(df, numFeats, dumFeats):
    """
    Preparation of the data: Log transform numerical variables
    and join on the dummy variables for the categorical features.
    Simple mean filling applied for missing values.

    """
    df_t = df[numFeats].fillna(df[numFeats].mean()).apply(np.log1p)
    df_t = df_t.join(df[dumFeats])
    return df_t

def drop_columns(_frame, _arr_columns):
    _frame.drop(_arr_columns, axis=1, inplace=True)
    return _frame

def group_by_attr_type(_frame):
    g = _frame.columns.to_series().groupby(_frame.dtypes).groups
    grpdict = {k.name: v for k, v in g.items()}
    return grpdict

def categorical_to_dummy(_frame, _categorial):
    dummies = pd.get_dummies(_frame[_categorial]).columns.values.tolist()
    frame = pd.get_dummies(_frame)
    return frame, dummies

def print_elapsed_time(_start_time):
    elapsed_time = time.time() - _start_time
    print str(elapsed_time) + ' seg.'

def print_results(_array):
    for result in _array:
        print result

def get_df_summary(_df):
    number_attrs = len(list(_df.columns.values))
    attrs_name = list(_df.columns.values)
    #info = _df.info()
    return [number_attrs, attrs_name]



#choosing most important categorical features
from sklearn.ensemble import RandomForestRegressor
def choosing(frame, dummies):
    rf = RandomForestRegressor(n_estimators=100)
    rf.fit(frame[dummies], frame['SalePrice'].apply(np.log1p))

    frameFeats = pd.DataFrame(data = None, columns=['Feature', 'Categorical Feature Importance'])
    frameFeats['Feature'] = dummies
    frameFeats['Categorical Feature Importance'] = rf.feature_importances_

    # Output feature importance coefficients, map them to their feature name, and sort values
    coef = frameFeats.sort_values('Categorical Feature Importance', ascending = False)
    coef.head(40).plot(x = 'Feature', y = 'Categorical Feature Importance', kind = 'bar')
    plt.tight_layout()
    plt.savefig('./img/cat_features_importance.png')
    #plt.show()
    top = frameFeats.sort_values('Categorical Feature Importance', ascending = False)
    dummyFeats = top['Feature'].values.tolist()

def explore_data(frame):
    ##print stats
    #print(frame['SalePrice'].describe())
    #print("\nmedian: ", frame['SalePrice'].median(axis=0))

    #draw_histogram(frame)
    draw_histogram2(frame)
    draw_heatmap_for_numerical_feats(frame)
    draw_pair_plot(frame, 6)

def prepare_data(train, test, categorical):


    all_data = pd.concat((train.loc[:, 'MSSubClass':'SaleCondition'], test.loc[:, 'MSSubClass':'SaleCondition']))
    #print all_data.info()
    #print all_data.shape
    cat_feats = pd.get_dummies(all_data[categorical]).columns.values.tolist()


    # log transform the target:
    train["SalePrice"] = np.log1p(train["SalePrice"])

    # log transform skewed numeric features:
    numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index


    skewed_feats = train[numeric_feats].apply(lambda x: skew(x.dropna()))  # compute skewness
    skewed_feats = skewed_feats[skewed_feats > 0.75]
    skewed_feats = skewed_feats.index

    #all_data[skewed_feats] = np.log1p(all_data[skewed_feats])
    all_data[numeric_feats] = np.log1p(all_data[numeric_feats])

    #Convert categorical variable into dummy/indicator variables
    all_data = pd.get_dummies(all_data)
    # filling NA's with the mean of the column:
    all_data = all_data.fillna(all_data.mean())

    # creating matrices for sklearn:
    X_train = all_data[:train.shape[0]]
    X_test = all_data[train.shape[0]:]
    y = train.SalePrice
    return X_train, X_test, cat_feats, numeric_feats.tolist(), y
