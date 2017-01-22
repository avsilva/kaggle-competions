import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns

def draw_each_features_added(tf, lastFeat, trainlr, testlr):
    #fig_f = plt.subplots(figsize = (20,10))

    plt.plot(trainlr[0:lastFeat],'b',label = 'LR Train')
    plt.plot(testlr[0:lastFeat],'b--',label = 'LR Test')
    plt.xlabel('Each feature added cumulatively to the data set')
    plt.ylabel('RMSE')
    plt.xticks(range(0, len(tf[0:lastFeat])), tf[0:lastFeat], rotation = 'vertical')
    plt.legend(loc = 1)
    plt.tight_layout()
    plt.savefig('./img/each_feature_added.png')

def draw_cat_features_importance(coef):
    coef.head(40).plot(x='Feature', y='Categorical Feature Importance', kind='bar')
    plt.tight_layout()
    plt.savefig('./img/cat_features_importance.png')

def draw_histogram2(frame):
    sns.set(font_scale = 1)
    histplot = sns.distplot(frame['SalePrice'], kde=False, color='b', hist_kws={'alpha': 0.9})
    histplot.get_figure().tight_layout()
    histplot.get_figure().savefig('./img/hist2.png')

def draw_histogram(frame):
    plt.rcParams['figure.figsize'] = (12.0, 6.0)
    prices = pd.DataFrame({"price": frame["SalePrice"], "log(price + 1)": np.log1p(frame["SalePrice"])})
    prices.hist()
    plt.tight_layout()
    plt.savefig('./img/hist.png')

def draw_heatmap_for_numerical_feats(frame):
    corr = frame.select_dtypes(include=['float64', 'int64']).iloc[:, 1:].corr()
    labels = frame.select_dtypes(include=['float64', 'int64']).iloc[:, 1:]
    plt.figure(figsize=(12, 12))
    heatmap = sns.heatmap(corr, vmax=1, square=True)
    heatmap.set_xticklabels(labels, rotation=90)
    heatmap.set_yticklabels(labels[::-1], rotation=0)
    heatmap.get_figure().savefig('./img/heatmap2.png')

def draw_pair_plot(frame, _n_feats):
    labels = frame.select_dtypes(include=['float64', 'int64']).iloc[:, 1:].columns.values
    corrm = np.corrcoef(frame[labels].dropna().values.T)
    most_corr_feats = [x for (y, x) in sorted(zip(corrm[-1], labels), reverse=True)]
    pairplot = sns.pairplot(frame[most_corr_feats[0:_n_feats]].dropna())
    pairplot.savefig('./img/pairplot.png')

def autolabel(rects):
    # attach some text labels
    fig, ax = plt.subplots()
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., 1.05*height,
                '%f' % float(height),
                ha='center', va='bottom')

def plt_compare_scores(minscrs, labels):
    #plt.figure(figsize=(20, 10))
    ind_scrs = np.arange(0, len(minscrs))
    width = 0.3
    fig, ax = plt.subplots()
    rects = plt.bar(ind_scrs, minscrs, width, color='b')
    ax.set_xticks(np.array(ind_scrs) + width / 2)
    ax.set_xticklabels(labels)
    ax.set_ylabel('RMSE on test set')
    ax.set_ylim([0, 0.2])
    #autolabel(rects)

