
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import csv

import sklearn.dummy
from sklearn.model_selection import train_test_split
from sklearn.tree import plot_tree, DecisionTreeClassifier
import warnings
import scipy.stats as stats
import math
import re
from sklearn.model_selection import cross_validate
from sklearn.metrics import mean_squared_error, make_scorer
from sklearn.linear_model import Ridge

warnings.filterwarnings('ignore')
from sklearn.base import BaseEstimator, RegressorMixin


class MultiRegressor(BaseEstimator, RegressorMixin):
 def __init__(self, h_male, h_female):
    self.h_male = h_male
    self.h_female = h_female

 def fit(self, dataset):
    men = dataset[dataset.Sex == 0]
    xmen = men.drop(['VariantScore'], axis=1)
    female = dataset[dataset.Sex == 1]
    xfemale = female.drop(['VariantScore'], axis=1)
    self.h_male.fit(xmen,men.VariantScore )
    self.h_female.fit(xfemale,female.VariantScore)
    return self

 def predict(self, X):
    # X should be a pandas dataframe
    all_predictions = []

    for index, x in X.iterrows():
        print(x)
        if(x.Sex == 0):
            y_pred = self.h_male.predict([x])
            all_predictions.append(y_pred)
        else:
            y_pred = self.h_female.predict([x])
            all_predictions.append(y_pred)
    return all_predictions

def CV_evaluation(h, X_train, y_train, n_splits=5):
    scores = cross_validate(h, X_train, y_train, cv=n_splits,
                            scoring=make_scorer(mean_squared_error), return_train_score=True)
    train_mse = np.sum(scores['train_score'])/n_splits
    valid_mse = np.sum(scores['test_score'])/n_splits
    return (train_mse, valid_mse)

def calcHyperparameter(scale, trainX, trainY):
    train = []
    validation = []
    for Alpha in scale:
        h = sklearn.linear_model.Ridge(alpha=Alpha, fit_intercept=True)
        train_mse, valid_mse = CV_evaluation(h, trainX, trainY, n_splits=5)
        train.append(train_mse)
        validation.append(valid_mse)
    return (train,validation)



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    dataset_labeled = pd.read_csv('variant_labeled.csv')#,index_col=[0]
    dataset_unlabeled = pd.read_csv('variant_unlabeled.csv')#,index_col=[0]
    # Q1:
    train, test = train_test_split(dataset_labeled, test_size=0.2, random_state=14)


    #correlation matrix with new features
    # plt.rc('xtick', labelsize=20)
    # plt.rc('ytick', labelsize=20)
    # corrMatrix = train.corr()
    # plt.figure(figsize=(20, 20))
    # plt.title('Final Correlation Map', fontsize=20)
    # ax = sns.heatmap(corrMatrix, xticklabels=True, yticklabels=True, annot=True)
    # # plt.show()
    # plt.savefig('correlation_matrix.jpg', bbox_inches='tight')

    print(dataset_labeled.info())

    # Q2 conditional distribution for Sex:
    """""
    sns.kdeplot(data=train, x="VariantScore", hue="Sex", multiple="stack")
    plt.title('Condition Distribution of Sex')
    plt.grid()
    plt.savefig('ConditionalDistributionSex.jpg', bbox_inches='tight')
    """""
    # Q3 conditional distribution for BloodType:
    """""
    train.BloodType = train.BloodType.fillna('missing value')
    g = sns.FacetGrid(train, col="BloodType", height=3.5, aspect=.65, col_wrap=5)
    g.map(sns.kdeplot, "VariantScore")
    g.set_titles(col_template="{col_name}", size=18)
    g.set_xlabels(size=18)
    g.set_axis_labels("VariantScore", "Density")
    for ax in g.axes:
        ax.grid(alpha=0.5)
    plt.savefig('ConditionalDistributionBloodType.jpg', bbox_inches='tight')
    """""

    # Q7 CV evaluation using dummy regressor:
    # reading the numeric and final train set:
    data_after_drop = pd.read_csv('train_labeld.csv',index_col=[0])
    trainX = data_after_drop.drop(['VariantScore'], axis=1)
    trainY = data_after_drop.VariantScore
    h1 = sklearn.dummy.DummyRegressor(strategy='mean')
    train_mse, valid_mse = CV_evaluation(h1, trainX, trainY, n_splits=5)
    print(train_mse)
    print(valid_mse)
    """""
    scale = np.logspace(-2.5,2.3,num=50)
    train_list, valid_list = calcHyperparameter(scale, trainX, trainY)
    plt.loglog(scale, train_list, c='b')
    plt.loglog(scale, valid_list, c='y')
    plt.loglog([min(scale), max(scale)],[valid_mse, valid_mse],c='g')
    plt.title('Tuning of $\lambda$')
    plt.xlabel('$\lambda$')
    plt.ylabel('mse')
    plt.legend(['train', 'validation'], loc='lower left')
    plt.grid()
    plt.savefig('lambda_dummy_log.jpg', bbox_inches='tight')
    plt.close()
    """""
    """""
    scale = range(10,100)
    scale = np.logspace(-5, 1.2, num=100)
   # scale = np.linspace(-4, 4, num=100)
    train_list, valid_list = calcHyperparameter(scale, trainX, trainY)
    plt.semilogx(scale, train_list, c='b')
    plt.semilogx(scale, valid_list, c='y')
    plt.title('Tuning of $\lambda$')
    plt.xlabel('$\lambda$')
    plt.ylabel('mse')
    plt.legend(['train', 'validation'], loc='lower left')
    plt.grid()
    plt.savefig('lambda_dummy.jpg', bbox_inches='tight')
    plt.close()
    """""
    Alpha = 50
    columns = list(trainX)
    h = sklearn.linear_model.Ridge(alpha=Alpha, fit_intercept=True)
    h.fit(trainX, trainY)
    pd.Series(abs(h.coef_), index=trainX.columns).nlargest(5).plot(kind='barh')
    plt.grid()
    # plt.show()
    plt.savefig('weights.jpg', bbox_inches='tight')

    #Q12
    Multi_Regressor =MultiRegressor(sklearn.linear_model.Ridge(alpha=10, fit_intercept=True),sklearn.linear_model.Ridge(alpha=0.2, fit_intercept=True))
    Multi_Regressor.fit(data_after_drop)
    temp = data_after_drop.drop(['VariantScore'], axis=1)
    Multi_Regressor.predict(temp)