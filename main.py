import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import csv
import matplotlib.patches as  mpatches

import sklearn.dummy
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.tree import plot_tree, DecisionTreeClassifier
import warnings
import scipy.stats as stats
import math
import re
from sklearn.model_selection import cross_validate
from sklearn.metrics import mean_squared_error, make_scorer
from sklearn.linear_model import Ridge
from sklearn.svm import SVR, SVC
from sklearn.ensemble import BaggingRegressor, BaggingClassifier
from sklearn.datasets import make_regression

warnings.filterwarnings('ignore')
from sklearn.base import BaseEstimator, RegressorMixin


class MultiRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, h_male, h_female):
        self.h_male = h_male
        self.h_female = h_female

    def fit(self, X, y):
        dataset = X
        dataset['VariantScore'] = y
        men = dataset[dataset.Sex == 0]
        xmen = men.drop(['VariantScore'], axis=1)
        xmen = xmen.drop(['Sex'], axis=1)
        female = dataset[dataset.Sex == 1]
        xfemale = female.drop(['VariantScore'], axis=1)
        xfemale = xfemale.drop(['Sex'], axis=1)
        ymen = men.VariantScore
        self.h_male = self.h_male.fit(xmen, ymen)
        yfemale = female.VariantScore
        self.h_female = self.h_female.fit(xfemale, yfemale)
        return self

    def predict(self, X):
        # X should be a pandas dataframe

        if 'VariantScore' in X.columns:
            X = X.drop(['VariantScore'], axis=1)
        all_predictions = []

        for index, x in X.iterrows():
            if (x.Sex == 0):
                x = x.drop(x.index[18])
                y_pred = self.h_male.predict([x])
                all_predictions.append(y_pred)
            else:
                x = x.drop(x.index[18])
                y_pred = self.h_female.predict([x])
                all_predictions.append(y_pred)

        return all_predictions


def CV_evaluation(h, X_train, y_train, n_splits=5):
    scores = cross_validate(h, X_train, y_train, cv=n_splits,
                            scoring=make_scorer(mean_squared_error), return_train_score=True)
    train_mse = np.sum(scores['train_score']) / n_splits
    valid_mse = np.sum(scores['test_score']) / n_splits
    return (train_mse, valid_mse)


def calcHyperparameter(scale, trainX, trainY):
    train = []
    validation = []
    for Alpha in scale:
        h = sklearn.linear_model.Ridge(alpha=Alpha, fit_intercept=True)
        train_mse, valid_mse = CV_evaluation(h, trainX, trainY, n_splits=5)
        train.append(train_mse)
        validation.append(valid_mse)
    return (train, validation)


def retrain_dummy(trainX, trainY, data_after_drop_test):
    h1 = sklearn.dummy.DummyRegressor(strategy='mean')
    dummy_regr = h1.fit(trainX, trainY)
    x_test = data_after_drop_test.drop(['VariantScore'], axis=1)
    res = dummy_regr.predict(x_test)
    mse = mean_squared_error(res, data_after_drop_test.VariantScore)
    print(mse)

def retrain_model(h1,trainX, trainY, data_after_drop_test):
    my_regr = h1.fit(trainX, trainY)
    x_test = data_after_drop_test.drop(['VariantScore'], axis=1)
    res = my_regr.predict(x_test)
    mse = mean_squared_error(res, data_after_drop_test.VariantScore)
    print(mse)

def calc_pred(h,trainX, trainY,dataset_unlabeled_normalize):
    my_regr = h.fit(trainX, trainY)
    IDs = dataset_unlabeled_normalize.ID
    dataset_unlabeled_normalize = dataset_unlabeled_normalize.drop(['ID'],axis=1)
    res = my_regr.predict(dataset_unlabeled_normalize)
    pred = pd.DataFrame(columns=['ID', 'VariantScore'])
    pred['ID'] = IDs
    pred['VariantScore'] = res
    return pred

def calc_pred4(h,trainX, trainY,dataset_unlabeled_normalize):
    my_regr = h.fit(trainX, trainY)
    IDs = dataset_unlabeled_normalize.ID
    dataset_unlabeled_normalize = dataset_unlabeled_normalize.drop(['ID'],axis=1)
    res = my_regr.predict(dataset_unlabeled_normalize)
    res1=[]
    for i in res:
        res1.append(i[0])
    new_series = pd.Series(res1)
    pred = pd.DataFrame(columns=['ID', 'VariantScore'])
    pred['ID'] = IDs
    pred['VariantScore'] = new_series
    return pred


# Press the green button in the gutter to run the script.
# TODO: retrain after evaluation
if __name__ == '__main__':
    dataset_labeled = pd.read_csv('variant_labeled.csv')  # ,index_col=[0]
    dataset_unlabeled = pd.read_csv('variant_unlabeled.csv')  # ,index_col=[0]
    dataset_unlabeled_normalize = pd.read_csv('unlabeld_test_normalize.csv' ,index_col=[0])
    all_data = pd.read_csv('all_data.csv',index_col=[0])
    all_dataX = all_data.drop(['VariantScore'], axis=1)
    all_dataY = all_data.VariantScore
    # Q1:
    train, test = train_test_split(dataset_labeled, test_size=0.2, random_state=14)

    # correlation matrix with new features
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
    data_after_drop = pd.read_csv('train_labeld.csv', index_col=[0])
    data_after_drop_test = pd.read_csv('test_labeld.csv', index_col=[0])

    trainX = data_after_drop.drop(['VariantScore'], axis=1).copy()
    trainY = data_after_drop.VariantScore.copy()
    # training the dummy regressor

    # """""
    h1 = sklearn.dummy.DummyRegressor(strategy='mean')
    train_mse, valid_mse = CV_evaluation(h1, trainX, trainY, n_splits=5)

    print(train_mse)
    print(valid_mse)
    # retrain:
    """""
    retrain_dummy(trainX, trainY, data_after_drop_test)
    """""


    # """""

    # Q8:
    """""
    scale = np.logspace(-2,5,num=100)
    train_list, valid_list = calcHyperparameter(scale, trainX, trainY)
    index_min = min(range(len(valid_list)), key=valid_list.__getitem__)
    print(index_min)
    print(min(valid_list))
    print(scale[index_min])
    plt.loglog(scale, train_list, c='b')
    plt.loglog(scale, valid_list, c='y')
    plt.loglog([min(scale), max(scale)],[valid_mse, valid_mse],c='g')
    plt.title('Tuning of $\lambda$ for basic linear model')
    plt.xlabel('$\lambda$')
    plt.ylabel('mse')
    plt.legend(['train', 'validation','dummy regressor validation'], loc='upper left')
    plt.grid()
    plt.savefig('lambda_dummy_log.jpg', bbox_inches='tight')
    plt.close()
    
    # retrain:
    h = sklearn.linear_model.Ridge(alpha=3.6565656565656566, fit_intercept=True)
    retrain_model(h,trainX, trainY, data_after_drop_test)
    """""
    #for part 8:

    h = sklearn.linear_model.Ridge(alpha=3.6565656565656566, fit_intercept=True)
    df = calc_pred(h,all_dataX, all_dataY,dataset_unlabeled_normalize)
    df.to_csv(r'C:\Users\dor\PycharmProjects\MLhw3\pred3.csv')

    # we ran a more dense check closer to the minimum of the last plot
    """""
    # scale = range(10,100)
    # scale = np.logspace(0, 10, num=100)
    scale = np.linspace(2, 4, num=100)
    train_list, valid_list = calcHyperparameter(scale, trainX, trainY)
    index_min = min(range(len(valid_list)), key=valid_list.__getitem__)
    print(index_min)
    print(min(valid_list))
    print(scale[index_min])
    print(index_min)
    print(train_list[index_min])
    plt.semilogx(scale, train_list, c='b')
    plt.semilogx(scale, valid_list, c='y')
    plt.title('Tuning of $\lambda$ for basic linear model')
    plt.xlabel('$\lambda$')
    plt.ylabel('mse')
    plt.legend(['train', 'validation', 'dummy regressor validation'], loc='lower left')
    plt.grid()
    plt.savefig('lambda_dummy.jpg', bbox_inches='tight')
    plt.close()
    """""

    # Q10
    """""
    Alpha = 3.6565656565656566
    columns = list(trainX)
    h = sklearn.linear_model.Ridge(alpha=Alpha, fit_intercept=True)
    h.fit(trainX, trainY)
    a = pd.Series(abs(h.coef_), index=trainX.columns).nlargest(5)[::-1]
    a.plot(kind='barh')
    # pd.Series(abs(h.coef_), index=trainX.columns).nlargest(5).plot(kind='barh')
    plt.title('Most Significant Coefficients for the Basic Linear Regressor')
    plt.xlabel('Coefficient')
    plt.grid()
    # plt.show()
    plt.savefig('weights.jpg', bbox_inches='tight')
    """""

    # Q11
    """""
    g = sns.kdeplot(data=data_after_drop, x="VariantScore", y="AgeGroup", hue="Sex", legend=False)
    # g.set(title='Kdeplot', legend=['Male', 'Female'])
    # g.legend(title='Sex', labels=['Male', 'Female'], loc='upper right')
    handles = [mpatches.Patch(facecolor=plt.cm.Blues(100), label='Male'), mpatches.Patch(facecolor=plt.cm.Oranges(100),label='Female')]
    # g.legend((['Male', 'Female']), title='Sex', loc='upper right')
    plt.legend(title = 'Sex', handles=handles)
    plt.title('Bivariate Distribution of VariantScore and AgeGroup for Male and Female')
    plt.grid()
    plt.savefig('bivariate_ageGroup.jpg', bbox_inches='tight')
    """""

    # Q12
    men = data_after_drop[data_after_drop.Sex == 0]
    men = men.drop(['Sex'], axis=1)
    xmen = men.drop(['VariantScore'], axis=1)
    female = data_after_drop[data_after_drop.Sex == 1]
    female = female.drop(['Sex'], axis=1)
    xfemale = female.drop(['VariantScore'], axis=1)

    """""
    scale = np.logspace(-2.5,4,num=100)
    train_list, valid_list = calcHyperparameter(scale, xmen, men.VariantScore)
    index_min = min(range(len(valid_list)), key=valid_list.__getitem__)
    print(index_min)
    print(min(valid_list))
    print(scale[index_min])
    print("print(train_list[index_min]): ",train_list[index_min])
    plt.loglog(scale, train_list, c='b')
    plt.loglog(scale, valid_list, c='y')
    plt.loglog([min(scale), max(scale)],[valid_mse, valid_mse],c='g')
    plt.title('Tuning of $\lambda$ for Male')
    plt.xlabel('$\lambda$')
    plt.ylabel('mse')
    plt.legend(['train', 'validation', 'dummy regressor validation'], loc='upper left')
    plt.grid()
    plt.savefig('male.jpg', bbox_inches='tight')
    plt.close()
    """""
    """""
    # TODO: have two plots: with and without dummy regressor
    scale = np.logspace(-0.5,3.5,num=50)
    train_list, valid_list = calcHyperparameter(scale, xfemale, female.VariantScore)
    index_min = min(range(len(valid_list)), key=valid_list.__getitem__)
    print(index_min)
    print(min(valid_list))
    print(scale[index_min])
    print("print(train_list[index_min]): ",train_list[index_min])
    plt.loglog(scale, train_list, c='b')
    plt.loglog(scale, valid_list, c='y')
    plt.loglog([min(scale), max(scale)],[valid_mse, valid_mse],c='g')
    # plt.title('Tuning of $\lambda$ for Female')
    plt.title('Tuning of $\lambda$ for Female\n Zoomed In')
    plt.xlabel('$\lambda$')
    plt.ylabel('mse')
    # plt.legend(['train', 'validation', 'dummy regressor validation'], loc='upper left')
    plt.legend(['train', 'validation'], loc='lower left')
    plt.grid()
    plt.savefig('female.jpg', bbox_inches='tight')
    plt.close()
    """""



    # Q13
    """""
    Multi_Regressor = MultiRegressor(sklearn.linear_model.Ridge(alpha=37.214061, fit_intercept=True),
                                     sklearn.linear_model.Ridge(alpha=702.973212, fit_intercept=True))
    # Multi_Regressor.fit(data_after_drop)
    # temp = data_after_drop.drop(['VariantScore'], axis=1)
    train_mse, valid_mse = CV_evaluation(Multi_Regressor, trainX, trainY, n_splits=5)
    print(train_mse)
    print(valid_mse)
    # Multi_Regressor.predict(temp)
      """""
    # retrain
    """""
    h = MultiRegressor(sklearn.linear_model.Ridge(alpha=37.214061, fit_intercept=True),
                                     sklearn.linear_model.Ridge(alpha=702.973212, fit_intercept=True))
    retrain_model(h,trainX, trainY, data_after_drop_test)
    """""

    #part 8:

    h = MultiRegressor(sklearn.linear_model.Ridge(alpha=37.214061, fit_intercept=True),
                                     sklearn.linear_model.Ridge(alpha=702.973212, fit_intercept=True))
    df = calc_pred4(h,all_dataX, all_dataY,dataset_unlabeled_normalize)
    df.to_csv(r'C:\Users\dor\PycharmProjects\MLhw3\pred4.csv')

    # Q17

    col = list(data_after_drop)
    not_include = ['BloodType', 'Sex', 'VariantScore', 'ID']
    poly_data = data_after_drop.copy()
    power = 2 * np.ones(data_after_drop.shape[0])
    for i in col:
        if (i not in not_include):
            new_col = pd.Series(data_after_drop[i])
            new_col = new_col.pow(power)
            new_name = i + '_square'
            poly_data[new_name] = new_col
    # print(poly_data.info())
    men = poly_data[poly_data.Sex == 0]
    xmen = men.drop(['VariantScore'], axis=1)
    female = poly_data[poly_data.Sex == 1]
    xfemale = female.drop(['VariantScore'], axis=1)

    #for retrain:
    data_after_drop_test_poly = data_after_drop_test.copy()

    col = list(data_after_drop_test_poly)
    power = 2 * np.ones(data_after_drop_test_poly.shape[0])
    for i in col:
        if (i not in not_include):
            new_col = pd.Series(data_after_drop_test_poly[i])
            new_col = new_col.pow(power)
            new_name = i + '_square'
            data_after_drop_test_poly[new_name] = new_col


    # for part 8:
    dataset_unlabeled_normalize_poly = dataset_unlabeled_normalize.copy()
    col = list(dataset_unlabeled_normalize_poly)
    power = 2 * np.ones(dataset_unlabeled_normalize_poly.shape[0])
    for i in col:
        if (i not in not_include):
            new_col = pd.Series(dataset_unlabeled_normalize_poly[i])
            new_col = new_col.pow(power)
            new_name = i + '_square'
            dataset_unlabeled_normalize_poly[new_name] = new_col

    all_data_poly = all_data.copy()
    col = list(all_data_poly)
    power = 2 * np.ones(all_data_poly.shape[0])
    for i in col:
        if (i not in not_include):
            new_col = pd.Series(all_data_poly[i])
            new_col = new_col.pow(power)
            new_name = i + '_square'
            all_data_poly[new_name] = new_col
    all_data_polyX = all_data_poly.copy()
    all_data_polyX = all_data_polyX.drop(['VariantScore'], axis=1)
    all_data_polyY = all_data_poly.VariantScore


    """""
    # scale = np.logspace(-2,12,num=500)
    #1000000000000.0
    scale = np.linspace(20000000000000, 900000000000000, num=200)
    xmen_no_sex =xmen.drop(['Sex'], axis=1)
    train_list, valid_list = calcHyperparameter(scale, xmen_no_sex, men.VariantScore)
    index_min = min(range(len(valid_list)), key=valid_list.__getitem__)
    print(index_min)
    print(min(valid_list))
    print(scale[index_min])
    print("print(train_list[index_min]): ",train_list[index_min])
    plt.loglog(scale, train_list, c='b')
    plt.loglog(scale, valid_list, c='y')
    plt.loglog([min(scale), max(scale)],[valid_mse, valid_mse],c='g')
    plt.title('Tuning of $\lambda$ for Male')
    plt.xlabel('$\lambda$')
    plt.ylabel('mse')
    plt.legend(['train', 'validation','dummy regressor validation'], loc='upper right')
    plt.grid()
    plt.savefig('male_poly.jpg', bbox_inches='tight')
    plt.close()
    """""

    """""
    # TODO: have two plots: with and without dummy regressor
    scale = np.logspace(6,9,num=500)
    scale = np.linspace(10160000, 12390000, num=500)
    train_list, valid_list = calcHyperparameter(scale, xfemale, female.VariantScore)
    index_min = min(range(len(valid_list)), key=valid_list.__getitem__)
    print(index_min)
    print(min(valid_list))
    print(scale[index_min])
    print("print(train_list[index_min]): ",train_list[index_min])
    plt.loglog(scale, train_list, c='b')
    plt.loglog(scale, valid_list, c='y')
    plt.loglog([min(scale), max(scale)],[valid_mse, valid_mse],c='g')
    plt.title('Tuning of $\lambda$ for Female')
    plt.xlabel('$\lambda$')
    plt.ylabel('mse')
    plt.legend(['train', 'validation', 'dummy regressor validation'], loc='upper right')
    plt.grid()
    plt.savefig('female_poly.jpg', bbox_inches='tight')
    plt.close()
      """""

    #     Q20:
    """""
    Multi_Regressor = MultiRegressor(sklearn.linear_model.Ridge(alpha=387035175879397.0, fit_intercept=True),
                                     sklearn.linear_model.Ridge(alpha=11268296.593186373, fit_intercept=True))
    X_poly_data = poly_data.drop(['VariantScore'], axis=1)
    Multi_Regressor.fit(X_poly_data, poly_data.VariantScore)
    #train_mse, valid_mse = CV_evaluation(Multi_Regressor, X_poly_data, poly_data.VariantScore, n_splits=5)
    #print(train_mse)
    #print(valid_mse)


    h =  MultiRegressor(sklearn.linear_model.Ridge(alpha=387035175879397.0, fit_intercept=True),
                                     sklearn.linear_model.Ridge(alpha=11268296.593186373, fit_intercept=True))
    retrain_model(h,X_poly_data, poly_data.VariantScore, data_after_drop_test_poly)
     """""

    #part 8:

    h = MultiRegressor(sklearn.linear_model.Ridge(alpha=37.214061, fit_intercept=True),
                                     sklearn.linear_model.Ridge(alpha=702.973212, fit_intercept=True))
    df = calc_pred4(h,all_data_polyX, all_data_polyY,dataset_unlabeled_normalize_poly)
    df.to_csv(r'C:\Users\dor\PycharmProjects\MLhw3\pred5.csv')



    # # Q21
    # # data_after_drop.drop(['VariantScore'], axis=1).copy()
    # # trainx_arr = trainX.to_numpy()
    # # trainy_arr = trainY.to_numpy()
    # print(data_after_drop.info())
    # mlp = MLPRegressor(max_iter=700,alpha=0.0001)
    # train_mse, valid_mse = CV_evaluation(mlp, trainX, trainY, n_splits=5)
    # print(train_mse)
    # print(valid_mse)
    # print(valid_mse)

    # # 2.2 Trying Bagging of SVM with polynomial kernel
    # bag_poly_svm = get_best_classifier_on_train(BaggingClassifier(base_estimator=SVC(gamma='auto', kernel="poly"), n_estimators=20, random_state=0), train_set, 7)
    # bag_pred = bag_poly_svm.predict(validation_set.drop('Vote', axis=1))
    # curr_f1_score = metrics.f1_score(validation_set['Vote'], bag_pred, average='weighted')
    # bag_poly_svm_accuracy = metrics.accuracy_score(validation_set['Vote'], bag_pred)
    # print("Bagging with 20 polynomial SVC: f1 score on validation is: {}, and accuracy: {}".format(curr_f1_score, bag_poly_svm_accuracy))


    # bag = BaggingRegressor(n_estimators=30, random_state=0)
    # train_mse, valid_mse = CV_evaluation(bag, trainX, trainY, n_splits=5)
    # print(train_mse)
    # print(valid_mse)
    scale = range(10,20,1)
    train = []
    validation = []
    for Alpha in scale:
        h = BaggingRegressor(n_estimators=Alpha, random_state=0)
        # h = MultiRegressor(sklearn.linear_model.Ridge(alpha=37.214061, fit_intercept=True),
        #                    sklearn.linear_model.Ridge(alpha=702.973212, fit_intercept=True))
        train_mse, valid_mse = CV_evaluation(h, all_data_polyX, all_data_polyY, n_splits=5)
        print(Alpha)
        print(train_mse)
        print(valid_mse)
        train.append(train_mse)
        validation.append(valid_mse)
    # h = BaggingRegressor(n_estimators=100, random_state=0)
    # train_mse, valid_mse = CV_evaluation(h, trainX, trainY, n_splits=5)
    # print(valid_mse)