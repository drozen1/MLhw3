
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import csv
from sklearn.model_selection import train_test_split
from sklearn.tree import plot_tree, DecisionTreeClassifier
import warnings
import scipy.stats as stats
import math
import re
from sklearn.model_selection import cross_validate
from sklearn.metrics import mean_squared_error, make_scorer

warnings.filterwarnings('ignore')

def CV_evaluation(h, X_train, y_train, n_splits=5):
    scores = cross_validate(h, X_train, y_train, cv=n_splits,
                            scoring=make_scorer(mean_squared_error), return_train_score=True)
    # TODO
    train_mse = 0
    valid_mse = 0
    return (train_mse, valid_mse)

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    dataset_labeled = pd.read_csv('variant_labeled.csv')
    dataset_unlabeled = pd.read_csv('variant_unlabeled.csv')
    # Q1:
    train, test = train_test_split(dataset_labeled, test_size=0.2, random_state=14)


    #correlation matrix with new features
    # plt.rc('xtick', labelsize=20)
    # plt.rc('ytick', labelsize=20)
    # corrMatrix = train.corr()
    # plt.figure(figsize=(20, 20))
    # plt.title('Final Correlation Map', fontsize=20)
    # ax = sns.heatmap(corrMatrix, xticklabels=True, yticklabels=True, annot=True)
    # plt.show()
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
