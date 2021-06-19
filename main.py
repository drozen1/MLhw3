
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

warnings.filterwarnings('ignore')

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    filename = 'virus_hw1.csv'
    dataset_labeled = pd.read_csv('variant_labeled.csv')
    dataset_unlabeled = pd.read_csv('variant_unlabeled.csv')
    # Q1:
    train, test = train_test_split(dataset_labeled, test_size=0.2, random_state=14)


    #correlation matrix with new features
    plt.rc('xtick', labelsize=20)
    plt.rc('ytick', labelsize=20)
    corrMatrix = train.corr()
    plt.figure(figsize=(20, 20))
    plt.title('Final Correlation Map', fontsize=20)
    ax = sns.heatmap(corrMatrix, xticklabels=True, yticklabels=True, annot=True)
    plt.show()
    # plt.savefig('correlation_matrix.jpg', bbox_inches='tight')