import time
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from joblib import dump
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix)
from sklearn.metrics import roc_auc_score, average_precision_score
from RefinedRandomForest import RefinedRandomForest

x_train = np.load('../feature_extraction/ResNext101/features_ResNext101_train.npy')
y_train = np.load('../feature_extraction/ResNext101/labels_ResNext101_train.npy')
x_valid = np.load('../feature_extraction/ResNext101/features_ResNext101_valid.npy')
y_valid = np.load('../feature_extraction/ResNext101/labels_ResNext101_valid.npy')
x_train = np.vstack((x_train, x_valid))
y_train = np.vstack((y_train, y_valid))
x_test = np.load('../feature_extraction/ResNext101/features_ResNext101_test.npy')
y_test = np.load('../feature_extraction/ResNext101/labels_ResNext101_test.npy')

aucs = np.load('aucs.npy')
nodes = np.load('nodes.npy')
optimal = aucs.argmax(axis=1)
for i in range(y_test.shape[1]):
    if optimal[i]!=0:
        times = optimal[i]
        rfc = joblib.load(f'saved_model/rfc_{i}.joblib')
        rrfc = RefinedRandomForest(rfc, C = 0.01, n_prunings = 1)
        for j in range(times):
            rrfc.fit(x_train, y_train[:,i])
        dump(rrfc, f'saved_model/rrfc/rrfc_{i}.joblib')