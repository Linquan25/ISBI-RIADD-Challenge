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

aucs = np.zeros((y_test.shape[1],51))
nodes = np.zeros((y_test.shape[1],51))
for i in range(y_test.shape[1]):
    rfc = joblib.load(f'saved_model/rfc_{i}.joblib')
    y_test_prob = rfc.predict_proba(x_test)
    y_test_prob = np.array(y_test_prob)
    pred = y_test_prob[:,1]
    auc = roc_auc_score(y_test[:,i], pred)
    aucs[i,0] = auc
    t0 = time.time()
    rrfc = RefinedRandomForest(rfc, C = 0.01, n_prunings = 1)
    nodes[i,0]=sum(rrfc.n_leaves_)
    print(f'Time it took for refinement of rfc {i}: {time.time() - t0:.3f} s. ')
    for j in range(50):
        t0 = time.time()
        rrfc.fit(x_train, y_train[:,i])
        out = rrfc.predict_proba(x_test)[:,1]
        auc = roc_auc_score(y_test[:,i], out)
        aucs[i,j+1] = auc
        nodes[i,j+1]=sum(rrfc.n_leaves_)
        
np.save('aucs.npy', aucs)
np.save('nodes.npy', nodes)