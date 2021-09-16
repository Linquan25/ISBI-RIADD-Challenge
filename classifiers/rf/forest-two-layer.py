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

from metrics import aar

plt.style.use('ggplot')


if __name__ == '__main__':
    print('aligned dlib only features')
    df = pd.read_csv('data/training_caip_contest.csv', 
                    header=None)
    

    x = np.load('data/features_non_aligned.npy')
    y = np.array(df.iloc[:, 1])
    
    train_index = np.loadtxt('data/train_index.txt', delimiter=',').astype('int')
    test_index = np.loadtxt('data/test_index.txt', delimiter=',').astype('int')
    
    x_train = x[train_index]
    y_train = y[train_index]
    y_train_group = np.clip(y_train // 10, 0, 7)
    
    x_test = x[test_index]
    y_test = y[test_index]
    y_test_group = np.clip(y_test // 10, 0, 7)

    n_estimators = 100
    min_samples_leaf = 5
    max_features = 128
    random_state = 1
    # rfc = RandomForestClassifier(n_estimators=n_estimators, 
    #                             min_samples_leaf=min_samples_leaf, 
    #                             max_features=max_features, 
    #                             random_state=random_state,
    #                             n_jobs=64,
    #                             verbose=2)
    
    # rfc.fit(x_train, y_train_group)
    
    rfc = joblib.load('data/two_layer_rf_rfr_100_5_128_1.joblib')
    
    y_train_prob = rfc.predict_proba(x_train)
    # x_train = np.concatenate([x_train, y_train_group.reshape(-1, 1)], axis=1)
    x_train = np.concatenate([x_train, y_train_prob], axis=1)

    # rfr = RandomForestRegressor(n_estimators=n_estimators, 
    #                             min_samples_leaf=min_samples_leaf, 
    #                             max_features=max_features, 
    #                             random_state=random_state,
    #                             n_jobs=64,
    #                             verbose=2)
    
    # t0 = time.time()
    # rfr.fit(x_train, y_train)
    rfr = joblib.load('data/two_layer_rf_rfc_100_5_128_1.joblib')
    # print(f'Time it took for training: {time.time() - t0:.3f} ms.')
    # dump(rfc, f'data/two_layer_rf_rfc_{n_estimators}_{min_samples_leaf}_{max_features}_{random_state}.joblib') 
    # dump(rfr, f'data/two_layer_rf_rfr_{n_estimators}_{min_samples_leaf}_{max_features}_{random_state}.joblib') 
    outc = rfc.predict_proba(x_test)
    y_pred_group = outc.argmax(axis=1)
    conf_mat = confusion_matrix(y_test_group, y_pred_group, normalize='true')
    
    plt.figure(figsize=(6, 6))
    sns.heatmap(conf_mat, annot=True, fmt='.2f', cbar=False,
                xticklabels=["< 10", "10-19", "20-29", "30-39", "40-24", "50-59", "60-69", "> 70"],
                yticklabels=["< 10", "10-19", "20-29", "30-39", "40-24", "50-59", "60-69", "> 70"])

    for i in range(8):
        y_pred_i = y_pred_group == i
        y_true_i = y_test_group == i
        print(f"Accuracy for group {i}: {accuracy_score(y_true_i, y_pred_i)}")

    print(f"Classifier accuracy: {accuracy_score(y_test_group, y_pred_group):.3f}")
    print(f"Classifier confusion matrix: \n {conf_mat}")
    print(f"Report: \n {classification_report(y_test_group, y_pred_group)}")
    # x_test = np.concatenate([x_test, y_test_group.reshape(-1, 1)], axis=1)
    x_test = np.concatenate([x_test, outc], axis=1)
    out = np.clip(rfr.predict(x_test).round(), 1, 81)
    print(f'rf MAE on validation: {np.abs(out - y_test).mean()}')
    ARR, *_, maes = aar(y_test, out)
    print(maes)
    plt.figure()
    plt.bar(["< 10", "10-19", "20-29", "30-39", "40-24", "50-59", "60-69", "> 70"], maes)
    plt.xlabel('Age goup $j$')
    plt.ylabel('$MAE^j$')
    print(f'rf AAR on validation: {ARR}')
    
    # rrf = RefinedRandomForest(clf, C = 0.01, n_prunings = 0)
    # rrf.fit(x_train, y_train)

    # out = rrf.predict_proba(x_test).argmax(axis=1)
    # print(f'rrf MAE on validation: {np.abs(out - y_test).mean()}')
    # ARR, *_ = aar(y_test, out)
    # print(f'rrf AAR on validation: {ARR}')
