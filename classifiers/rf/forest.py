import time

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

from metrics import aar


def main():
    print('aligned dlib only features')
    df = pd.read_csv('data/training_caip_contest.csv', 
                    header=None)
    
    # x = np.load('data/features_dlib_aligned.npy')
    # x = np.load('data/features_cr.npy')
    # x = np.load('data/features_vggface.npy')
    x = np.load('data/features_pred_resnext_aar.npy')
    # x = np.concatenate([np.load('data/features_dlib_non_aligned.npy'),
    #             np.load('data/features_non_aligned.npy')], axis=1)
    y = np.array(df.iloc[:, 1])
    
    train_index = np.loadtxt('data/train_index.txt', delimiter=',').astype('int')
    test_index = np.loadtxt('data/test_index.txt', delimiter=',').astype('int')
    
    x_train = x[train_index]
    y_train = y[train_index]
    
    x_test = x[test_index]
    y_test = y[test_index]
    
    # x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, 
    #                                                     random_state=1)

    n_estimators = 100
    min_samples_leaf = 5
    max_features = 128
    random_state = 1
    rf = RandomForestRegressor(n_estimators=n_estimators, 
                                min_samples_leaf=min_samples_leaf, 
                                max_features=max_features, 
                                random_state=random_state,
                                n_jobs=64,
                                verbose=2)
    
    t0 = time.time()
    rf.fit(x_train, y_train)
    print(f'Time it took for training: {time.time() - t0:.3f} ms.')
    dump(rf, f'data/rf_vggface2_{n_estimators}_{min_samples_leaf}_{max_features}_{random_state}.joblib') 
    out = np.clip(rf.predict(x_test).round(), 1, 81)
    print(f'rf MAE on validation: {np.abs(out - y_test).mean()}')
    ARR, *_ = aar(y_test, out)
    print(f'rf AAR on validation: {ARR}')
    
    # rrf = RefinedRandomForest(clf, C = 0.01, n_prunings = 0)
    # rrf.fit(x_train, y_train)

    # out = rrf.predict_proba(x_test).argmax(axis=1)
    # print(f'rrf MAE on validation: {np.abs(out - y_test).mean()}')
    # ARR, *_ = aar(y_test, out)
    # print(f'rrf AAR on validation: {ARR}')

if __name__ == '__main__':
    main()
