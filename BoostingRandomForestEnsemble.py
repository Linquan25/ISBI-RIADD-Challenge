import numpy as np
import torch
import pandas as pd
from torchvision import models
from torchvision.models.densenet import densenet161
import tqdm
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score
import ISBI_data
import ISBI_rareset
from metrics import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from nets import ResNet, ResNext, ViT, ResNet152, Densenet161, effNetB7, effNetB6

def loadModel(ckpt_path):
    model = ResNext()
    ckpt = torch.load(ckpt_path, map_location=torch.device('cpu'))
    new_dict = {k.replace('vit.', 'model.'): v for k, v in ckpt['state_dict'].items()}
    model.load_state_dict(new_dict)
    model.eval()
    model.model.fc = torch.nn.Identity()
    model.cuda()
    return model

########## load models ###########
model0 = loadModel('saved_model/ISBI-WeightedBCE-ResNext101-epoch=013-val_loss=0.0892.ckpt')
model1 = loadModel('data/checkpoints/ISBI-WeightedBCE-boosting-b1-ResNext101-b0-epoch=009-val_loss=0.0870.ckpt')
model2 = loadModel('data/checkpoints/ISBI-WeightedBCE-boosting-b1-ResNext101-b1-epoch=008-val_loss=0.0862.ckpt')
model3 = loadModel('data/checkpoints/ISBI-WeightedBCE-boosting-b1-ResNext101-b2-epoch=011-val_loss=0.0885.ckpt')
model4 = loadModel('data/checkpoints/ISBI-WeightedBCE-boosting-b1-ResNext101-b3-epoch=010-val_loss=0.0870.ckpt')
model5 = loadModel('data/checkpoints/ISBI-WeightedBCE-boosting-b1-ResNext101-b4-epoch=010-val_loss=0.0832.ckpt')
M = []
M.append(model0)
M.append(model1)
M.append(model2)
M.append(model3)
M.append(model4)
M.append(model5)
##################### load train dataset for RF training #################
train_img_path = '../Training_Set/Training/'
train_df = '../Training_Set/RFMiD_Training_Labels.csv'
train_set = ISBI_data.ISBIDataset(train_df, train_img_path, testing=True, reweight=False)
N = len(train_set)
batch_size = 32
train_dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=False, 
                        num_workers=24)

outs_train = np.zeros((N, 2048, len(M)))
labels_train = np.zeros((N, 29))
for n in range(len(M)):
    for i, (imgs, label) in enumerate(tqdm.tqdm(train_dataloader)):

        idx = i * batch_size
        imgs = imgs.cuda()
        out = M[n](imgs).detach().cpu().numpy()
        #out = np.round(out).astype('int').clip(1, None)
        outs_train[idx:idx + len(out),:, n] = out
        if n ==0:
            labels_train[idx:idx + len(label),:]  = label.detach().cpu().numpy()
outs_train = torch.tensor(outs_train).numpy()  
##################### load valid dataset for RF training #################
valid_img_path = '../Evaluation_Set/Validation/'
valid_df = '../Evaluation_Set/RFMiD_Validation_Labels.csv'
valset = ISBI_data.ISBIDataset(valid_df, valid_img_path, testing=True, reweight=False)
N = len(valset)
batch_size = 32
valid_dataloader = DataLoader(valset, batch_size=batch_size, shuffle=False, 
                        num_workers=24)

outs_valid = np.zeros((N, 2048, len(M)))
labels_valid = np.zeros((N, 29))
for n in range(len(M)):
    for i, (imgs, label) in enumerate(tqdm.tqdm(valid_dataloader)):

        idx = i * batch_size
        imgs = imgs.cuda()
        out = M[n](imgs).detach().cpu().numpy()
        #out = np.round(out).astype('int').clip(1, None)
        outs_valid[idx:idx + len(out),:, n] = out
        if n ==0:
            labels_valid[idx:idx + len(label),:]  = label.detach().cpu().numpy()
outs_valid = torch.tensor(outs_valid).numpy()            
##################### load test dataset for RF training #################
testing_img_path = '../Test_Set/Test/'
testing_df = '../Test_Set/RFMiD_Testing_Labels.csv'
testset = ISBI_data.ISBIDataset(testing_df, testing_img_path, testing=True, reweight=False)
N = len(testset)
batch_size = 32
test_dataloader = DataLoader(testset, batch_size=batch_size, shuffle=False, 
                        num_workers=24)

outs_test = np.zeros((N, 2048, len(M)))
labels_test = np.zeros((N, 29))
for n in range(len(M)):
    for i, (imgs, label) in enumerate(tqdm.tqdm(test_dataloader)):

        idx = i * batch_size
        imgs = imgs.cuda()
        out = M[n](imgs).detach().cpu().numpy()
        outs_test[idx:idx + len(out),:, n] = out
        if n ==0:
            labels_test[idx:idx + len(label),:]  = label.detach().cpu().numpy()
#outs_test = sig(torch.tensor(outs_test)).numpy()
outs_test = torch.tensor(outs_test).numpy()

feature_train = outs_train.reshape((len(outs_train),-1))
feature_valid = outs_valid.reshape((len(outs_valid),-1))
training_features = np.vstack((feature_train,feature_valid))
testing_features = outs_test.reshape((len(outs_test),-1))
training_labels = np.vstack((labels_train,labels_valid))
np.save('ensemble/features/training_features.npy', training_features)
np.save('ensemble/features/testing_features.npy', testing_features)
np.save('ensemble/features/training_labels.npy', training_labels)
np.save('ensemble/features/testing_labels.npy', labels_test)
# clf = RandomForestClassifier(n_estimators=20, criterion="gini")
# clf.fit(outs.reshape((len(labels),-1)), labels)
# pre = clf.predict_proba(outs.reshape((len(labels),-1)))
# pre = pre[:,1]
# auc1 = roc_auc_score(labels[:,0], pre[:,0])
# print(f'AUC of Challenge 1: {auc1}')

# auc2 = roc_auc_score(labels[:,1:], pre[:,1:])
# print(f'AUC of Challenge 2: {auc2}')

# mAP = average_precision_score(labels[:,1:], pre[:,1:])
# print(f'mAP of Challenge 2: {mAP}')

# C1_Score = auc1
# C2_Score = mAP * 0.5 + auc2 * 0.5
# final_Score =  C2_Score * 0.5 + C1_Score * 0.5
# print(f'C1 Score: {C1_Score} C2 Score: {C2_Score} Final Score: {final_Score}')

# df = pd.read_csv('../Training_Set/RFMiD_Training_Labels.csv')
# predict = sig(torch.tensor(outs)).numpy()
# predict_csv = pd.DataFrame(predict,columns=df.columns[1:])
# predict_csv.index+=1
# predict_csv.to_csv('results.csv', index=True)