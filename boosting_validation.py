import numpy as np
import torch
from torchvision.models.densenet import densenet161
import tqdm
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score
import boosting_data
from metrics import *
from nets import ResNet, ResNext, ViT, ResNet152, Densenet161, effNetB7, effNetB6

# ckpt = torch.load('weights/resnext50_32x4d-epoch=027-val_arr=7.96.ckpt', map_location=torch.device('cpu'))

model = ResNext()
#ckpt = torch.load('saved_model/ISBI-WeightedBCE-ResNext101-epoch=013-val_loss=0.0892.ckpt', map_location=torch.device('cpu'))
ckpt = torch.load('data/checkpoints/ISBI-WeightedBCE-boosting-b1-ResNext101-epoch=005-val_loss=0.0823.ckpt', map_location=torch.device('cpu'))
new_dict = {k.replace('vit.', 'model.'): v for k, v in ckpt['state_dict'].items()}
model.load_state_dict(new_dict)
model.eval()
model.cuda()

valid_img_path = '../Evaluation_Set/Validation/'
valid_df = '../Evaluation_Set/RFMiD_Validation_Labels.csv'
testing_img_path = '../Test_Set/Test/'
testing_df = '../Test_Set/RFMiD_Testing_Labels.csv'
test_set = boosting_data.ISBIDataset(testing_df, testing_img_path, weight_csv=None, testing=True)
valid_set = boosting_data.ISBIDataset(valid_df, valid_img_path, weight_csv=None, testing=True)
N = len(test_set)
batch_size = 32
dataloader_valid = DataLoader(valid_set, batch_size=batch_size, shuffle=False, 
                        num_workers=24)
dataloader_test = DataLoader(test_set, batch_size=batch_size, shuffle=False, 
                        num_workers=24)

outs_valid = np.zeros((N, 29))
labels_valid = np.zeros((N, 29))
outs_test = np.zeros((N, 29))
labels_test = np.zeros((N, 29))
############ valid Set ############
for i, (imgs, label, w) in enumerate(tqdm.tqdm(dataloader_valid)):

    idx = i * batch_size
    imgs = imgs.cuda()
    out = model(imgs).detach().cpu().numpy()
    outs_valid[idx:idx + len(out),:] = out
    labels_valid[idx:idx + len(label),:]  = label.detach().cpu().numpy()
    
sig = torch.nn.Sigmoid()
# rounded_illness_pred = np.round(sig(torch.tensor(outs_valid[:,0])).numpy()).astype('int')
# illness_label = labels_valid[:,0]
# acc = (rounded_illness_pred==illness_label).astype(int).sum()/len(illness_label)
# print(f'Accuracy: {acc}')
illness_label_valid = labels_valid[:,0]
illness_pred_valid = sig(torch.tensor(outs_valid[:,0])).numpy()
auc1_valid = roc_auc_score(illness_label_valid, illness_pred_valid)
print(f'AUC of Challenge 1: {auc1_valid}')
diseases_label_valid = labels_valid[:,1:]
diseases_pred_valid = sig(torch.tensor(outs_valid[:,1:])).numpy()
auc2_valid = roc_auc_score(diseases_label_valid, diseases_pred_valid)
print(f'AUC of Challenge 2: {auc2_valid}')
mAP_valid = average_precision_score(diseases_label_valid, diseases_pred_valid)
print(f'mAP of Challenge 2: {mAP_valid}')
C1_Score_valid = auc1_valid
C2_Score_valid = mAP_valid * 0.5 + auc2_valid * 0.5
final_Score_valid =  C2_Score_valid * 0.5 + C1_Score_valid * 0.5
print(f'Valid set C1 Score: {C1_Score_valid} C2 Score: {C2_Score_valid} Final Score: {final_Score_valid}')

weight = np.zeros((29,))
rounded_valid_pred = np.round(sig(torch.tensor(outs_valid)).numpy()).astype('int')
for i in range(labels_valid.shape[0]):
    for j in range(labels_valid.shape[1]):
        if labels_valid[i][j] != rounded_valid_pred[i][j]:
            weight[j]+=1
weight[weight==0]=1
weight_df = pd.DataFrame(weight)
weight_df.to_csv('boosting/weight_b3.csv', index=False)
print(weight)
# ########## Test Set ############
# for i, (imgs, label, w) in enumerate(tqdm.tqdm(dataloader_test)):

#     idx = i * batch_size
#     imgs = imgs.cuda()
#     out = model(imgs).detach().cpu().numpy()
#     out = np.round(out).astype('int').clip(1, None)
#     outs_test[idx:idx + len(out),:] = out
#     labels_test[idx:idx + len(label),:]  = label.detach().cpu().numpy()
    
# illness_label = labels_test[:,0]
# illness_pred = sig(torch.tensor(outs_test[:,0])).numpy()
# auc1 = roc_auc_score(illness_label, illness_pred)
# print(f'AUC of Challenge 1: {auc1}')
# diseases_label = labels_test[:,1:]
# diseases_pred = sig(torch.tensor(outs_test[:,1:])).numpy()
# print("omax = ", np.amax(outs_test), "pmax = ",  np.amax(diseases_pred))
# auc2 = roc_auc_score(diseases_label, diseases_pred)
# print(f'AUC of Challenge 2: {auc2}')
# mAP = average_precision_score(diseases_label, diseases_pred)
# print(f'mAP of Challenge 2: {mAP}')
# C1_Score = auc1
# C2_Score = mAP * 0.5 + auc2 * 0.5
# final_Score =  C2_Score * 0.5 + C1_Score * 0.5
# print(f'Test set C1 Score: {C1_Score} C2 Score: {C2_Score} Final Score: {final_Score}')