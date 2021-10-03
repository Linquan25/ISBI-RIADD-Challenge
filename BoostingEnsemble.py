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
from nets import ResNet, ResNext, ViT, ResNet152, Densenet161, effNetB7, effNetB6

def loadModel(ckpt_path):
    model = ResNext()
    ckpt = torch.load(ckpt_path, map_location=torch.device('cpu'))
    new_dict = {k.replace('vit.', 'model.'): v for k, v in ckpt['state_dict'].items()}
    model.load_state_dict(new_dict)
    model.eval()
    model.cuda()
    return model

########## load models ###########
model0 = loadModel('saved_model/ISBI-WeightedBCE-ResNext101-epoch=013-val_loss=0.0892.ckpt')
model1 = loadModel('boosting/checkpoints_ResNext101/ISBI-WeightedBCE-boosting-b1-ResNext101-b0-epoch=009-val_loss=0.0870.ckpt')
model2 = loadModel('boosting/checkpoints_ResNext101/ISBI-WeightedBCE-boosting-b1-ResNext101-b1-epoch=008-val_loss=0.0862.ckpt')
model3 = loadModel('boosting/checkpoints_ResNext101/ISBI-WeightedBCE-boosting-b1-ResNext101-b2-epoch=011-val_loss=0.0885.ckpt')
model4 = loadModel('boosting/checkpoints_ResNext101/ISBI-WeightedBCE-boosting-b1-ResNext101-b3-epoch=010-val_loss=0.0870.ckpt')
model5 = loadModel('boosting/checkpoints_ResNext101/ISBI-WeightedBCE-boosting-b1-ResNext101-b4-epoch=010-val_loss=0.0832.ckpt')
model6 = loadModel('data/checkpoints/ISBI-WeightedBCE-boosting-b1-ResNext101-b5-epoch=006-val_loss=0.0839.ckpt')
model7 = loadModel('data/checkpoints/ISBI-WeightedBCE-boosting-b1-ResNext101-b6-epoch=007-val_loss=0.0882.ckpt')
#model8 = loadModel('data/checkpoints/ISBI-WeightedBCE-boosting-b1-ResNext101-b7-epoch=008-val_loss=0.0887.ckpt')
#model9 = loadModel('data/checkpoints/ISBI-WeightedBCE-boosting-b1-ResNext101-b8-epoch=010-val_loss=0.0859.ckpt')
#model10 = loadModel('data/checkpoints/ISBI-WeightedBCE-boosting-b1-ResNext101-b9-epoch=007-val_loss=0.0879.ckpt')
M = []
M.append(model0)
M.append(model1)
M.append(model2)
M.append(model3)
M.append(model4)
M.append(model5)
M.append(model6)
M.append(model7)
#M.append(model8)
#M.append(model9)
#M.append(model10)
testing_img_path = '../Test_Set/Test/'
testing_df = '../Test_Set/RFMiD_Testing_Labels.csv'
valset = ISBI_data.ISBIDataset(testing_df, testing_img_path, testing=True)
N = len(valset)
batch_size = 32
dataloader = DataLoader(valset, batch_size=batch_size, shuffle=False, 
                        num_workers=24)

outs = np.zeros((N, 29, len(M)))
labels = np.zeros((N, 29))
for n in range(len(M)):
    for i, (imgs, label, w) in enumerate(tqdm.tqdm(dataloader)):

        idx = i * batch_size
        imgs = imgs.cuda()
        out = M[n](imgs).detach().cpu().numpy()
        #out = np.round(out).astype('int').clip(1, None)
        outs[idx:idx + len(out),:, n] = out
        if n ==0:
            labels[idx:idx + len(label),:]  = label.detach().cpu().numpy()
    
sig = torch.nn.Sigmoid()
outs = sig(torch.tensor(outs)).numpy()
average = np.mean(outs, axis=2)
auc1 = roc_auc_score(labels[:,0], average[:,0])
print(f'AUC of Challenge 1: {auc1}')

auc2 = roc_auc_score(labels[:,1:], average[:,1:])
print(f'AUC of Challenge 2: {auc2}')

mAP = average_precision_score(labels[:,1:], average[:,1:])
print(f'mAP of Challenge 2: {mAP}')

C1_Score = auc1
C2_Score = mAP * 0.5 + auc2 * 0.5
final_Score =  C2_Score * 0.5 + C1_Score * 0.5
print(f'C1 Score: {C1_Score} C2 Score: {C2_Score} Final Score: {final_Score}')

# df = pd.read_csv('../Training_Set/RFMiD_Training_Labels.csv')
# predict = sig(torch.tensor(outs)).numpy()
# predict_csv = pd.DataFrame(predict,columns=df.columns[1:])
# predict_csv.index+=1
# predict_csv.to_csv('results.csv', index=True)