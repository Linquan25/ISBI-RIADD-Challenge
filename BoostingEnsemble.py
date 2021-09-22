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
    return model

########## load models ###########
model1 = loadModel('saved_model/ISBI-WeightedBCE-ResNext101-epoch=013-val_loss=0.0892.ckpt')
model1.eval()
model1.cuda()
M = []
M.append(model1)

testing_img_path = '../Test_Set/Test/'
testing_df = '../Test_Set/RFMiD_Testing_Labels.csv'
valset = ISBI_data.ISBIDataset(testing_df, testing_img_path, testing=True)
N = len(valset)
batch_size = 32
dataloader = DataLoader(valset, batch_size=batch_size, shuffle=False, 
                        num_workers=24)

outs = np.zeros((N, 29, len(M)))
labels = np.zeros((N, 29, len(M)))
for n in range(len(M)):
    for i, (imgs, label, w) in enumerate(tqdm.tqdm(dataloader)):

        idx = i * batch_size
        imgs = imgs.cuda()
        out = M[0](imgs).detach().cpu().numpy()
        #out = np.round(out).astype('int').clip(1, None)
        outs[idx:idx + len(out),:, n] = out
        if n ==0:
            labels[idx:idx + len(label),:]  = label.detach().cpu().numpy()
    
sig = torch.nn.Sigmoid()
outs = sig(torch.tensor(outs)).numpy()
rounded_illness_pred = np.round(outs[:,0]).astype('int')
illness_label = labels[:,0]

acc = (rounded_illness_pred==illness_label).astype(int).sum()/len(illness_label)
print(f'Accuracy: {acc}')
illness_pred = outs[:,0]
auc1 = roc_auc_score(illness_label, illness_pred)
print(f'AUC of Challenge 1: {auc1}')

diseases_label = labels[:,1:]
diseases_pred = outs[:,1:]
auc2 = roc_auc_score(diseases_label, diseases_pred)
print(f'AUC of Challenge 2: {auc2}')

mAP = average_precision_score(diseases_label, diseases_pred)
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