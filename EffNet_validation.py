import numpy as np
import torch
import pandas as pd
from torchvision.models.densenet import densenet161
import tqdm
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score
import EffNet_dataset
from metrics import *
from nets import effNetB6


model = effNetB6()
#ckpt = torch.load('saved_model/ISBI-WeightedBCE-effNetB6-epoch=019-val_loss=0.0876.ckpt', map_location=torch.device('cpu'))
ckpt = torch.load('data/checkpoints/ISBI-WeightedBCE-effNetB6-input960-epoch=006-val_loss=0.0865.ckpt', map_location=torch.device('cpu'))
new_dict = {k.replace('vit.', 'model.'): v for k, v in ckpt['state_dict'].items()}
model.load_state_dict(new_dict)
model.eval()
model.cuda()

testing_img_path = '../Test_Set/Test/'
testing_df = '../Test_Set/RFMiD_Testing_Labels.csv'
valset = EffNet_dataset.ISBIDataset(testing_df, testing_img_path, testing=True)
N = len(valset)
batch_size = 4
dataloader = DataLoader(valset, batch_size=batch_size, shuffle=False, 
                        num_workers=24)

outs = np.zeros((N, 29))
labels = np.zeros((N, 29))
for i, (imgs, label, w) in enumerate(tqdm.tqdm(dataloader)):

    idx = i * batch_size
    imgs = imgs.cuda()
    out = model(imgs).detach().cpu().numpy()
    #out = np.round(out).astype('int').clip(1, None)
    outs[idx:idx + len(out),:] = out
    labels[idx:idx + len(label),:]  = label.detach().cpu().numpy()
    
sig = torch.nn.Sigmoid()

def postPro(outs):
    for i in range(outs.shape[0]):
        if outs[i][0]<0.1:
            outs[i,1:] = outs[i,1:] * outs[i][0]
    return outs
def postPro2(outs):
    for i in range(outs.shape[0]):
        if outs[i][0]<0.5:
            for j in range(outs.shape[1]-1):
                if outs[i][j+1] > 0.6:
                    outs[i][0] = outs[i][0]/outs[i][j+1]
    return outs
outs = sig(torch.tensor(outs)).numpy()
outs = postPro2(outs)
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