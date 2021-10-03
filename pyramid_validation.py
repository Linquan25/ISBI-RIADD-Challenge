import numpy as np
import torch
import pandas as pd
from torchvision import models
from torchvision.models.densenet import densenet161
import tqdm
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score
import boosting_data
from metrics import *
from nets import ResNet, ResNext, ViT, ResNet152, Densenet161, effNetB7, effNetB6

def loadModel(ckpt_path, n):
    model = ResNext()
    ckpt = torch.load(ckpt_path, map_location=torch.device('cpu'))
    new_dict = {k.replace('vit.', 'model.'): v for k, v in ckpt['state_dict'].items()}
    model.load_state_dict(new_dict)
    model.eval()
    model.cuda()
    testing_img_path = '../Test_Set/Test/'
    testing_df = '../Test_Set/RFMiD_Testing_Labels.csv'
    valset = boosting_data.ISBIDataset(testing_df, testing_img_path, weight_csv = None, testing=True, input_size=((n+1)*244))
    N = len(valset)
    batch_size = 2
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
    return outs, labels

########## load models ###########
out0, labels = loadModel('saved_model/ISBI-WeightedBCE-ResNext101-epoch=013-val_loss=0.0892.ckpt', 0)
out1, _ = loadModel('boosting/checkpoints_MixedNet/pyramid-boosting-ResNext101-b1-epoch=018-val_loss=0.0830.ckpt',1)
out2, _ = loadModel('boosting/checkpoints_MixedNet/pyramid-boosting-ResNext101-b2-epoch=020-val_loss=0.0829.ckpt',2)
out3, _ = loadModel('boosting/checkpoints_MixedNet/pyramid-boosting-ResNext101-b3-epoch=019-val_loss=0.0805.ckpt',3)

outs = (out0+out1+out2+out3)/4
    
sig = torch.nn.Sigmoid()
outs = sig(torch.tensor(outs)).numpy()
auc1 = roc_auc_score(labels[:,0], outs[:,0])
print(f'AUC of Challenge 1: {auc1}')

auc2 = roc_auc_score(labels[:,1:], outs[:,1:])
print(f'AUC of Challenge 2: {auc2}')

mAP = average_precision_score(labels[:,1:], outs[:,1:])
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