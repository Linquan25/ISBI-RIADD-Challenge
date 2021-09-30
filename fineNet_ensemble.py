import numpy as np
import torch
from torchvision.models.densenet import densenet161
import tqdm
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score
import ISBI_rareset
import ISBI_data
from metrics import *
from nets import ResNet, ResNext, ViT, ResNet152, Densenet161, effNetB7, effNetB6, FineNet

model_fine = FineNet()
ckpt = torch.load('data/checkpoints/ISBI-BCE-FineNet-epoch=021-val_loss=0.0907.ckpt', map_location=torch.device('cpu'))
new_dict = {k.replace('vit.', 'model.'): v for k, v in ckpt['state_dict'].items()}
model_fine.load_state_dict(new_dict)
model_fine.eval()
model_fine.cuda()

model = ResNext()
ckpt = torch.load('saved_model/ISBI-WeightedBCE-ResNext101-epoch=013-val_loss=0.0892.ckpt', map_location=torch.device('cpu'))
new_dict = {k.replace('vit.', 'model.'): v for k, v in ckpt['state_dict'].items()}
model.load_state_dict(new_dict)
model.eval()
model.cuda()

testing_img_path = '../Test_Set/Test/'
testing_df = '../Test_Set/RFMiD_Testing_Labels.csv'
############ fine net results ########################
valset_fine = ISBI_rareset.ISBIRareset(testing_df, testing_img_path, testing=True, reweight=False)
N = len(valset_fine)
batch_size = 32
dataloader_fine = DataLoader(valset_fine, batch_size=batch_size, shuffle=False, 
                        num_workers=24)
outs_fine = np.zeros((N, 19))
labels_fine = np.zeros((N, 19))
for i, (imgs, label) in enumerate(tqdm.tqdm(dataloader_fine)):

    idx = i * batch_size
    imgs = imgs.cuda()
    out = model_fine(imgs).detach().cpu().numpy()
    #out = np.round(out).astype('int').clip(1, None)
    outs_fine[idx:idx + len(out),:] = out
    labels_fine[idx:idx + len(label),:]  = label.detach().cpu().numpy()
############ base net results ########################
valset = ISBI_data.ISBIDataset(testing_df, testing_img_path, testing=True, reweight=False)
N = len(valset)
batch_size = 32
dataloader = DataLoader(valset, batch_size=batch_size, shuffle=False, 
                        num_workers=24)

outs = np.zeros((N, 29))
labels = np.zeros((N, 29))
for i, (imgs, label) in enumerate(tqdm.tqdm(dataloader)):

    idx = i * batch_size
    imgs = imgs.cuda()
    out = model(imgs).detach().cpu().numpy()
    #out = np.round(out).astype('int').clip(1, None)
    outs[idx:idx + len(out),:] = out
    labels[idx:idx + len(label),:]  = label.detach().cpu().numpy() 
sig = torch.nn.Sigmoid()
outs_fine = sig(torch.tensor(outs_fine)).numpy()
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
