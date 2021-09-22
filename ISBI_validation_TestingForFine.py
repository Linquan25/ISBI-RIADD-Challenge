import numpy as np
import torch
from torchvision.models.densenet import densenet161
import tqdm
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score
import ISBI_data
import ISBI_rareset
from metrics import *
from nets import ResNet, ResNext, ViT, ResNet152, Densenet161, effNetB7, effNetB6

# ckpt = torch.load('weights/resnext50_32x4d-epoch=027-val_arr=7.96.ckpt', map_location=torch.device('cpu'))

model = ResNext()
ckpt = torch.load('saved_model/ISBI-WeightedBCE-ResNext101-epoch=013-val_loss=0.0892.ckpt', map_location=torch.device('cpu'))
#ckpt = torch.load('data/checkpoints/ISBI-WeightedBCE-ResNext101-epoch=013-val_loss=0.0892.ckpt', map_location=torch.device('cpu'))
new_dict = {k.replace('vit.', 'model.'): v for k, v in ckpt['state_dict'].items()}
model.load_state_dict(new_dict)
model.eval()
model.cuda()

testing_img_path = '../Test_Set/Test/'
testing_df = 'fineNet/test_rare.csv'
valset = ISBI_rareset.ISBIRareset(testing_df, testing_img_path, testing=True)
N = len(valset)
batch_size = 32
dataloader = DataLoader(valset, batch_size=batch_size, shuffle=False, 
                        num_workers=24)

outs = np.zeros((N, 29))
labels = np.zeros((N, 19))
for i, (imgs, label) in enumerate(tqdm.tqdm(dataloader)):

    idx = i * batch_size
    imgs = imgs.cuda()
    out = model(imgs).detach().cpu().numpy()
    #out = np.round(out).astype('int').clip(1, None)
    outs[idx:idx + len(out),:] = out
    labels[idx:idx + len(label),:]  = label.detach().cpu().numpy()
sig = torch.nn.Sigmoid()  
outs2 = outs[:,(8,  9, 10, 11, 13, 14, 15, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28)]

# rounded_illness_pred = np.round(sig(torch.tensor(outs[:,0])).numpy()).astype('int')
# illness_label = labels[:,0]

# acc = (rounded_illness_pred==illness_label).astype(int).sum()/len(illness_label)
# print(f'Accuracy: {acc}')
# illness_pred = sig(torch.tensor(outs[:,0])).numpy()
# auc1 = roc_auc_score(illness_label, illness_pred)
# print(f'AUC of Challenge 1: {auc1}')

# lossFunc = torch.nn.BCEWithLogitsLoss()
# bce = lossFunc(torch.tensor(outs), torch.tensor(labels))

# print(f'BCE: {bce}')
diseases_label = labels[:,1:]
diseases_pred = sig(torch.tensor(outs2)).numpy()
# for i in range(outs.shape[0]):
#     if outs2[i].sum()+1 != outs[i].sum():
#         outs2[i]= np.hstack((outs2[i],1))
#     else:
#         outs2[i] = np.hstack((outs2[i],0))
auc2 = roc_auc_score(diseases_label, diseases_pred)
print(f'AUC of Challenge 2: {auc2}')

mAP = average_precision_score(diseases_label, diseases_pred)
print(f'mAP of Challenge 2: {mAP}')

C2_Score = mAP * 0.5 + auc2 * 0.5
print(f'auc2: {auc2} mAP: {mAP} Final Score: {C2_Score}')


