import numpy as np
import torch
from torchvision.models.densenet import densenet161
import tqdm
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score
import ISBI_rareset
from metrics import *
from nets import ResNet, ResNext, ViT, ResNet152, Densenet161, effNetB7, effNetB6, FineNet

model = FineNet()
#ckpt = torch.load('saved_model/ISBI-WeightedBCE-ResNext101-epoch=004-val_loss=0.0868.ckpt', map_location=torch.device('cpu'))
ckpt = torch.load('data/checkpoints/ISBI-BCE-FineNet-epoch=241-val_loss=0.1002.ckpt', map_location=torch.device('cpu'))

new_dict = {k.replace('vit.', 'model.'): v for k, v in ckpt['state_dict'].items()}
model.load_state_dict(new_dict)

model.eval()
model.cuda()

testing_img_path = '../Test_Set/Test/'
testing_df = '../Test_Set/RFMiD_Testing_Labels.csv'
# testing_img_path = '../Test_Set/Test/'
# testing_df = 'test_rare.csv'

valset = ISBI_rareset.ISBIRareset(testing_df, testing_img_path, testing=True)
N = len(valset)
batch_size = 32
dataloader = DataLoader(valset, batch_size=batch_size, shuffle=False, 
                        num_workers=24)

outs = np.zeros((N, 19))
labels = np.zeros((N, 19))
for i, (imgs, label) in enumerate(tqdm.tqdm(dataloader)):

    idx = i * batch_size
    imgs = imgs.cuda()
    out = model(imgs).detach().cpu().numpy()
    #out = np.round(out).astype('int').clip(1, None)
    outs[idx:idx + len(out),:] = out
    labels[idx:idx + len(label),:]  = label.detach().cpu().numpy()
    
sig = torch.nn.Sigmoid()

lossFunc = torch.nn.BCEWithLogitsLoss()
bce = lossFunc(torch.tensor(outs), torch.tensor(labels))
print(f'BCE: {bce}')

diseases_label = labels
diseases_pred = sig(torch.tensor(outs)).numpy()
auc = roc_auc_score(diseases_label, diseases_pred)

mAP = average_precision_score(diseases_label, diseases_pred)

FineScore_Score = mAP * 0.5 + auc * 0.5
print(f'FineNet auc: {auc} mAP: {mAP} Final Score: {FineScore_Score}')
