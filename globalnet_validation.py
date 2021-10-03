import numpy as np
import torch
from torchvision.models.densenet import densenet161
import tqdm
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score
import ISBI_globalnet_Dataset
from metrics import *
from nets import Global_ResNext

model = Global_ResNext()
#ckpt = torch.load('saved_model/ISBI-WeightedBCE-ResNext101-epoch=004-val_loss=0.0868.ckpt', map_location=torch.device('cpu'))
ckpt = torch.load('data/checkpoints/globalnet-WeightedBCE-ResNext50-epoch=005-val_loss=0.4270.ckpt', map_location=torch.device('cpu'))

new_dict = {k.replace('vit.', 'model.'): v for k, v in ckpt['state_dict'].items()}
model.load_state_dict(new_dict)

model.eval()
model.cuda()

testing_img_path = '../Test_Set/Test/'
testing_df = '../Test_Set/RFMiD_Testing_Labels.csv'
# testing_img_path = '../Test_Set/Test/'
# testing_df = 'test_rare.csv'

valset = ISBI_globalnet_Dataset.ISBIDataset(testing_df, testing_img_path, testing=True)
N = len(valset)
batch_size = 32
dataloader = DataLoader(valset, batch_size=batch_size, shuffle=False, 
                        num_workers=24)

outs = np.zeros((N, ))
labels = np.zeros((N, ))
for i, (imgs, label) in enumerate(tqdm.tqdm(dataloader)):

    idx = i * batch_size
    imgs = imgs.cuda()
    out = model(imgs).detach().cpu().numpy()
    #out = np.round(out).astype('int').clip(1, None)
    outs[idx:idx + len(out)] = out.squeeze()
    labels[idx:idx + len(label)]  = label.detach().cpu().numpy()
    
sig = torch.nn.Sigmoid()

lossFunc = torch.nn.BCEWithLogitsLoss()
bce = lossFunc(torch.tensor(outs), torch.tensor(labels))
print(f'BCE: {bce}')

diseases_label = labels
diseases_pred = sig(torch.tensor(outs)).numpy()
auc = roc_auc_score(diseases_label, diseases_pred)
rounded_pred = np.round(diseases_pred).astype('int')
acc = (rounded_pred==diseases_label).astype(int).sum()/len(diseases_label)
print(f'Accuracy: {acc}')
print(f'Global auc: {auc}')
