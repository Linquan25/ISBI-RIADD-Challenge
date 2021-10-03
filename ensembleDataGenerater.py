import numpy as np
import pandas as pd
import torch
from torchvision.models.densenet import densenet161
import tqdm
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score
import ISBI_data
from metrics import *
from nets import Global_ResNext, FineNet, ResNext
#######load global net ##########
model_Gobal = Global_ResNext()
ckpt = torch.load('saved_Global/globalnet-WeightedBCE-ResNext50-epoch=005-val_loss=0.4270.ckpt', map_location=torch.device('cpu'))
new_dict = {k.replace('vit.', 'model.'): v for k, v in ckpt['state_dict'].items()}
model_Gobal.load_state_dict(new_dict)
model_Gobal.eval()
model_Gobal.cuda()
#######load Fine net ##########
model_Fine = FineNet()
ckpt = torch.load('saved_FineNet/ISBI-BCE-FineNet-epoch=241-val_loss=0.1002.ckpt', map_location=torch.device('cpu'))
new_dict = {k.replace('vit.', 'model.'): v for k, v in ckpt['state_dict'].items()}
model_Fine.load_state_dict(new_dict)
model_Fine.eval()
model_Fine.cuda()
#######load base net ##########
model_Base = ResNext()
ckpt = torch.load('saved_model/ISBI-WeightedBCE-ResNext101-epoch=013-val_loss=0.0892.ckpt', map_location=torch.device('cpu'))
new_dict = {k.replace('vit.', 'model.'): v for k, v in ckpt['state_dict'].items()}
model_Base.load_state_dict(new_dict)
model_Base.eval()
model_Base.cuda()

testing_img_path = '../Test_Set/Test/'
testing_df = '../Test_Set/RFMiD_Testing_Labels.csv'
# testing_img_path = '../Test_Set/Test/'
# testing_df = 'test_rare.csv'

valset = ISBI_data.ISBIDataset(testing_df, testing_img_path, testing=True)
N = len(valset)
batch_size = 32
dataloader = DataLoader(valset, batch_size=batch_size, shuffle=False, 
                        num_workers=24)

outs = np.zeros((N, 49))
labels = np.zeros((N, 29))
for i, (imgs, label, w) in enumerate(tqdm.tqdm(dataloader)):

    idx = i * batch_size
    imgs = imgs.cuda()
    out_global = model_Gobal(imgs).detach().cpu().numpy()
    out_Fine = model_Fine(imgs).detach().cpu().numpy()
    out_Base = model_Base(imgs).detach().cpu().numpy()
    out = np.hstack((out_global,out_Fine,out_Base))
    outs[idx:idx + len(out),:] = out
    labels[idx:idx + len(label),:]  = label.detach().cpu().numpy()
dataset_df = pd.DataFrame(outs)
dataset_df.to_csv('ensemble/ensemble_testingset.csv', index=False)

label_df = pd.DataFrame(labels)
label_df.to_csv('ensemble/ensemble_test_label.csv', index=False)
sig = torch.nn.Sigmoid()
predict = sig(torch.tensor(outs)).numpy()
dataset_sig_df = pd.DataFrame(predict)
dataset_sig_df.to_csv('ensemble/ensemble_test_sig_label.csv', index=False)