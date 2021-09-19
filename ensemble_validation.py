import numpy as np
import torch
from torchvision.models.densenet import densenet161
import tqdm
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score
import ensemble_data
from metrics import *
from nets import ensembleNet

model = ensembleNet()
ckpt = torch.load('data/checkpoints/ensemble-epoch=002-val_loss=0.10717.ckpt', map_location=torch.device('cpu'))
new_dict = {k.replace('vit.', 'model.'): v for k, v in ckpt['state_dict'].items()}
model.load_state_dict(new_dict)
model.eval()
model.cuda()

testingset_path = 'ensemble/ensemble_testingset_sig.csv'
testing_df = 'ensemble/ensemble_testing_label.csv'
valset = ensemble_data.ensembleDataset(testing_df, testingset_path)
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
    outs[idx:idx + len(out),:] = out
    labels[idx:idx + len(label),:]  = label.detach().cpu().numpy()
sig = torch.nn.Sigmoid()
rounded_illness_pred = np.round(sig(torch.tensor(outs[:,0])).numpy()).astype('int')
illness_label = labels[:,0]

acc = (rounded_illness_pred==illness_label).astype(int).sum()/len(illness_label)
print(f'Accuracy: {acc}')
illness_pred = sig(torch.tensor(outs[:,0])).numpy()
auc1 = roc_auc_score(illness_label, illness_pred)
print(f'AUC of Challenge 1: {auc1}')

lossFunc = torch.nn.BCEWithLogitsLoss()
bce = lossFunc(torch.tensor(outs[:,1:]), torch.tensor(labels[:,1:]))

print(f'BCE: {bce}')

diseases_label = labels[:,1:]
diseases_pred = sig(torch.tensor(outs[:,1:])).numpy()
print("omax = ", np.amax(outs), "pmax = ",  np.amax(diseases_pred))
auc2 = roc_auc_score(diseases_label, diseases_pred)
print(f'AUC of Challenge 2: {auc2}')

mAP = average_precision_score(diseases_label, diseases_pred)
print(f'mAP of Challenge 2: {mAP}')

C1_Score = auc1
C2_Score = mAP * 0.5 + auc2 * 0.5
final_Score =  C2_Score * 0.5 + C1_Score * 0.5
print(f'C1 Score: {C1_Score} C2 Score: {C2_Score} Final Score: {final_Score}')
