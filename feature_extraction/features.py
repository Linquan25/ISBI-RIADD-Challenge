import numpy as np
import torch
import tqdm
from torch.utils.data import DataLoader
import sys
sys.path.append("..")
#import data
from nets import ResNext, effNetB6
from ISBI_data import ISBIDataset

model = ResNext()
# model = ResNet()

# for ResNet
# ckpt = torch.load('data/checkpoints/AAR-Loss-ResNet-epoch=011-val_aar=7.90.ckpt', map_location='cpu')

# for ResNext
# ckpt = torch.load('data/checkpoints/AAR-Loss-ResNext-epoch=025-val_arr=0.00.ckpt', map_location='cpu')
ckpt = torch.load('../saved_model/ISBI-WeightedBCE-ResNext101-epoch=013-val_loss=0.0892.ckpt', map_location='cpu')
# Linquan's
# ckpt = torch.load('data/checkpoints/Aligned-ResNext-epoch=014-val_arr=7.65.ckpt', map_location='cpu')
# ckpt = torch.load('data/checkpoints/AAR-Loss-ResNet-epoch=027-val_aar=8.07.ckpt', map_location='cpu')


# CR
# ckpt = torch.load('data/checkpoints/CR-ResNext-epoch=055-val_arr=7.68.ckpt', map_location='cpu')
new_dict = {k.replace('vit.', 'model.'): v for k, v in ckpt['state_dict'].items()}
model.load_state_dict(new_dict)

model.eval()
# For features
# ResNext->fc EffNet->_fc
model.model.fc = torch.nn.Identity()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)

training_img_path = '../../Training_Set/Training/'
evaluation_img_path = '../../Evaluation_Set/Validation'
testing_img_path = '../../Test_Set/Test/'
train_df = '../../Training_Set/RFMiD_Training_Labels.csv'
val_df = '../../Evaluation_Set/RFMiD_Validation_Labels.csv'
test_df = '../../Test_Set/RFMiD_Testing_Labels.csv'
#dataset = ISBIDataset(train_df, training_img_path, testing=True, reweight=False)
#dataset = ISBIDataset(val_df, evaluation_img_path, testing=True, reweight=False)
dataset = ISBIDataset(test_df, testing_img_path, testing=True, reweight=False)
N = len(dataset)
batch_size = 16
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, 
                        num_workers=24)

outs = np.zeros((N, 2048)) #EffNetB6->2304 ResNext->2048
labels = np.zeros((N, 29))
for i, (imgs, label) in enumerate(tqdm.tqdm(dataloader)):

    idx = i * batch_size
    imgs = imgs.cuda()
    out = model(imgs).detach().cpu().numpy()
    outs[idx:idx + len(out),:] = out
    labels[idx:idx + len(label),:]  = label.detach().cpu().numpy()
np.save('ResNext101/features_ResNext101_test.npy', outs)
np.save('ResNext101/labels_ResNext101_test.npy', labels)
