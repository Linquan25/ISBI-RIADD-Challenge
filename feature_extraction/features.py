import numpy as np
import torch
import tqdm
from torch.utils.data import DataLoader

import data
from model import ResNet, ResNext

model = ResNext()
# model = ResNet()

# for ResNet
# ckpt = torch.load('data/checkpoints/AAR-Loss-ResNet-epoch=011-val_aar=7.90.ckpt', map_location='cpu')

# for ResNext
# ckpt = torch.load('data/checkpoints/AAR-Loss-ResNext-epoch=025-val_arr=0.00.ckpt', map_location='cpu')
ckpt = torch.load('data/checkpoints/MSE-Loss-ResNext-epoch=024-val_aar=8.01.ckpt', map_location='cpu')
# Linquan's
# ckpt = torch.load('data/checkpoints/Aligned-ResNext-epoch=014-val_arr=7.65.ckpt', map_location='cpu')
# ckpt = torch.load('data/checkpoints/AAR-Loss-ResNet-epoch=027-val_aar=8.07.ckpt', map_location='cpu')


# CR
# ckpt = torch.load('data/checkpoints/CR-ResNext-epoch=055-val_arr=7.68.ckpt', map_location='cpu')
new_dict = {k.replace('vit.', 'model.'): v for k, v in ckpt['state_dict'].items()}
model.load_state_dict(new_dict)



model.eval()
# For features
model.model.fc = torch.nn.Identity()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)

dataset = data.GTADataset('data/training_caip_contest.csv',
                          'data/training_caip_contest',
                          transform=data.EVAL_TRANSFORMS,
                          return_paths=True)
N = len(dataset)
batch_size = 128
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, 
                        num_workers=24)

acc = np.zeros((N, 2048))
for i, (imgs, _, paths) in enumerate(tqdm.tqdm(dataloader)):
    idx = i * batch_size
    imgs = imgs.to(device)
    feats = model(imgs)
    feats = feats.detach().cpu().numpy()
    acc[idx:min(N, idx + batch_size)] = feats
np.save('data/features_pred_resnext_mse.npy', acc)
