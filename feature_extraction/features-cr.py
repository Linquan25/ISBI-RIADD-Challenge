import numpy as np
import torch
import tqdm
from torch.utils.data import DataLoader

import data
from model import ResNext, TSNet

model = TSNet()

# for ResNext
# ckpt = torch.load('data/checkpoints/resnext50_32x4d-epoch=019-val_arr=8.01.ckpt', map_location='cpu')
# Linquan's
# ckpt = torch.load('data/checkpoints/Aligned-ResNext-epoch=014-val_arr=7.65.ckpt', map_location='cpu')



# CR
ckpt = torch.load('data/checkpoints/CR-ResNext-epoch=055-val_arr=7.68.ckpt', map_location='cpu')
model.load_state_dict(ckpt['state_dict'])



model.eval()
# For features
model.regression = torch.nn.Identity()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)

dataset = data.GTADataset('data/training_caip_contest.csv',
                          'data/training_caip_contest',
                          transform=data.EVAL_TRANSFORMS,
                          return_paths=True)
N = len(dataset)
batch_size = 256
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

acc = np.zeros((N, 2048))
for i, (imgs, _, paths) in enumerate(tqdm.tqdm(dataloader)):
    idx = i * batch_size
    imgs = imgs.to(device)
    _, outs = model(imgs)
    outs = outs.detach().cpu().numpy()
    acc[idx:min(N, idx + batch_size)] = outs
np.save('data/features_cr.npy', acc)
