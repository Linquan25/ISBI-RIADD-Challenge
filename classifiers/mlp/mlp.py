import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from metrics import aar
from model import MLP


def main():
    print('aligned dlib only features')
    df = pd.read_csv('data/training_caip_contest.csv', 
                    header=None)
    
    # x = np.load('data/features_dlib_aligned.npy')
    # x = np.load('data/features_cr.npy')
    # x = np.load('data/features_vggface.npy')
    x = np.load('data/features_non_aligned.npy')
    # x = np.concatenate([np.load('data/features_dlib_non_aligned.npy'),
    #             np.load('data/features_non_aligned.npy')], axis=1)
    y = np.array(df.iloc[:, 1])

    test_index = np.loadtxt('data/train_index.txt', delimiter=',').astype('int')
    
    x_test = torch.Tensor(x[test_index])
    y_test = torch.Tensor(y[test_index])
    testset = TensorDataset(x_test, y_test) 
    batch_size = 1024
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)
    
    model = MLP()
    model = model.load_from_checkpoint('data/checkpoints/MSE-Loss-MLP-epoch=053-val_aar=8.31.ckpt')
    model = model.eval()
    model.cuda()
    N = len(testset)
    
    outs = np.zeros(N).astype('int')
    ages = np.zeros(N).astype('int')
    for i, (imgs, age) in enumerate(tqdm(testloader)):
        idx = i * batch_size
        imgs = imgs.cuda()
        out = (model(imgs) * 81).detach().cpu().numpy()
        # print(out[:4])
        age = age.detach().cpu().numpy()
        out = np.round(out).astype('int').clip(1, None)
        age = np.round(age).astype('int')
        outs[idx:min(N, idx + batch_size)] = out.squeeze()
        ages[idx:min(N, idx + batch_size)] = age.squeeze()

    AAR = aar(outs, ages)
    print(f'MLP AAR on validation: {AAR}')

if __name__ == '__main__':
    main()
