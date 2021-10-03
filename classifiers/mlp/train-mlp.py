import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

import config
import data
import wandb
from model import MLP

plt.style.use('ggplot')

def main():
    

    model = MLP()
    
    df = pd.read_csv('data/training_caip_contest.csv', header=None)
    

    x = torch.tensor(np.load('data/features_non_aligned.npy'))
    y = torch.tensor(np.array(df.iloc[:, 1]))
    
    train_index = np.loadtxt('data/train_index.txt', delimiter=',').astype('int')

    x_train = x[train_index]
    y_train = y[train_index]
    
    dataset = TensorDataset(x_train, y_train) 
    val_size = int(len(dataset) * .1)
    train_size = len(dataset) - val_size
    trainset, valset = torch.utils.data.random_split(dataset, [train_size, val_size])

    testset = TensorDataset(x_train, y_train) 

    trainloader = DataLoader(trainset, batch_size=1024, shuffle=True)
    valloader = DataLoader(valset, batch_size=1024, shuffle=False)
    testloader = DataLoader(testset, batch_size=1024, shuffle=False)

    wandb.init(project='GTA-MSE-MLP')
    wandb_logger = WandbLogger(project='GTA-MSE-MLP')
    checkpoint_callback = ModelCheckpoint(monitor='val_aar', 
                                          dirpath='data/checkpoints',
                                          filename='MSE-Loss-MLP-{epoch:03d}-{val_aar:.2f}',
                                          save_top_k=3,
                                          mode='max')

    trainer = pl.Trainer(gpus=config.DEVICES, 
                        #  num_nodes=2,
                         logger=wandb_logger, 
                         log_every_n_steps=config.LOG_STEP,
                         callbacks=[checkpoint_callback])
    trainer.fit(model, trainloader, val_dataloaders=valloader)
    res = trainer.test(model, testloader, verbose=True)
    print("Finished Training")
    print(res)


if __name__ == "__main__":
    main()
