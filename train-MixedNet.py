import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch
import tqdm
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import boosting_data
import config
import data
import wandb
from nets import ResNext

plt.style.use('ggplot')
#### start from 244x244 resnext 101 -> 488x488 ->732x732 -> 976x976
def main():
    boosting_number = 4
    for n in range(3,boosting_number):
        model = ResNext()

        #data_df = pd.read_csv(config.CSV_PATH)
        #train_df, val_df = train_test_split(data_df, test_size=0.1) 
        training_img_path = '../Training_Set/Training/'
        evaluation_img_path = '../Evaluation_Set/Validation'
        train_df = '../Training_Set/RFMiD_Training_Labels.csv'
        val_df = '../Evaluation_Set/RFMiD_Validation_Labels.csv'
        weight_id = n-1
        weight_path = f'boosting/weight_b{weight_id}.csv'
        trainset = boosting_data.ISBIDataset(train_df, training_img_path, weight_path, testing=False, input_size=((n+1)*244))
        valset = boosting_data.ISBIDataset(val_df, evaluation_img_path, weight_csv = None, testing=True, input_size=((n+1)*244))
        bs = 10
        trainloader = DataLoader(trainset, batch_size=bs, shuffle=True, num_workers=20)
        valloader = DataLoader(valset, batch_size=bs, shuffle=False, num_workers=20)

        wandb.init(project='boosting-ResNext')
        wandb_logger = WandbLogger(project='boosting-ResNext')
        checkpoint_callback = ModelCheckpoint(monitor='val_loss', 
                                            #save_last = True,
                                            dirpath='data/checkpoints',
                                            #every_n_val_epochs = 10,
                                            filename=f'pyramid-boosting-ResNext101-b{n}'+'-{epoch:03d}-{val_loss:.4f}',
                                            save_top_k=3,
                                            mode='min')

        trainer = pl.Trainer(gpus=config.DEVICES, 
                            #  num_nodes=2,
                            logger=wandb_logger, 
                            log_every_n_steps=config.LOG_STEP,
                            callbacks=[checkpoint_callback], max_epochs=25)
        trainer.fit(model, trainloader, val_dataloaders=valloader)

        print("Finished Training")
        
        N = len(valset)
        batch_size = bs
        outs_valid = np.zeros((N, 29))
        labels_valid = np.zeros((N, 29))
        for i, (imgs, label, w) in enumerate(tqdm.tqdm(valloader)):
            idx = i * batch_size
            imgs = imgs.type(torch.FloatTensor)
            out = model(imgs).detach().cpu().numpy()
            outs_valid[idx:idx + len(out),:] = out
            labels_valid[idx:idx + len(label),:]  = label.detach().cpu().numpy()
        
        sig = torch.nn.Sigmoid()
        weight = np.zeros((29,))
        count = np.zeros((29,))
        rounded_valid_pred = np.round(sig(torch.tensor(outs_valid)).numpy()).astype('int')
        for i in range(labels_valid.shape[0]):
            for j in range(labels_valid.shape[1]):
                if labels_valid[i][j] != rounded_valid_pred[i][j]:
                    weight[j]+=1
                if labels_valid[i][j] == 1:
                    count[j] += 1
        #print(weight)
        weight[weight==0]=1
        weight = weight/count
        #weight = weight/min(weight)
        weight_df = pd.DataFrame(weight)
        weight_df.to_csv(f'boosting/weight_b{n}.csv', index=False)
        #print(weight)
        #print(count)
if __name__ == "__main__":
    main()
