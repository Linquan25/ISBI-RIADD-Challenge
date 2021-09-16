import matplotlib.pyplot as plt
import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

import config
import data
from nets import ViT

plt.style.use('ggplot')

def main():

    model = ViT()

    data_df = pd.read_csv(config.CSV_PATH)
    train_df, val_df = train_test_split(data_df, test_size=0.1)
    
    if config.ALIGN:
        img_path = config.IMG_PATH + '_aligned'
    else:
        img_path = config.IMG_PATH      

    trainset = data.GTADataset(train_df, img_path, transform=data.TRAIN_TRANSFORMS, align=False)
    valset = data.GTADataset(val_df, img_path, transform=data.EVAL_TRANSFORMS, align=False)

    trainloader = DataLoader(trainset, batch_size=128, shuffle=True, num_workers=8)
    valloader = DataLoader(valset, batch_size=128, shuffle=False, num_workers=8)

    wandb_logger = WandbLogger(project='GTA-VIT-Aligned')
    checkpoint_callback = ModelCheckpoint(monitor='val_arr', 
                                          dirpath='data/checkpoints',
                                          filename='Aligned-VIT-{epoch:03d}-{val_arr:.2f}',
                                          save_top_k=3,
                                          mode='max')

    trainer = pl.Trainer(gpus=config.DEVICES, 
                         logger=wandb_logger, 
                         log_every_n_steps=config.LOG_STEP,
                         callbacks=[checkpoint_callback])
    trainer.fit(model, trainloader, val_dataloaders=valloader)

    print("Finished Training")


if __name__ == "__main__":
    main()
