import matplotlib.pyplot as plt
import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import ensemble_data
import config
import data
import wandb
from nets import ensembleNet

plt.style.use('ggplot')

def main():
    

    model = ensembleNet()
    
    trainingset_path = 'ensemble/ensemble_trainingset_sig.csv'
    evaluationset_path = 'ensemble/ensemble_validset_sig.csv'
    train_df = 'ensemble/ensemble_training_label.csv'
    val_df = 'ensemble/ensemble_valid_label.csv'
    trainset = ensemble_data.ensembleDataset(train_df, trainingset_path)
    valset = ensemble_data.ensembleDataset(val_df, evaluationset_path)

    trainloader = DataLoader(trainset, batch_size=32, shuffle=True, num_workers=20)
    valloader = DataLoader(valset, batch_size=32, shuffle=False, num_workers=20)

    wandb.init(project='ensemble')
    wandb_logger = WandbLogger(project='ensemble')
    checkpoint_callback = ModelCheckpoint(monitor='val_loss', 
                                          dirpath='data/checkpoints',
                                          filename='ensemble-{epoch:03d}-{val_loss:.5f}',
                                          save_top_k=4,
                                          mode='min')

    trainer = pl.Trainer(gpus=config.DEVICES, 
                        #  num_nodes=2,
                         logger=wandb_logger, 
                         log_every_n_steps=config.LOG_STEP,
                         callbacks=[checkpoint_callback])
    trainer.fit(model, trainloader, val_dataloaders=valloader)

    print("Finished Training")


if __name__ == "__main__":
    main()
