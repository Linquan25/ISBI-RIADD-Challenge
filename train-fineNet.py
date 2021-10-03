import matplotlib.pyplot as plt
import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import ISBI_rareset
import config
import data
import wandb
from nets import FineNet

plt.style.use('ggplot')

def main():
    

    model = FineNet()

    #data_df = pd.read_csv(config.CSV_PATH)
    #train_df, val_df = train_test_split(data_df, test_size=0.1) 
    training_img_path = '../Training_Set/Training/'
    evaluation_img_path = '../Evaluation_Set/Validation'
    train_df = 'fineNet/train_rare.csv'
    val_df = 'fineNet/valid_rare.csv'
    trainset = ISBI_rareset.ISBIRareset(train_df, training_img_path, testing=False, reweight=False)
    valset = ISBI_rareset.ISBIRareset(val_df, evaluation_img_path, testing=True, reweight=False)

    trainloader = DataLoader(trainset, batch_size=32, shuffle=True, num_workers=20)
    valloader = DataLoader(valset, batch_size=32, shuffle=False, num_workers=20)

    wandb.init(project='ISBI-BCE-FineNet')
    wandb_logger = WandbLogger(project='ISBI-BCE-FineNet')
    checkpoint_callback = ModelCheckpoint(monitor='val_loss', 
                                          dirpath='data/checkpoints',
                                          filename='ISBI-BCE-FineNet_fromScratch-{epoch:03d}-{val_loss:.4f}',
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
