import config
import cv2
import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch
import wandb
from metrics import *
from sklearn.metrics import confusion_matrix
import torch.nn as nn

cmap = plt.get_cmap('jet')

class BaseNet(pl.LightningModule):
    
    def forward(self, x):
        # in lightning, forward defines the prediction/inference actions
        return self.model(x)

    def training_step(self, batch, batch_idx):
        # training_step defined the train loop.
        # It is independent of forward
        x, y, w = batch
        x = x.float()
        y = y.float()
        y_pred = self(x)

        loss = self.loss_func(y_pred.squeeze(), y, w)
        self.log('train_loss', loss)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y, w= batch
        x = x.float()
        y = y.float()

        y_pred = self(x)
        loss = self.loss_func(y_pred.squeeze(), y, w)

        return {"loss": loss, "pred": y_pred, "y": y}

    def validation_epoch_end(self, validation_step_outputs):
        y_true = torch.tensor([]).to(validation_step_outputs[0]["loss"].device)
        y_pred = torch.tensor([]).to(validation_step_outputs[0]["loss"].device)
        for out in validation_step_outputs:
            y_true = torch.cat([y_true, out["y"]])
            y_pred = torch.cat([y_pred, out["pred"]])

        criterion = nn.BCEWithLogitsLoss()
        loss = criterion(y_pred.squeeze(), y_true)
        #loss = self.loss_func(y_pred.squeeze(), y_true)
        self.log('val_loss', loss)
        #wandb.log({"val_loss": loss})

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=config.LR)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, config.WARM_RESTARTS)
        return [optimizer], [scheduler]
