import config
import cv2
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import wandb
from loss import soft_aar_loss
from metrics import *
from sklearn.metrics import confusion_matrix

from .base import BaseNet


class TSNet(BaseNet):
    def __init__(self):
        super().__init__()
        self.features = torch.hub.load('pytorch/vision:v0.9.0', 
                                       'resnext50_32x4d', 
                                       pretrained=True)
        self.features.fc = nn.Identity()
        self.classification = nn.Linear(2048, 8)
        self.regression = nn.Linear(2048, 1)
        
        self.loss_func_c = nn.CrossEntropyLoss()
        self.loss_func_r = nn.MSELoss()
        
    def forward(self, x):
        feats = self.features(x)
        # Classification
        age_c = self.classification(feats)
        # Regression
        age_r = self.regression(feats)
        
        return age_c, age_r
    
    def training_step(self, batch, batch_idx):
        # training_step defined the train loop.
        # It is independent of forward
        x, y = batch
        x = x.float()
        y_r = y.float() / 81
        y_c = torch.clip(y // 10, 0, 7)
        y_pred_c, y_pred_r = self(x)

        loss_r = self.loss_func_r(y_pred_r.squeeze(), y_r)
        loss_c = self.loss_func_c(y_pred_c.squeeze(), y_c)
        loss = loss_c + loss_r
        
        # Logging to TensorBoard by default
        if batch_idx % 100 == 0:
            wandb.log({
                "train_loss": loss,
                "train_loss_c": loss_c,
                "train_loss_r": loss_r
            })

            y_true_r = y.long().squeeze().detach().cpu().numpy()
            y_pred_r = (y_pred_r * 81).round().long().squeeze().detach().cpu().numpy()

            conf_mat = confusion_matrix(y_true_r, y_pred_r, labels=np.arange(1, 81), normalize='true')
            conf_mat = (conf_mat - conf_mat.min()) / (conf_mat.max() - conf_mat.min())
            conf_mat = cv2.resize(
                conf_mat, (8 * 81, 8 * 81), interpolation=cv2.INTER_NEAREST
            )

            conf_mat = cmap(conf_mat)[:, :, :3] * 255

            aar_score, mae_score, sigma, sigmas, maes = aar(y_true_r, y_pred_r)

            plt.bar(
                ["< 10", "10-19", "20-29", "30-39", "40-24", "50-59", "60-69", "> 70"],
                sigmas,
            )

            wandb.log(
                {
                    "Train - AAR": aar_score,
                    "Train - MAE": mae_score,
                    "Train - Sigma": sigma,
                    "Train - Sigmas": wandb.Image(plt),
                    "Train - Confusion Matrix": wandb.Image(conf_mat),
                }
            )

            plt.clf()

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        x = x.float()
        
        y_r = y.float() / 81
        y_c = torch.clip(y // 10, 0, 7)
        
        y_pred_c, y_pred_r = self(x)

        loss_r = self.loss_func_r(y_pred_r.squeeze(), y_r)
        loss_c = self.loss_func_c(y_pred_c.squeeze(), y_c)
        loss = loss_c  + loss_r
        
        return {"loss": loss, "loss_c": loss_c, "loss_r": loss_r, 
                "y_pred_r": y_pred_r, "y_pred_c": y_pred_c, "y": y_r}

    def validation_epoch_end(self, validation_step_outputs):
        y_true = torch.tensor([]).to(validation_step_outputs[0]["loss"].device)
        y_pred = torch.tensor([]).to(validation_step_outputs[0]["loss"].device)
        for out in validation_step_outputs:
            y_true = torch.cat([y_true, out["y"]])
            y_pred = torch.cat([y_pred, out["y_pred_r"]])

        loss = torch.mean(torch.tensor([l['loss'] for l in validation_step_outputs]))
        wandb.log({"val_loss": loss})

        y_true = (y_true * 81).round().long().squeeze().detach().cpu().numpy()
        y_pred = (y_pred * 81).round().long().squeeze().detach().cpu().numpy()

        conf_mat = confusion_matrix(y_true, y_pred, labels=np.arange(1, 81), normalize='true')
        conf_mat = (conf_mat - conf_mat.min()) / (conf_mat.max() - conf_mat.min())
        conf_mat = cv2.resize(
            conf_mat, (8 * 81, 8 * 81), interpolation=cv2.INTER_NEAREST
        )
        conf_mat = cmap(conf_mat)[:, :, :3] * 255

        aar_score, mae_score, sigma, sigmas, maes = aar(y_true, y_pred)

        plt.bar(
            ["< 10", "10-19", "20-29", "30-39", "40-24", "50-59", "60-69", "> 70"],
            sigmas,
        )
        
        self.log('val_aar', aar_score)
        self.log('Validation - MAE', mae_score)
        self.log('Validation - Sigma', sigma)
        
        wandb.log(
            {
                "Validation - Sigmas": wandb.Image(plt),
                "Validation - Confusion Matrix": wandb.Image(conf_mat),
            }
        )

        plt.clf()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=config.LR)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, config.WARM_RESTARTS)
        return [optimizer], [scheduler]


