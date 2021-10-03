import config
import cv2
import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch
import wandb
from metrics import *
from sklearn.metrics import confusion_matrix

cmap = plt.get_cmap('jet')

class BaseNet(pl.LightningModule):
    
    def forward(self, x):
        # in lightning, forward defines the prediction/inference actions
        return self.model(x)

    def training_step(self, batch, batch_idx):
        # training_step defined the train loop.
        # It is independent of forward
        x, y = batch
        x = x.float()
        y = y.float() / 81
        y_pred = self(x)

        loss = self.loss_func(y_pred.squeeze(), y)

        # Logging to TensorBoard by default
        if batch_idx % 100 == 0:
            wandb.log({"train_loss": loss})

            y_true = (y * 81).round().long().squeeze().detach().cpu().numpy()
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
            self.log("Train - AAR", aar_score)
            self.log("Train - MAE", mae_score)
            self.log("Train - Sigma", sigma)
                    
            wandb.log(
                {
                    "Train - Sigmas": wandb.Image(plt),
                    "Train - Confusion Matrix": wandb.Image(conf_mat),
                }
            )

            plt.clf()

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        x = x.float()
        y = y.float() / 81

        y_pred = self(x)

        loss = self.loss_func(y_pred.squeeze(), y)

        return {"loss": loss, "pred": y_pred, "y": y}

    def validation_epoch_end(self, validation_step_outputs):
        y_true = torch.tensor([]).to(validation_step_outputs[0]["loss"].device)
        y_pred = torch.tensor([]).to(validation_step_outputs[0]["loss"].device)
        for out in validation_step_outputs:
            y_true = torch.cat([y_true, out["y"]])
            y_pred = torch.cat([y_pred, out["pred"]])

        loss = self.loss_func(y_pred.squeeze(), y_true)
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
        
        self.log("val_aar", aar_score)
        self.log("Validation - AAR", aar_score)
        self.log("Validation - MAE", mae_score)
        self.log("Validation - Sigma", sigma)

        wandb.log(
            {
                "Validation - Sigmas": wandb.Image(plt),
                "Validation - Confusion Matrix": wandb.Image(conf_mat),
            }
        )
        plt.clf()
        
        plt.bar(
            ["< 10", "10-19", "20-29", "30-39", "40-24", "50-59", "60-69", "> 70"],
            maes,
        )
        wandb.log(
            {
                "Validation - MAEs": wandb.Image(plt),
            }
        )
        plt.clf()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=config.LR)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, config.WARM_RESTARTS)
        return [optimizer], [scheduler]
