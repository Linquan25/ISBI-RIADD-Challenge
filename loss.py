import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score
import torch.nn as nn

def soft_aar_loss(y_pred: torch.FloatTensor, y_true: torch.Tensor):

    mae = F.smooth_l1_loss(y_pred * 81, y_pred * 81)
    true_age_groups = torch.clip(y_true // 10, 0, 7)
    std = 0
    for i in range(8):
        idx = true_age_groups == i
        if y_true[idx].shape[0] != 0:
            mae_age_group = F.smooth_l1_loss(y_true[idx] * 81, y_pred[idx] * 81)
            std += (mae_age_group - mae) ** 2

    return 7 * mae + 3 * torch.sqrt(std / 8)

def weighted_mse_loss(inputs, targets, weights=None):
    loss = (inputs - targets) ** 2
    if weights is not None:
        loss *= weights
    loss = torch.mean(loss)
    return loss

def weighted_BCEWithLogitsLoss(inputs, targets, weights=None):
    criterion = nn.BCEWithLogitsLoss()
    loss = weights * criterion(inputs, targets)
    loss = torch.mean(loss)
    return loss
    