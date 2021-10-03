import numpy as np


def mae(y_true : np.ndarray, y_pred: np.ndarray) -> float:
    if y_true.shape[0] == 0:
        return 0
    return np.abs(y_true - y_pred).mean()

def aar(y_true : np.ndarray, y_pred: np.ndarray) -> float:
    true_age_groups = np.clip(y_pred // 10, 0, 7)
    mae_score = mae(y_true, y_pred)
    
    # MAE per age group
    sigmas = []
    maes = []
    for i in range(8):
        idx = true_age_groups == i
        mae_age_group = mae(y_true[idx], y_pred[idx])
        maes.append(mae_age_group)
        sigmas.append((mae_age_group - mae_score) ** 2)

    sigma = np.sqrt(np.array(sigmas).mean())
    
    aar_score = max(0, 7 - mae_score) + max(0, 3 - sigma)
    
    return aar_score, mae_score, sigma, sigmas, maes


def top_k_accuracy(y_true : np.ndarray, probs: np.ndarray, k: int) -> float:
    """Returns ranked accuracy given true classes and predicted probabilities.

    Args:
        y_true (np.ndarray): Ground truth labels of shape (N,).
        probs (np.ndarray): Class probabilities (soft one-hot encoding) of shape
        (N, C) where C is the number of classes.
        k (int): Rank

    Returns:
        float: Top k accuracy
    """
    
    # Top k sorted preds
    sorted_probs = probs.argsort()[:,-k:]

    # Does the truth intersect with any of the top k predictions?
    matches = np.max(sorted_probs == y_true.reshape(-1, 1), axis=1)
    return matches.mean()


    
    
