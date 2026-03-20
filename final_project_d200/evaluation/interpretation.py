import torch
import torch.nn as nn
from torch.utils.data import DataLoader

def get_predicted_parameters(model: nn.Module, data_loader: DataLoader) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Return predicted mixture probabilities, means, and scales across dataset.

    Parameters
    ----------
    model : nn.Module
        Trained MDN model.
    data_loader : DataLoader
        Data loader with inputs and targets.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        Predicted probabilities, means, and scales for components for every observation.
    """
    model.eval() 
    device = next(model.parameters()).device

    probs = []
    means = []
    scales = []
    with torch.no_grad():
        for (X, y) in data_loader:
            X, y = X.to(device), y.to(device)
            mixture_probs, mean, scale = model(X)
            probs.append(mixture_probs)
            means.append(mean)
            scales.append(scale)

        probs = torch.cat(probs, dim=0)
        means = torch.cat(means, dim=0)
        scales = torch.cat(scales, dim=0)

    return probs, means, scales