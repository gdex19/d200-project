import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from final_project_d200.evaluation import compute_nll_from_mdn_output, compute_pretrain_loss_from_output
from typing import Callable

def train_mdn(model: nn.Module, data_loader: DataLoader, optimizer: torch.optim.Optimizer, lambda_mean: float = 0.0) -> float:
    """
    Train one epoch for MDN predicting return distribution.

    Parameters
    ----------
    model : nn.Module
        MDN model.
    data_loader : DataLoader
        Data loader with inputs and targets.
    optimizer : torch.optim.Optimizer
        Optimizer for training.
    lambda_mean : float, optional
        Regularization parameter for mean predictions, defaults to 0.0.

    Returns
    -------
    float
        Mean nll across batches.
    """
    device = next(model.parameters()).device
    model = model.to(device)

    model.train()
    nlls = []


    for _, (X, y) in enumerate(data_loader):
        X, y = X.to(device), y.to(device)

        optimizer.zero_grad()
        mixture_weights, means, scales = model(X)

        loss = compute_nll_from_mdn_output(y, mixture_weights, means, scales, lambda_mean=lambda_mean)
        nlls.append(loss.item())

        loss.backward()
        optimizer.step()
    
    return sum(nlls) / len(nlls)


def pretrain_mdn(model: nn.Module, data_loader: DataLoader, optimizer: torch.optim.Optimizer, loss_fn: Callable) -> float:
    """
    Train one epoch for MDN backbone used for transfer learning.

    Parameters
    ----------
    model : nn.Module
        Backbone model.
    data_loader : DataLoader
        Data loader with inputs and targets.
    optimizer : torch.optim.Optimizer
        Optimizer for training.
    loss_fn : Callable
        Loss function for regression targets.

    Returns
    -------
    float
        Mean loss across batches.
    """
    device = next(model.parameters()).device
    model = model.to(device)

    model.train()
    losses = []

    for _, (X, y) in enumerate(data_loader):
        X, y = X.to(device), y.to(device)

        optimizer.zero_grad()

        vol_5_pred, vol_15_pred, vol_30_pred, abs_30_pred, direction_pred = model(X)

        loss = compute_pretrain_loss_from_output(
            y[:, 0:1], vol_5_pred,
            y[:, 1:2], vol_15_pred,
            y[:, 2:3], vol_30_pred,
            y[:, 3:4], abs_30_pred,
            y[:, 4].long(), direction_pred,
            loss_fn,
        )

        losses.append(loss.item())

        loss.backward()
        optimizer.step()
    
    return sum(losses) / len(losses)