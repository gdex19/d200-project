import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from final_project_d200.evaluation import (
    compute_nll_from_mdn_output, compute_pretrain_loss_from_output, compute_mdn_nll, 
    compute_mean_crps_mdn, compute_mean_crps_mdn_outputs
)
from typing import Callable
import numpy as np
from typing import Any

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

def train_mdn_crps(model: nn.Module, data_loader: DataLoader, optimizer: torch.optim.Optimizer) -> float:
    """
    Train one epoch for MDN predicting return distribution, using crps as loss.

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
    crpss = []


    for _, (X, y) in enumerate(data_loader):
        X, y = X.to(device), y.to(device)

        optimizer.zero_grad()
        mixture_weights, means, scales = model(X)

        loss = compute_mean_crps_mdn_outputs(y, mixture_weights, means, scales)
        crpss.append(loss.item())

        loss.backward()
        optimizer.step()
    
    return sum(crpss) / len(crpss)

def train_mdn_crps_and_nll(model: nn.Module, data_loader: DataLoader, optimizer: torch.optim.Optimizer, crps_weight: float = 0.0) -> float:
    """
    Train one epoch for MDN predicting return distribution, using crps and nll as loss.

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
    losses = []


    for _, (X, y) in enumerate(data_loader):
        X, y = X.to(device), y.to(device)

        optimizer.zero_grad()
        mixture_probs, means, scales = model(X)

        crps_loss = compute_mean_crps_mdn_outputs(y, mixture_probs, means, scales)
        nll_loss = compute_nll_from_mdn_output(y, mixture_probs, means, scales)

        crps_loss_scaled = crps_loss * 1000
        loss = crps_loss_scaled * crps_weight + nll_loss * (1 - crps_weight)
        losses.append(loss.item())

        loss.backward()
        optimizer.step()
    
    return sum(losses) / len(losses)


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


def run_mdn_config(
        model_builder: Callable[[dict[str, Any], int], nn.Module], params: dict[str, Any], 
        X_train: torch.Tensor, y_train: torch.Tensor, 
        X_val: torch.Tensor, y_val: torch.Tensor, 
        epochs: int = 300, early_stopping_patience: int = 50
) -> tuple[float, float, float, float, int]:
    """
    Run a training loop for MDN, used for hyperparameter grid search.

    Parameters
    ----------
    model_builder : Callable[[dict[str, Any], int], nn.Module]
        Function to build the MDN model.
    params : dict[str, Any]
        Hyperparameter dict.
    X_train, y_train : torch.Tensor
        Training data.
    X_val, y_val : torch.Tensor
        Validation data.
    epochs : int
        Max number of epochs.
    early_stopping_patience : int
        Max number of epochs without improvement before stopping.

    Returns
    -------
    tuple[float, float, float, float, int]e
        Best validation NLL, average train NLL near optimum, average validation NLL 
        near optimum, average validation CRPS near optimum, and number of epochs until stopping.
    """

    train_data = TensorDataset(X_train, y_train)
    val_data = TensorDataset(X_val, y_val)

    train_loader = DataLoader(train_data, batch_size=params["batch_size"], shuffle=True)
    val_loader = DataLoader(val_data, batch_size=params["batch_size"], shuffle=False)
    
    model: nn.Module = model_builder(params, X_train.shape[1])

    optimizer = torch.optim.Adam(model.parameters(), lr=params["lr"], weight_decay=params["weight_decay"])

    results = []
    best_val_nll = np.inf
    last_improvement = 0
    for t in range(epochs):
        epoch_loss = train_mdn(model, train_loader, optimizer, lambda_mean=params["lambda_mean"])
        epoch_train_nll = compute_mdn_nll(model, train_loader)
        epoch_val_nll = compute_mdn_nll(model, val_loader)
        epoch_val_crps = compute_mean_crps_mdn(model, val_loader)

        if epoch_val_nll < best_val_nll:
            best_val_nll = epoch_val_nll
            last_improvement = t

        results.append([epoch_loss, epoch_train_nll, epoch_val_nll, epoch_val_crps])
        
        if t - last_improvement >= early_stopping_patience:
            break

    results = np.array(results)

    best_epoch = last_improvement
    left = max(0, best_epoch - 10)
    right = min(len(results), best_epoch + 3)

    avg_train_nll_near_optimum = float(np.mean(results[left:right, 1]))
    avg_val_nll_near_optimum = float(np.mean(results[left:right, 2]))
    avg_val_crps_near_optimum = float(np.mean(results[left:right, 3]))

    return best_val_nll, avg_train_nll_near_optimum, avg_val_nll_near_optimum, avg_val_crps_near_optimum, len(results)
    
