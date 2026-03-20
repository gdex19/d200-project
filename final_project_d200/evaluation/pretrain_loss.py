import torch
import torch.nn.functional as F
from typing import Callable
import torch.nn as nn
from torch.utils.data import DataLoader

def compute_pretrain_loss_from_output(
    vol_5_true: torch.Tensor,
    vol_5_pred: torch.Tensor,
    vol_15_true: torch.Tensor,
    vol_15_pred: torch.Tensor,
    vol_30_true: torch.Tensor,
    vol_30_pred: torch.Tensor,
    abs_30_true: torch.Tensor,
    abs_30_pred: torch.Tensor,
    direction_true: torch.Tensor,
    direction_pred: torch.Tensor,
    loss_fn: Callable,
    lambda_vol_5: float = 1.0,
    lambda_vol_15: float = 1.0,
    lambda_vol_30: float = 1.0,
    lambda_abs_30: float = 1e-4,
    lambda_direction: float = 5e-3,
) -> torch.Tensor:
    """
    Compute weighted pretrain loss for 5 predictions

    Parameters
    ----------
    vol_5_true : torch.Tensor
        True future 5 min mean of 1m sq returns.
    vol_5_pred : torch.Tensor
        Predicted future 5 min mean of 1m sq returns.
    vol_15_true : torch.Tensor
        True future 15 min mean of 1m sq returns.
    vol_15_pred : torch.Tensor
        Predicted future 15 min mean of 1m sq returns.
    vol_30_true : torch.Tensor
        True future 30 min mean of 1m sq returns.
    vol_30_pred : torch.Tensor
        Predicted future 30 min mean of 1m sq returns.
    abs_30_true : torch.Tensor
        True future 30 min mean of 1m abs returns.
    abs_30_pred : torch.Tensor
        Predicted future 30 min mean of 1m abs returns.
    direction_true : torch.Tensor
        True direction labels.
    direction_pred : torch.Tensor
        Predicted logits for direction.
    loss_fn : Callable
        Loss function for regression targets.
    lambda_vol_5 : float, optional
        Weight for vol_5 loss.
    lambda_vol_15 : float, optional
        Weight for vol_15 loss.
    lambda_vol_30 : float, optional
        Weight for vol_30 loss.
    lambda_abs_30 : float, optional
        Weight for abs_30 loss.
    lambda_direction : float, optional
        Weight for direction loss.

    Returns
    -------
    torch.Tensor
        Weighted total loss.
    """
    loss_vol_5 = loss_fn(vol_5_pred, vol_5_true)
    loss_vol_15 = loss_fn(vol_15_pred, vol_15_true)
    loss_vol_30 = loss_fn(vol_30_pred, vol_30_true)
    loss_abs_30 = loss_fn(abs_30_pred, abs_30_true)

    loss_direction = F.cross_entropy(direction_pred, direction_true)

    return (
        lambda_vol_5 * loss_vol_5
        + lambda_vol_15 * loss_vol_15
        + lambda_vol_30 * loss_vol_30
        + lambda_abs_30 * loss_abs_30
        + lambda_direction * loss_direction
    )


def compute_pretrain_loss(model: nn.Module, data_loader: DataLoader, loss_fn: Callable) -> float:
    """
    Compute pretrain loss from DataLoader. Requires y elements to be in order
    [vol_5, vol_15, vol_30, abs_30, direction].

    Parameters
    ----------
    model : nn.Module
        MDN model for predicting given y elements.
    data_loader : DataLoader
        Data loader with inputs and targets.
    loss_fn : Callable
        Loss function for regression targets.

    Returns
    -------
    float
        Mean loss across all batches.
    """
    model.eval() 
    device = next(model.parameters()).device
    size = 0
    loss_sum = 0
    with torch.no_grad():
        for (X, y) in data_loader:
            X, y = X.to(device), y.to(device)

            vol_5_pred, vol_15_pred, vol_30_pred, abs_30_pred, direction_pred = model(X)

            loss = compute_pretrain_loss_from_output(
                y[:, 0:1], vol_5_pred,
                y[:, 1:2], vol_15_pred,
                y[:, 2:3], vol_30_pred,
                y[:, 3:4], abs_30_pred,
                y[:, 4].long(), direction_pred,
                loss_fn,
            )

            loss_sum += loss.item() * X.size(0)
            size += X.size(0)

    return loss_sum / size


def compute_pretrain_loss_components(model: nn.Module, data_loader: DataLoader, loss_fn: Callable) -> tuple[float, float, float, float, float]:
    """
    Compute values of pretrain loss components and return them separately. Requires y elements to be in order
    [vol_5, vol_15, vol_30, abs_30, direction].

    Parameters
    ----------
    model : nn.Module
        MDN model for predicting given y elements.
    data_loader : DataLoader
        Data loader with inputs and targets.
    loss_fn : Callable
        Loss function for regression targets.

    Returns
    -------
    tuple[float, float, float, float, float]
        Mean losses across all batches.
    """
    model.eval()
    device = next(model.parameters()).device
    size = 0

    loss_vol_5_sum = 0
    loss_vol_15_sum = 0
    loss_vol_30_sum = 0
    loss_abs_30_sum = 0
    loss_direction_sum = 0

    with torch.no_grad():
        for (X, y) in data_loader:
            X, y = X.to(device), y.to(device)

            vol_5_pred, vol_15_pred, vol_30_pred, abs_30_pred, direction_pred = model(X)

            loss_vol_5 = loss_fn(vol_5_pred, y[:, 0:1])
            loss_vol_15 = loss_fn(vol_15_pred, y[:, 1:2])
            loss_vol_30 = loss_fn(vol_30_pred, y[:, 2:3])
            loss_abs_30 = loss_fn(abs_30_pred, y[:, 3:4])

            loss_direction = F.cross_entropy(direction_pred, y[:, 4].long())

            loss_vol_5_sum += loss_vol_5.item() * X.size(0)
            loss_vol_15_sum += loss_vol_15.item() * X.size(0)
            loss_vol_30_sum += loss_vol_30.item() * X.size(0)
            loss_abs_30_sum += loss_abs_30.item() * X.size(0)
            loss_direction_sum += loss_direction.item() * X.size(0)

            size += X.size(0)

    return (
        loss_vol_5_sum / size,
        loss_vol_15_sum / size,
        loss_vol_30_sum / size,
        loss_abs_30_sum / size,
        loss_direction_sum / size,
    )