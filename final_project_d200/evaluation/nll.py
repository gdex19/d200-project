import numpy as np
from numpy.typing import NDArray
import pandas as pd
from scipy.optimize import minimize_scalar
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

def calculate_normal_nll(y: pd.Series, sigma: NDArray, min_sigma: float = 1e-12):
    """
    Calculate negative log-likelihood from returns and 
    scale parameters.

    Parameters
    ----------
    y : pd.Series
        Observed returns.
    sigma : NDArray 
        Predicted std of normal distribution (returns).
    min_sigma : float, optional
        Minimum value for sigma.

    Returns
    -------
    float
        Mean nll.
    """
    sigma = np.clip(sigma, min_sigma, None)
    return np.mean(
        0.5 * np.log(2 * np.pi)
        + np.log(sigma)
        + 0.5 * (y / sigma) ** 2
    )


def optimize_gbt_constant_for_nll(y_train: pd.Series, sigma_train: NDArray) -> float:
    """
    Get nll minimizing constant for gbt prediction nll.

    Parameters
    ----------
    y_train : pd.Series
        Observed returns.
    sigma_train : NDArray 
        Predicted std of normal distribution (returns).

    Returns
    -------
    float
        Optimal constant multiplier.
    """
    objective = lambda c : calculate_normal_nll(y_train, sigma_train * c)
    c_hat = minimize_scalar(objective, bounds=(0,5), method="bounded").x
    return c_hat


def compute_nll_from_mdn_output(y: torch.Tensor, mixture_probs: torch.Tensor, means: torch.Tensor, scales: torch.Tensor, lambda_mean: float = 0.0) -> torch.Tensor:
    """
    Compute nll from model output.

    Parameters
    ----------
    y : torch.Tensor
        Observed returns.
    mixture_probs : torch.Tensor
        Mixture probabilities.
    means : torch.Tensor
        Component means.
    scales : torch.Tensor
        Component standard deviations.
    lambda_mean : float, optional
        Regularization parameter on means, not used in final model.

    Returns
    -------
    float
        Mean negative log-likelihood.
    """
    y = y.unsqueeze(1)

    log_probs = (
        -0.5 * np.log(2 * np.pi)
        - torch.log(scales)
        - 0.5 * ((y - means) / scales) ** 2
    )

    log_mix = torch.log(mixture_probs)
    log_density = torch.logsumexp(log_mix + log_probs, dim=1)
    

    return -log_density.mean() + lambda_mean * torch.mean(torch.sum(mixture_probs * means ** 2, dim=1))

def compute_mdn_nll(model: nn.Module, data_loader: DataLoader, lambda_mean: float = 0.0) -> float:
    """
    Compute average nll from data loader and mdn model.

    Parameters
    ----------
    model : nn.Module
        MDN model.
    data_loader : DataLoader
        Data loader with inputs and targets.
    lambda_mean : float, optional
        Regularization parameter on means, not used in final model.

    Returns
    -------
    float
        Mean negative log-likelihood across batches.
    """
    model.eval() 
    device = next(model.parameters()).device
    size = 0
    nll_sum: torch.Tensor = torch.tensor(0.0, device=device)

    with torch.no_grad():
        for (X, y) in data_loader:
            X, y = X.to(device), y.to(device)
            mixture_probs, means, scales = model(X)

            nll = compute_nll_from_mdn_output(y, mixture_probs, means, scales, lambda_mean=lambda_mean)
            nll_sum += nll * X.size(0)
            size += X.size(0)

    return (nll_sum / size).item()
