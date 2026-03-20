import numpy as np
from typing import Callable
from numpy.typing import NDArray
from scipy import stats
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

def compute_single_crps_arbitrary(F: NDArray , y: NDArray, x_grid: NDArray, dx: float):
    """
    Compute CRPS for distribution and realization.

    Parameters
    ----------
    F : NDArray
        CDF values evaluated on grid.
    y : NDArray
        Realized value.
    x_grid : NDArray
        Grid for cdf.
    dx : float
        Grid spacing.

    Returns
    -------
    float
        CRPS value.
    """

    # grid
    left = x_grid <= y
    right = ~left

    crps = (
        np.sum(F[left]**2) * dx +
        np.sum((1 - F[right])**2) * dx
    )

    return crps

def compute_mean_crps_arbitrary(cdf: Callable, y_vals: NDArray, x_min: float, x_max: float, n_grid: int = 2000):
    """
    Compute mean crps across a set of realizations.

    Parameters
    ----------
    cdf : Callable
        Normal cdf function.
    y_vals : NDArray
        Array of realized values.
    x_min : float
        Lower bound of evaluated region.
    x_max : float
        Upper bound of evaluated region.
    n_grid : int, optional
        Number of grid points to split region into.

    Returns
    -------
    float
        Mean CRPS across all datapoints.
    """
    x = np.linspace(x_min, x_max, n_grid)
    dx = x[1] - x[0]
    F = cdf(x)
    scores = []
    for y in y_vals:
        scores.append(compute_single_crps_arbitrary(F, y, x, dx))
    return np.mean(scores)


def compute_mean_crps_gaussian(y_vals: NDArray, sigmas: NDArray) -> float:
    """
    CRPS for N(0, sigma^2) predictions, closed form solution.

    Parameters
    ----------
    y_vals : NDArray
        Realized returns for CRPS.
    sigmas : NDArray
        Stds for normal distributions.

    Returns
    -------
    float
        Mean CRPS across all datapoints.
    """
    z = y_vals / sigmas
    crps = sigmas * (
        z * (2 * stats.norm.cdf(z) - 1)
        + 2 * stats.norm.pdf(z)
        - 1 / np.sqrt(np.pi)
    )
    return np.mean(crps)


def crps_mdn_helper(a: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """
    Closed form function used for gaussian mixture CRPS, computed for whole tensor.

    Parameters
    ----------
    a : torch.Tensor
        Difference term used for calculation.
    scale : torch.Tensor
        Standard deviation term for each mixture + observation.

    Returns
    -------
    torch.Tensor
        Value of each term for each observation.
    """
    value = 2 * scale * stats.norm.pdf(a / scale) +  a * (2 * stats.norm.cdf(a / scale) - 1)
    return value

def compute_mean_crps_mdn(model: nn.Module, data_loader: DataLoader) -> float:
    """
    Compute mean crps using MDN and data loader.

    Parameters
    ----------
    model : nn.Module
        MDN model.
    data_loader : DataLoader
        Data loader with inputs and targets.

    Returns
    -------
    float
        Mean CRPS on dataloader.
    """
    model.eval() 
    device = next(model.parameters()).device

    all_crps = []
    with torch.no_grad():
        for (X, y) in data_loader:
            X, y = X.to(device), y.to(device)
            mixture_probs, means, scales = model(X)

            a_one = y.unsqueeze(1) - means
            scale_one = scales
            weights_one = mixture_probs
            first_term_raw = crps_mdn_helper(a_one, scale_one)
            first_term = torch.sum(first_term_raw * weights_one, dim=1)
            
            a_two = means.unsqueeze(2) - means.unsqueeze(1)
            scale_two = torch.sqrt(scales.unsqueeze(2) ** 2 + scales.unsqueeze(1) ** 2)
            weights_two = mixture_probs.unsqueeze(2) * mixture_probs.unsqueeze(1)
            second_term_raw = crps_mdn_helper(a_two, scale_two)
            second_term = torch.sum(second_term_raw * weights_two, dim=(1, 2))

            crpss = first_term - 0.5 * second_term
            all_crps.append(crpss)

    crps_tensor = torch.concat(all_crps, dim=0)

    return torch.mean(crps_tensor).item()
