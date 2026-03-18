import numpy as np
from numpy.typing import NDArray
import pandas as pd
from scipy.optimize import minimize_scalar

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
