import numpy as np
from typing import Callable
from numpy.typing import NDArray
from scipy import stats

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

