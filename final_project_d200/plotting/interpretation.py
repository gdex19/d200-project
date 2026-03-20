import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import seaborn as sns
from pathlib import Path
import pandas as pd
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from pathlib import Path
from final_project_d200.evaluation import get_predicted_parameters


def plot_predicted_parameters(model: nn.Module, model_name: str, data_loader: DataLoader, dataset_name: str, save_figs: bool = False) -> tuple[Axes, Axes, Axes]:
    """
    Plot distributions of predicted mixture component parameters for all observations in data loader.

    Parameters
    ----------
    model : nn.Module
        Trained MDN model.
    model_name : str
        Name of the model, used for labeling plots and saved files.
    data_loader : DataLoader
        Data loader with inputs for predictions.
    dataset_name : str
        Name of dataset, used for labeling plots and saved files.
    save_figs : bool, optional
        Whether to save plots to disk.

    Returns
    -------
    tuple[Axes, Axes, Axes]
        Axes objects for weights, means, and scales plots.
    """

    weights, means, scales = (p.numpy() for p in get_predicted_parameters(model, data_loader))

    order = weights.mean(axis=0).argsort()
    weights = weights[:, order]
    means = means[:, order]
    scales = scales[:, order]

    if save_figs:
        plot_path = Path("../plots")
        plot_path.mkdir(parents=True, exist_ok=True)
    else:
        plot_path = Path()

    plt.figure(figsize=(5,4))
    ax1 = sns.histplot(weights, bins=50, alpha=0.5)
    ax1.set_title(f"Mixture weights, {model_name} on {dataset_name} data")
    plt.tight_layout()
    if save_figs:
        file_path = plot_path / f"mixture_weights_{model_name}_{dataset_name}.pdf"
        plt.savefig(file_path, bbox_inches="tight")
    plt.show()
    plt.close()

    plt.figure(figsize=(5,4))
    ax2 = sns.histplot(means, bins=100, alpha=0.5)
    ax2.set_title(f"Mixture means, {model_name} on {dataset_name} data")
    plt.tight_layout()
    if save_figs:
        file_path = plot_path / f"mixture_means_{model_name}_{dataset_name}.pdf"
        plt.savefig(file_path, bbox_inches="tight")
    plt.show()
    plt.close()

    plt.figure(figsize=(5,4))
    ax3 = sns.histplot(scales, bins=100, alpha=0.5)
    ax3.set_title(f"Mixture scales, {model_name} on {dataset_name} data")
    plt.tight_layout()
    if save_figs:
        file_path = plot_path / f"mixture_scales_{model_name}_{dataset_name}.pdf"
        plt.savefig(file_path, bbox_inches="tight")
    plt.show()
    plt.close()

    return (ax1, ax2, ax3)


def plot_mdns_vs_gbt_vol_medoid(
    X: torch.Tensor, models: dict[str, nn.Module], 
    df: pd.DataFrame, left: float = -0.02, right: float = 0.02, 
    n_grid: int = 1000, save_fig: bool = False,
    file_name: str = "mdn_vs_gbt_density.pdf"
) -> Axes:
    """
    Plot predicted distribution for mdns in models and two-stage GBT model on
    medoid data point in X/df.

    Parameters
    ----------
    X : torch.Tensor
        Input data to MDN.
    models : dict[str, nn.Module]
        Trained MDN model with names as keys.
    df : pd.DataFrame
        df used to create X, with same indices
    left : float, optional
        Minimum x value.
    right : float, optional
        Maximum x value.
    n_grid : int, optional
        Number of points to evaluate density on.
    save_fig : bool, optional
        Whether to save plot to disk.
    file_name : str
        Name of file to save plot.

    Returns
    -------
    Axes
        Axes objects for plot.
    """
    gbt_col = "gbt_future_30m_mean_sq_ret_pred"
    gbt_pred = df[gbt_col].values

    # medoid
    x_mean = X.mean(dim=0, keepdim=True)
    distances = torch.norm(X - x_mean, dim=1)
    medoid_idx = int(torch.argmin(distances))
    x_medoid = X[medoid_idx:medoid_idx+1]

    grid = np.linspace(left, right, n_grid)
    fig, ax = plt.subplots(figsize=(6,4))

    palette = sns.color_palette("tab10", n_colors=len(models)+1)

    for i, (name, model) in enumerate(models.items()):
        model.eval()
        device = next(model.parameters()).device
        x_medoid_dev = x_medoid.to(device)

        with torch.no_grad():
            w, mu, sigma = model(x_medoid_dev)

        w = w.cpu().numpy()[0]
        mu = mu.cpu().numpy()[0]
        sigma = sigma.cpu().numpy()[0]

        pdf = np.zeros_like(grid)

        for k in range(len(w)):
            pdf += w[k] * norm.pdf(grid, mu[k], sigma[k])

        ax.plot(grid, pdf, label=name, linewidth=1, color=palette[i+1])

    sigma = np.sqrt(gbt_pred[medoid_idx] * 30)

    ax.plot(grid,
             norm.pdf(grid, 0, sigma),
             linewidth=1,
             label="GBT Two-Stage",
             color=palette[0])

    ax.axvline(0, color="gray", linestyle=":")
    ax.set_xlim(left, right)
    ax.set_xlabel("return")
    ax.set_ylabel("density")
    ax.set_title("Predicted Return Density")
    ax.legend(loc="upper right", fontsize=7)
    fig.tight_layout()

    if save_fig:
        plot_path = Path("../plots")
        plot_path.mkdir(parents=True, exist_ok=True)

        file_path = plot_path / file_name
        fig.savefig(file_path, bbox_inches="tight")
        
    plt.show()
    plt.close(fig)

    return ax
