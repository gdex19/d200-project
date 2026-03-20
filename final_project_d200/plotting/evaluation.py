import torch
from torch import nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from scipy.stats import norm
import seaborn as sns

def plot_pit(model: nn.Module, model_name: str, data_loader: DataLoader, dataset_name: str) -> Axes:
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

    Returns
    -------
    Axes
        Axes objects for PIT plot.
    """
    model.eval()
    device = next(model.parameters()).device
    pits = []

    with torch.no_grad():
        for X, y in data_loader:

            X = X.to(device)
            y = y.to(device)

            w, mu, sigma = model(X)

            w = w.cpu().numpy()
            mu = mu.cpu().numpy()
            sigma = sigma.cpu().numpy()
            y = y.cpu().numpy()

            for i in range(len(y)):

                cdf = 0
                for k in range(w.shape[1]):
                    cdf += w[i,k] * norm.cdf(y[i], mu[i,k], sigma[i,k])

                pits.append(cdf)

    ax = sns.histplot(pits, bins=50, stat="density")
    plt.axhline(1, linestyle="--")
    plt.title(f"PIT Histogram, {model_name} on {dataset_name} data")
    plt.xlabel("F(y)")
    plt.ylabel("density")
    plt.show()

    return ax