from matplotlib.axes import Axes
import pandas as pd
import numpy as np
from numpy.typing import NDArray
import seaborn as sns


def plot_train_and_val_loss(
    results: NDArray,
    title: str = "Training and Validation Loss Curves"
) -> Axes:
    """
    Plot training and validation for each epoch. Assumes training and validation
    losses are in column 1 and 2 respectively.

    Parameters
    ----------
    results : NDArray
        Array of training results by epoch.
    title : str
        Title for plot.

    Returns
    -------
    Axes
        Matplotlib axes with loss curves.
    """
    train_loss = results[:, 1]
    val_loss = results[:, 2]

    df_plots = pd.DataFrame({
        "epoch": np.arange(len(results)),
        "train_loss": train_loss,
        "val_loss": val_loss,
    })

    ax = sns.lineplot(data=df_plots[["train_loss", "val_loss"]])

    ax.lines[0].set_linestyle("-")
    ax.lines[0].set_linestyle("--")
    min_idx_val_loss = int(np.argmin(val_loss))
    ax.axvline(min_idx_val_loss, linestyle="--", label="best val epoch", color="black")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(title)
    ax.legend()

    return ax