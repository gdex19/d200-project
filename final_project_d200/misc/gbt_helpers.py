import pandas as pd
from pathlib import Path
from numpy.typing import NDArray
import numpy as np

def get_gbt_sigmas(df: pd.DataFrame) -> NDArray:
    """
    Convert gbt pred into sigma for normal.

    Parameters
    ----------
    df : pd.DataFrame
        df with gbt predictions.

    Returns
    -------
    NDArray
        Array of predicted sigmas for future 30m ret.
    """

    return np.sqrt((df["gbt_future_30m_mean_sq_ret_pred"] * 30).reset_index(drop=True))

