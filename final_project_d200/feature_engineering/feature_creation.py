import pandas as pd
import numpy as np

def add_past_returns(df: pd.DataFrame, windows: list[int], log: bool = False) -> pd.DataFrame:
    """
    Return dataframe with past returns computed for each window (in minutes). 
    Assumes data is every minute.

    Parameters
    ----------
    df : pd.DataFrame
        Bitcoin price df.
    windows : list[int]
        Past windows in minutes.
    log : bool, optional
        Whether to compute in log-return or return space.

    Returns
    -------
    pd.DataFrame
        df with return columns added.
    """
    df = df.copy()
    for window in windows:
        ret = df["open"] / df["open"].shift(window) - 1
        df[f"past_{window}m_{'log_' if log else ''}ret"] = np.log(1 + ret) if log else ret
    return df

def add_past_sq_returns(df: pd.DataFrame, windows: list[int]) -> pd.DataFrame:
    """
    Return dataframe with past squared returns computed for each window (in minutes). 
    Assumes data is every minute.

    Parameters
    ----------
    df : pd.DataFrame
        Bitcoin price df.
    windows : list[int]
        Past windows in minutes.

    Returns
    -------
    pd.DataFrame
        df with past squared return columns added.
    """
    df = df.copy()
    for window in windows:
        ret = df["open"] / df["open"].shift(window) - 1
        df[f"past_{window}m_sq_ret"] = ret ** 2
    return df

def add_future_sq_returns(df: pd.DataFrame, windows: list[int]) -> pd.DataFrame:
    """
    Return dataframe with future squared returns computed for each window (in minutes). 
    Assumes data is every minute.

    Parameters
    ----------
    df : pd.DataFrame
        Bitcoin price df.
    windows : list[int]
        Forward windows in minutes.

    Returns
    -------
    pd.DataFrame
        df with future squared return columns added.
    """
    df = df.copy()
    for window in windows:
        ret = df["open"].shift(-window) / df["open"] - 1
        df[f"future_{window}m_sq_ret"] = ret ** 2
    return df

def add_future_mean_sq_returns(df: pd.DataFrame, windows: list[int]) -> pd.DataFrame:
    """
    Return dataframe with future mean 1m sq returns for each window (in minutes). 
    Assumes data is every minute.

    Parameters
    ----------
    df : pd.DataFrame
        Bitcoin price df.
    windows : list[int]
        Forward windows in minutes.

    Returns
    -------
    pd.DataFrame
        df with future mean squared return columns added.
    """
    df = df.copy()
    ret = df["open"] / df["open"].shift(1) - 1
    df["past_1m_sq_ret"] = ret ** 2
    
    for window in windows:
        df[f"future_{window}m_mean_sq_ret"] = df["past_1m_sq_ret"].rolling(window).mean().shift(-window)
    
    return df

def add_past_mean_sq_returns(df: pd.DataFrame, windows: list[int]) -> pd.DataFrame:
    """
    Return dataframe with past mean 1m sq returns for each window (in minutes). 
    Assumes data is every minute.

    Parameters
    ----------
    df : pd.DataFrame
        Bitcoin price df.
    windows : list[int]
        Past windows in minutes.

    Returns
    -------
    pd.DataFrame
        df with past mean squared return columns added.
    """
    df = df.copy()
    ret = df["open"] / df["open"].shift(1) - 1
    df["past_1m_sq_ret"] = ret ** 2
    
    for window in windows:
        df[f"past_{window}m_mean_sq_ret"] = df["past_1m_sq_ret"].rolling(window).mean()
    
    return df

def add_normalized_past_returns(df: pd.DataFrame, windows: list[int]) -> pd.DataFrame:
    """
    Return dataframe with past returns normalized by mean 1m sq returns over that period.
    for each window. Past vol is normalized to window. Assumes data is every minute.

    Parameters
    ----------
    df : pd.DataFrame
        Bitcoin price df.
    windows : list[int]
        Past windows in minutes.

    Returns
    -------
    pd.DataFrame
        df with normalized return columns added.
    """
    df = df.copy()
    df = add_past_returns(df, windows=windows)
    df = add_past_mean_sq_returns(df, windows=windows)
    for window in windows:
        df[f"past_{window}m_norm_ret"] = df[f"past_{window}m_ret"] / np.sqrt(df[f"past_{window}m_mean_sq_ret"] * window)
    return df

def add_future_returns(df: pd.DataFrame, windows: list[int], log: bool = False) -> pd.DataFrame:
    """
    Return dataframe with future returns for each window. Assumes data is every minute.

    Parameters
    ----------
    df : pd.DataFrame
        Bitcoin price df.
    windows : list[int]
        Forward windows in minutes.
    log : bool, optional
        Whether to compute in log-return or return space.

    Returns
    -------
    pd.DataFrame
        df with future return columns added.
    """
    df = df.copy()
    for window in windows:
        ret = df["open"].shift(-window) / df["open"] - 1
        df[f"future_{window}m_{'log_' if log else ''}ret"] = np.log(1 + ret) if log else ret
    return df

def add_past_abs_returns(df: pd.DataFrame, windows: list[int]) -> pd.DataFrame:
    """
    Return dataframe with past abs returns for each window. Assumes data is every minute.

    Parameters
    ----------
    df : pd.DataFrame
        Bitcoin price df.
    windows : list[int]
        Past windows in minutes.

    Returns
    -------
    pd.DataFrame
        Dataframe with past absolute return columns added.
    """
    df = df.copy()
    df = add_past_returns(df, windows)
    for window in windows:
        df[f"past_{window}m_abs_ret"] = np.abs(df[f"past_{window}m_ret"])
    return df

def add_future_mean_abs_returns(df: pd.DataFrame, windows: list[int]) -> pd.DataFrame:
    """
    Return dataframe with mean abs value of 1m future returns for each window. Assumes data is every minute.

    Parameters
    ----------
    df : pd.DataFrame
        Bitcoin price df.
    windows : list[int]
        Forward windows in minutes.

    Returns
    -------
    pd.DataFrame
        Dataframe with future mean absolute return columns added.
    """
    df = df.copy()
    df["past_1m_abs_ret"] = np.abs(df["open"] / df["open"].shift(1) - 1)
    for window in windows:
        df[f"future_{window}m_mean_abs_ret"] = df["past_1m_abs_ret"].rolling(window).mean().shift(-window)
    return df


def add_lagged_feature(df: pd.DataFrame, col: str, max_lag: int) -> pd.DataFrame:
    """
    Return dataframe with lagged col feature.

    Parameters
    ----------
    df : pd.DataFrame
        df with features.
    col : str
        Feature column to lag.
    max_lag : int
        Maximum lag.

    Returns
    -------
    pd.DataFrame
        Dataframe with lagged feature columns added.
    """
    df = df.copy()
    for lag in range(1, max_lag + 1):
        new_col = f"{col}_lag_{lag}"
        df[new_col] = df[col].shift(lag)
    return df

def add_neutral_up_down_label(df: pd.DataFrame, threshold: float, window: int) -> pd.DataFrame:
    """
    Return dataframe with numerical neutral (1), up (2), down (0) labeling for returns within [-threshold, threshold], 
    (threshold, inf), and (-inf, threshold) respectively.

    Parameters
    ----------
    df : pd.DataFrame
        Bitcoin price df.
    threshold : float
        Threshold for buckets in abs.
    window : int
        Forward window in minutes.

    Returns
    -------
    pd.DataFrame
        Dataframe with label column added.
    """
    df = df.copy()
    df = add_future_returns(df, [window])
    df[f"future_{window}m_neutral_up_down"] = np.where(df[f"future_{window}m_ret"] > threshold, 2, np.where(df[f"future_{window}m_ret"] < -threshold, 0, 1))
    return df

def add_day_of_week(df: pd.DataFrame,) -> pd.DataFrame:
    """
    Return dataframe with day of week categorical feature.

    Parameters
    ----------
    df : pd.DataFrame
        Bitcoin price df with date column.

    Returns
    -------
    pd.DataFrame
        Dataframe with day of week feature added.
    """
    df = df.copy()
    date_str = df["date"].str
    df["day_of_week"] = pd.to_datetime({"year": date_str[:4], "month": date_str[5:7], "day":date_str[8:10]}).dt.day_of_week
    return df