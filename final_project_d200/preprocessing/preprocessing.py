from final_project_d200.feature_engineering import (
    add_normalized_past_returns, add_future_returns,
    add_past_returns, add_past_sq_returns, add_future_mean_abs_returns,
    add_future_mean_sq_returns, add_neutral_up_down_label,
    add_past_abs_returns, add_day_of_week
)
import pandas as pd

NUM_FEATURES = [
    "past_1m_ret",
    "past_5m_ret",
    "past_30m_ret",
    "past_60m_ret",
    "past_120m_ret",
    "past_1m_sq_ret",
    "past_5m_sq_ret",
    "past_30m_sq_ret",
    "past_60m_sq_ret",
    "past_120m_sq_ret",
    "past_1m_abs_ret",
    "past_5m_abs_ret",
    "past_30m_abs_ret",
    "past_60m_abs_ret",
    "past_120m_abs_ret",
    "past_1440m_vol",
    "past_10080m_vol",
    "past_43200m_vol",
    "past_50m_span_ewm_vol",
    "past_50m_vol_pct_change",
    "past_30m_quote_volume",
    "past_30m_trades",
    "quote_volume_pct_change",
    "is_us_trading_day",
    "is_eu_trading_day",
    "is_uk_trading_day",
    "is_jp_trading_day",
    "is_cn_trading_day",
    "is_hk_trading_day",
    "is_us_dst",
    "is_eu_dst"
]

CAT_FEATURES = ["time_of_day", 
                "event_code", 
                "day_of_week"
]

TARGET_COL = "future_30m_ret"

TARGET_COLS_PRE = ["future_5m_mean_sq_ret", "future_15m_mean_sq_ret", 
              "future_30m_mean_sq_ret", "future_30m_mean_abs_ret",
              "future_30m_neutral_up_down"]

def add_features_responders(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add features and responders to df.

    Parameters
    ----------
    df : pd.DataFrame
        Bitcoin price df.

    Returns
    -------
    pd.DataFrame
        df with features and responders added.
    """
    df = add_normalized_past_returns(df, [1, 5, 30, 60, 120])
    df = add_future_returns(df, [30])
    df = add_past_returns(df, [1, 5, 30, 60, 120])
    df = add_past_sq_returns(df, [1, 5, 30, 60, 120])
    df = add_future_mean_abs_returns(df, [30])
    df = add_future_mean_sq_returns(df, [5, 15, 30])
    df = add_neutral_up_down_label(df, 0.001, 30)
    df = add_past_abs_returns(df, [1, 5, 30, 60, 120])
    df = add_day_of_week(df)

    return df

def downsample(df: pd.DataFrame, minutes: set[int]) -> pd.DataFrame:
    """
    Return dataframe with datapoints from specified minutes within each hour.

    Parameters
    ----------
    df : pd.DataFrame
        Bitcoin price df.
    
    minutes : set[int]
        Minutes to keep in downsample.


    Returns
    -------
    pd.DataFrame
        Downsampled df.
    """
    return df[df["minute"].isin(minutes)]