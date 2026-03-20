import pandas as pd
import numpy as np
from numpy.typing import NDArray
from final_project_d200.options import price_call_normal


def simulate_call_trades_normal(return_strike: float, returns: pd.Series, sigmas_one: NDArray, sigmas_two: NDArray,
                                tol: float = 1e-5) -> list:
    """
    Simulate two strategies trading at the midpoint between their fair values.

    Parameters
    ----------
    return_strike : float
        Option strike in return space.
    returns : pd.Series
        Realized returns.
    sigmas_one : NDArray
        Future return std estimates from model one.
    sigmas_two : NDArray
        Future return std estimates from model two.
    tol : float, optional
        Tolerance for equality/no trade.
    
    Returns
    -------
    list[float]
        PnL for each time step.
    """
    pnl = []

    for i in range(len(sigmas_one)):
        price_one = price_call_normal(sigmas_one[i], return_strike)
        price_two = price_call_normal(sigmas_two[i], return_strike)
        trade_price = 0.5 * (price_one + price_two)

        payoff = max(returns.iloc[i] - return_strike, 0)

        direction = np.where(np.abs(price_one - price_two) < tol, 0, np.where(price_one < price_two, -1, 1))

        pnl.append(direction * (payoff - trade_price))

    return pnl
