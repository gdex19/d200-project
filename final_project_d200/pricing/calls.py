from scipy import stats

def price_call_normal(sigma: float, strike_return: float, mu: float = 0.0):
    """
    Price one call option based on normal distribution with given params.

    Parameters
    ----------
    sigma : float
        Future return std estimates.
    strike_return : float
        Option strike in return space.
    mu : float, optional
        Future mean return estimate, defaults to 0.

    Returns
    -------
    float
        Price given parameters.
    """
    if sigma <= 0:
        return max(mu - strike_return, 0.0)
    z = (mu - strike_return) / sigma
    return (mu - strike_return) * stats.norm.cdf(z) + sigma * stats.norm.pdf(z)
