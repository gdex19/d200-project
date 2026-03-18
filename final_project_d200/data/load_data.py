import pandas as pd
from pathlib import Path


def read_data(symbol: str) -> pd.DataFrame:
    """
    Read the parquet into a pandas DataFrame.

    Parameters
    ----------
    symbol: str
        Parquet of interest.

    Returns
    -------
    pd.DataFrame
    """
    root_path = Path(__file__).parent.parent.parent
    data_path = root_path / "data"
    fp = data_path / f"{symbol}.pq"

    if not fp.is_file():
        raise Exception("Data not found")

    return pd.read_parquet(fp)

