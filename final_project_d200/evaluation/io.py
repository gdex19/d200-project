import pandas as pd
from pathlib import Path

def write_grid_search_results(df_results: pd.DataFrame, file_name: str = "hyperparams.csv") -> None:
    """
    Write grid search results to results directory.

    Parameters
    ----------
    df_results : pd.DataFrame
        df containing grid search results.
    file_name : str, optional
        Name of results file.

    Returns
    -------
    None
    """
    result_path = Path("../results")
    result_path.mkdir(parents=True, exist_ok=True)

    hyperparam_path = result_path / file_name

    if hyperparam_path.is_file():
        df_results.to_csv(str(hyperparam_path), mode="a", header=False, index=False)
        print(f"Appended results to {file_name}")
    else:
        df_results.to_csv(str(hyperparam_path), index=False)
        print(f"Wrote results to new file {file_name}")


def read_grid_search_results(file_name: str = "hyperparams.csv") -> pd.DataFrame:
    """
    Read grid search results to df, returns empty if file does not exist in
    results directory.

    Parameters
    ----------
    file_name : str, optional
        Name of results file.

    Returns
    -------
    pd.DataFrame
        df of results, empty if file does not exist.
    """
    result_path = Path("../results")
    if not result_path.is_dir():
        return pd.DataFrame({})
    else:
        hyperparam_path = result_path / file_name
        if hyperparam_path.is_file():
            df = pd.read_csv(str(hyperparam_path))
            return df
        else:
            return pd.DataFrame({})