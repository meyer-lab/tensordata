"""Shared helpers for loading and reshaping the packaged datasets."""

from pathlib import Path

import pandas as pd
import xarray as xr

PACKAGE_DIR = Path(__file__).parent


def read_dataset_csv(dataset_dir: str, name: str, **kwargs) -> pd.DataFrame:
    """Read a CSV file out of one of the packaged dataset directories."""
    kwargs.setdefault("comment", "#")
    return pd.read_csv(PACKAGE_DIR / dataset_dir / f"{name}.csv", **kwargs)


def melt_to_xarray(
    df: pd.DataFrame,
    id_vars: list[str],
    split_into: list[str],
    sep: str,
    index: list[str],
    *,
    rsplit: bool = False,
    maxsplit: int = -1,
    rename: dict[str, str] | None = None,
) -> xr.DataArray:
    """Melt a wide dataframe and split its column names into coordinates.

    Many datasets store one column per coordinate combination, e.g. a
    column named "IgG1 RBD". This melts that wide layout to long form,
    splits the resulting column name on `sep` into `split_into` coordinate
    columns, and returns the result indexed by `index` as an xarray
    DataArray.
    """
    df = pd.melt(df, id_vars=id_vars, var_name="Measurement", value_name="Value")
    splitter = df["Measurement"].str.rsplit if rsplit else df["Measurement"].str.split
    df[split_into] = splitter(sep, n=maxsplit, expand=True)
    df = df.drop(columns="Measurement")

    if rename:
        df = df.rename(columns=rename)

    return df.set_index(index).to_xarray()["Value"]
