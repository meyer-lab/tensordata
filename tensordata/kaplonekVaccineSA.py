from pathlib import Path

import numpy as np
import pandas as pd
import re
import xarray as xr

DATA_DIR = Path(__file__).parent / "kaplonekVaccineSA2023"

df = pd.read_csv(DATA_DIR / "luminex.csv")


def data():
    # Separate metadata and luminex data columns
    luminex_cols = [col for col in df.columns if re.match("^FcR|^Ig[G|M|A]", col)]
    meta_cols = [col for col in df.columns if col not in luminex_cols]

    # Create 2D DataArray for metadata
    meta_da = xr.DataArray(
        df[meta_cols].values,
        coords={"Subject": df.index, "Metadata": meta_cols},
        dims=["Subject", "Metadata"],
    )

    # Reshape luminex data for 3D DataArray
    # Parsing Antigen and Detection from column names
    antigens, detections = zip(*[col.split("_", 1) for col in luminex_cols])
    luminex_data = df[luminex_cols].values.reshape(
        len(df), len(set(antigens)), len(set(detections))
    )

    # Create 3D DataArray for luminex data
    luminex_da = xr.DataArray(
        luminex_data,
        coords={
            "Subject": df.index,
            "Antigen": list(
                dict.fromkeys(antigens)
            ),  # remove duplicates, preserve order
            "Receptor": list(dict.fromkeys(detections)),
        },
        dims=["Subject", "Antigen", "Receptor"],
    )

    assert np.all(luminex_da.Subject.values == meta_da.Subject.values)

    return xr.Dataset({"Meta": meta_da, "Luminex": luminex_da})
