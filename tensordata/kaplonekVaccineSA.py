import re
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

DATA_DIR = Path(__file__).parent / "kaplonekVaccineSA2023"


def data():
    # Separate metadata and luminex data columns
    df = pd.read_csv(DATA_DIR / "luminex.csv")

    luminex_cols = [col for col in df.columns if re.match("^FcR|^Ig[G|M|A]", col)]
    meta_cols = [col for col in df.columns if col not in luminex_cols]

    # Create 2D DataArray for metadata
    meta_da = xr.DataArray(
        df[meta_cols].values,
        coords={"Subject": df.index, "Metadata": meta_cols},
        dims=["Subject", "Metadata"],
    )

    detections, antigens = zip(*[col.split("_", 1) for col in luminex_cols])
    detections = list(dict.fromkeys(detections))
    antigens = list(dict.fromkeys(antigens))

    luminex_da = xr.DataArray(
        np.full((len(df), len(antigens), len(detections)), np.nan),
        coords={
            "Subject": df.index,
            "Antigen": antigens,
            "Receptor": detections,
        },
        dims=["Subject", "Antigen", "Receptor"],
    )

    for s in df.index:
        for ag in antigens:
            for d in detections:
                luminex_da.loc[s, ag, d] = df.loc[s, f"{d}_{ag}"]

    # Create 3D DataArray for luminex data
    ds = xr.Dataset({"Meta": meta_da, "Luminex": luminex_da})

    return ds
