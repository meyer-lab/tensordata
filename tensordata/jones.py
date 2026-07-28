import numpy as np
import pandas as pd

from .utils import PACKAGE_DIR, read_dataset_csv

DATASET_DIR = "jones2017"
RA_DATAFRAME_PATH = PACKAGE_DIR / DATASET_DIR / "RA_DataFrame.csv"

donorDict = {
    "1869": "RA",
    "1931": "RA",
    "2159": "RA",
    "2586": "N",
    "2645": "N",
    "2708": "RA",
    "2759": "N",
}


def process_RA_Tensor():
    """Structures all Rheumatoid Arthritis Synovial Fibroblast data into a
    usable tensor"""
    RA_df = pd.DataFrame()
    rep_list = [1, 2]
    stimulants = ["IL1-a", "TNF-a", "PolyIC"]

    for donor in donorDict:
        for rep in rep_list:
            file_name = f"SF_Donor_{donor}_{donorDict[donor]}_Rep{rep}"
            raw_data = read_dataset_csv(DATASET_DIR, file_name)
            avg_data = (
                raw_data.groupby(["Sample Name"]).mean(numeric_only=True).reset_index()
            )

            for index, row in avg_data.iterrows():
                expString = row["Sample Name"].split(", ")
                if len(expString) == 2:
                    avg_data.loc[index, "Stimulant"] = expString[0]
                    avg_data.loc[index, "Inhibitor"] = expString[1]
                elif expString[0] == "bufferonly":
                    avg_data.loc[index, "Stimulant"] = "Buffer"
                    avg_data.loc[index, "Inhibitor"] = "Buffer"
                elif len(row["Sample Name"].split("_")) == 2:
                    avg_data.loc[index, "Stimulant"] = row["Sample Name"].split("_")[0]
                    avg_data.loc[index, "Inhibitor"] = "Spike"
            avg_data = avg_data.drop(
                ["Sample Name", "ExpPlate", "LMXassayPlate"], axis=1
            )
            cytokines = avg_data.iloc[:, 0:-2].columns

            # Background (Spike and Otherwise) Subtraction
            for stimulant in stimulants:
                if stimulant in avg_data.columns:
                    avg_data.loc[(avg_data["Stimulant"] == stimulant), stimulant][
                        stimulant
                    ] = np.nan
                spike_row = (
                    avg_data.loc[
                        (avg_data["Stimulant"] == stimulant)
                        & (avg_data["Inhibitor"] == "Spike")
                    ]
                    .reset_index()
                    .drop("index", axis=1)
                    .iloc[0, 0:-2]
                )
                basal_row = (
                    avg_data.loc[
                        (avg_data["Stimulant"] == "nostim")
                        & (avg_data["Inhibitor"] == "noinh")
                    ]
                    .reset_index()
                    .drop("index", axis=1)
                    .iloc[0, 0:-2]
                )
                for inh in avg_data.loc[
                    (avg_data["Stimulant"] == stimulant)
                ].Inhibitor.unique():
                    stim_inh_row = avg_data.loc[
                        (avg_data["Stimulant"] == stimulant)
                        & (avg_data["Inhibitor"] == inh)
                    ].iloc[0, 0:-2]
                    basal_spike_df = pd.concat(
                        [stim_inh_row - spike_row, basal_row], axis=1
                    ).transpose()
                    avg_data.loc[
                        (avg_data["Stimulant"] == stimulant)
                        & (avg_data["Inhibitor"] == inh),
                        cytokines,
                    ] = basal_spike_df.max().to_frame().transpose().values
            avg_data[cytokines] = np.log(avg_data[cytokines].values)
            avg_data[cytokines] -= avg_data.loc[
                avg_data.Stimulant == "Buffer", cytokines
            ].values
            avg_data[cytokines] = avg_data[cytokines].clip(lower=0)

            avg_data["Donor"] = donor
            avg_data["Replicate"] = rep
            avg_data["Status"] = donorDict[donor]
            avg_data = pd.melt(
                avg_data,
                id_vars=["Stimulant", "Inhibitor", "Donor", "Status", "Replicate"],
                value_vars=cytokines,
            ).rename({"variable": "Cytokine", "value": "Log MFI"}, axis=1)
            avg_data = avg_data.loc[
                (avg_data.Stimulant != "Buffer") & (avg_data.Inhibitor != "Spike")
            ]
            RA_df = pd.concat([RA_df, avg_data], axis=0)

    # Average Over Replicates
    RA_df = (
        RA_df.groupby(["Stimulant", "Inhibitor", "Donor", "Status", "Cytokine"])[
            "Log MFI"
        ]
        .mean()
        .reset_index()
    )
    RA_df.to_csv(RA_DATAFRAME_PATH)
    return RA_df


def make_RA_Tensor():
    """Processes RA DataFrame into Xarray Tensor"""
    RA_df = (
        pd.read_csv(RA_DATAFRAME_PATH)
        if RA_DATAFRAME_PATH.exists()
        else process_RA_Tensor()
    )

    # A stimulant can't inhibit itself; every other Stimulant/Inhibitor/
    # Cytokine/Donor combination not present in RA_df is filled with NaN.
    RA_df = RA_df[RA_df["Stimulant"] != RA_df["Inhibitor"]]
    tensor = RA_df.set_index(["Stimulant", "Inhibitor", "Cytokine", "Donor"])[
        "Log MFI"
    ].to_xarray()

    # Normalize each cytokine's slice by its own maximum
    tensor = tensor / tensor.max(dim=["Stimulant", "Inhibitor", "Donor"], skipna=True)

    return tensor
