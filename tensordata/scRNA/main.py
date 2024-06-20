from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import anndata
import numpy.typing as npt
import pandas as pd
from scipy.sparse import csr_matrix

from .read_10x_mtx import read_10x_mtx

THIS_DIR = Path(__file__).parent


def gate_thomson_cells(X) -> npt.ArrayLike:
    cell_type_df_path = THIS_DIR / "thomson" / "ThomsonCellTypes.csv"
    """Manually gates cell types for Thomson UMAP"""
    cell_type_df = pd.read_csv(cell_type_df_path, index_col=0)
    cell_type_df.index.name = "cell_barcode"
    X.obs = X.obs.join(cell_type_df, on="cell_barcode", how="inner")
    X.obs["Cell Type"] = X.obs["Cell Type"].values.astype(str)
    X.obs["Cell Type2"] = X.obs["Cell Type2"].values.astype(str)
    return X


def import_thomson(
    anndata_path=Path("/opt") / "andrew" / "thomson_raw.h5ad",
) -> anndata.AnnData:
    """
    Import Thomson lab PBMC dataset.

    Args:
        gene_threshold: Minimum mean gene expression to keep a gene.
        anndata_path: Path to the anndata file. The default value is the path on
            Aretha.
    """
    # Cell barcodes, sample id of treatment and sample number (33482, 3)
    metafile_path = THIS_DIR / "thomson" / "meta.csv"
    doublet_path = THIS_DIR / "thomson" / "ThomsonDoublets.csv"
    metafile = pd.read_csv(metafile_path, usecols=[0, 1])

    X = anndata.read_h5ad(anndata_path)
    obs = X.obs.reset_index(names="cell_barcode")

    # Left merging should put the barcodes in order
    metafile = pd.merge(
        obs, metafile, on="cell_barcode", how="left", validate="one_to_one"
    )

    X.obs = pd.DataFrame(
        {
            "cell_barcode": metafile["cell_barcode"],
            "Condition": pd.Categorical(metafile["sample_id"]),
        }
    )

    doublet_df = pd.read_csv(doublet_path, index_col=0)
    doublet_df.index.name = "cell_barcode"
    X.obs = X.obs.join(doublet_df, on="cell_barcode", how="inner")

    singlet_indices = X.obs.loc[X.obs["doublet"] == 0].index.values
    X.obs = X.obs.reset_index(drop=True)
    X = X[singlet_indices, :]

    X.obs = X.obs.set_index("cell_barcode")
    gate_thomson_cells(X)

    return X


def import_lupus(
    anndata_path=Path("/opt") / "andrew" / "lupus" / "lupus.h5ad",
    protein_path=Path("/opt")
    / "andrew"
    / "lupus"
    / "Lupus_study_protein_adjusted.h5ad",
) -> anndata.AnnData:
    """
    Import Lupus PBMC dataset.

    Args:
        gene_threshold: Minimum mean gene expression to keep a gene.
        anndata_path: Path to the anndata file. The default value is the path on
            Aretha.
        protein_path: Path to the protein anndata file. The default value is the path
            on Aretha.

    -- columns from observation data:
    {'batch_cov': POOL (1-23) cell was processed in,
    'ind_cov': patient cell was derived from,
    'Processing_Cohort': BATCH (1-4) cell was derived from,
    'louvain': louvain cluster group assignment,
    'cg_cov': broad cell type,
    'ct_cov': lymphocyte-specific cell type,
    'L3': marks a balanced subset of batch 4 used for model training,
    'ind_cov_batch_cov': combination of patient and pool, proxy for sample ID,
    'Age':	age of patient,
    'Sex': sex of patient,
    'pop_cov': ancestry of patient,
    'Status': SLE status: healthy, managed, treated, or flare,
    'SLE_status': SLE status: healthy or SLE}

    """
    X = anndata.read_h5ad(anndata_path)
    X = anndata.AnnData(X.raw.X, X.obs, X.raw.var, X.uns, X.obsm)

    protein = anndata.read_h5ad(protein_path)
    protein_df = protein.to_df()

    X.obs = X.obs.rename(
        {
            "batch_cov": "pool",
            "ind_cov": "patient",
            "cg_cov": "Cell Type",
            "ct_cov": "cell_type_lympho",
            "ind_cov_batch_cov": "Condition",
            "Age": "age",
            "Sex": "sex",
            "pop_cov": "ancestry",
        },
        axis=1,
    )

    X.obs = X.obs.merge(protein_df, how="left", left_index=True, right_index=True)

    # get rid of IGTB1906_IGTB1906:dmx_count_AHCM2CDMXX_YE_0831 (only 3 cells)
    X = X[X.obs["Condition"] != "IGTB1906_IGTB1906:dmx_count_AHCM2CDMXX_YE_0831"]

    return X


def import_citeseq(
    data_dir_path=Path("/opt") / "andrew" / "HamadCITEseq",
) -> anndata.AnnData:
    """Imports 5 datasets from Hamad CITEseq."""
    files = ["control", "ic_pod1", "ic_pod7", "sc_pod1", "sc_pod7"]

    with ProcessPoolExecutor(max_workers=5) as executor:
        futures = [
            executor.submit(
                read_10x_mtx,
                data_dir_path / k,
                gex_only=False,
                make_unique=True,
            )
            for k in files
        ]

        data = {k: futures[i].result() for i, k in enumerate(files)}

    X = anndata.concat(data, merge="same", label="Condition")

    return X


def import_HTAN(
    HTAN_path=Path("/opt") / "extra-storage" / "HTAN",
) -> anndata.AnnData:
    """
    Imports Vanderbilt's HTAN 10X data.

    Args:
        HTAN_path: Path to the directory containing the HTAN data. The default
            value is the path on Aretha.
    """
    files = list(HTAN_path.glob("*.mtx.gz"))
    futures = []
    data = {}

    with ProcessPoolExecutor(max_workers=10) as executor:
        for filename in files:
            future = executor.submit(
                read_10x_mtx,
                HTAN_path,
                gex_only=False,
                make_unique=True,
                prefix=filename.stem.split("matrix")[0],
            )
            futures.append(future)

        for i, filename in enumerate(files):
            result = futures[i].result()
            data[filename.stem.split("_matrix")[0]] = result

    for filename in files:
        data[filename.stem.split("_matrix")[0]] = read_10x_mtx(
            HTAN_path,
            gex_only=False,
            make_unique=True,
            prefix=filename.stem.split("matrix")[0],
        )

    X = anndata.concat(data, merge="same", label="Condition")
    return X


def import_CCLE(
    data_dir_path=Path("/opt") / "extra-storage" / "asm" / "Heiser-barcode" / "CCLE",
) -> anndata.AnnData:
    """
    Imports barcoded cell data.

    Args:
        data_dir_path: Path to the directory containing the data. The default value
            is the path on Aretha.
    """
    # TODO: Still need to add gene names and barcodes.
    adatas = {
        "HCT116_1": anndata.read_text(
            Path(data_dir_path / "HCT116_tracing_T1.count_mtx.tsv")
        ).T,
        "HCT116_2": anndata.read_text(
            Path(data_dir_path / "HCT116_tracing_T2.count_mtx.tsv")
        ).T,
        "MDA-MB-231_1": anndata.read_text(
            Path(data_dir_path / "MDA-MB-231_tracing_T1.count_mtx.tsv")
        ).T,
        "MDA-MB-231_2": anndata.read_text(
            Path(data_dir_path / "MDA-MB-231_tracing_T2.count_mtx.tsv")
        ).T,
    }

    X = anndata.concat(adatas, label="sample")
    X.X = csr_matrix(X.X)

    return X


def import_cytokine(
    data_path=Path("/opt") / "extra-storage" / "Treg_h5ads" / "Treg_raw.h5ad",
) -> anndata.AnnData:
    """Import Meyer Cytokine PBMC dataset.

    Args:
        data_path: Path to the anndata file. The default value is the path on .

    -- columns from observation data:
    {'Stimulation': Cytokine and Dose}
    """
    X = anndata.read_h5ad(data_path)

    # Remove multiplexing identifiers
    X = X[:, ~X.var_names.str.match("^CMO3[0-9]{2}$")]  # type: ignore

    return X
