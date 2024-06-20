from os import PathLike, fspath
from pathlib import Path
from typing import Literal

import anndata
import pandas as pd
from scipy.io import mmread
from scipy.sparse import csr_matrix

text_exts = {
    "csv",
    "tsv",
    "tab",
    "data",
    "txt",  # these four are all equivalent
}
avail_exts = {
    "anndata",
    "xlsx",
    "h5",
    "h5ad",
    "mtx",
    "mtx.gz",
    "soft.gz",
    "loom",
} | text_exts


def read_10x_mtx(
    path: Path | str,
    *,
    var_names: Literal["gene_symbols", "gene_ids"] = "gene_symbols",
    make_unique: bool = True,
    gex_only: bool = True,
    prefix: str | None = None,
) -> anndata.AnnData:
    """
    Read 10x-Genomics-formatted mtx directory.

    Parameters
    ----------
    path
        Path to directory for `.mtx` and `.tsv` files,
        e.g. './filtered_gene_bc_matrices/hg19/'.
    var_names
        The variables index.
    make_unique
        Whether to make the variables index unique by appending '-1',
        '-2' etc. or not.
    cache
        If `False`, read from source, if `True`, read from fast 'h5ad' cache.
    cache_compression
        See the h5py :ref:`dataset_compression`.
        (Default: `settings.cache_compression`)
    gex_only
        Only keep 'Gene Expression' data and ignore other feature types,
        e.g. 'Antibody Capture', 'CRISPR Guide Capture', or 'Custom'
    prefix
        Any prefix before `matrix.mtx`, `genes.tsv` and `barcodes.tsv`. For instance,
        if the files are named `patientA_matrix.mtx`, `patientA_genes.tsv` and
        `patientA_barcodes.tsv` the prefix is `patientA_`.
        (Default: no prefix)

    Returns
    -------
    An :class:`~anndata.AnnData` object
    """
    path = Path(path)
    prefix = "" if prefix is None else prefix
    is_legacy = (path / f"{prefix}genes.tsv").is_file()
    adata = _read_10x_mtx(
        path,
        var_names=var_names,
        make_unique=make_unique,
        prefix=prefix,
        is_legacy=is_legacy,
    )
    if is_legacy or not gex_only:
        return adata
    gex_rows = adata.var["feature_types"] == "Gene Expression"
    return adata[:, gex_rows].copy()


def _read_10x_mtx(
    path: Path,
    *,
    var_names: Literal["gene_symbols", "gene_ids"] = "gene_symbols",
    make_unique: bool = True,
    prefix: str = "",
    is_legacy: bool,
) -> anndata.AnnData:
    """
    Read mex from output from Cell Ranger v2- or v3+
    """
    suffix = "" if is_legacy else ".gz"

    filepath = path / f"{prefix}matrix.mtx{suffix}"
    assert is_valid_filename(filepath)
    adata = read_mtx(filepath).T
    genes = pd.read_csv(
        path / f"{prefix}{'genes' if is_legacy else 'features'}.tsv{suffix}",
        header=None,
        sep="\t",
    )
    if var_names == "gene_symbols":
        var_names_idx = pd.Index(genes[1].values)
        if make_unique:
            var_names_idx = anndata.utils.make_index_unique(var_names_idx)
        adata.var_names = var_names_idx
        adata.var["gene_ids"] = genes[0].values
    elif var_names == "gene_ids":
        adata.var_names = genes[0].values
        adata.var["gene_symbols"] = genes[1].values
    else:
        raise ValueError("`var_names` needs to be 'gene_symbols' or 'gene_ids'")
    if not is_legacy:
        adata.var["feature_types"] = genes[2].values
    barcodes = pd.read_csv(path / f"{prefix}barcodes.tsv{suffix}", header=None)
    adata.obs_names = barcodes[0].values
    return adata


def is_valid_filename(filename: Path, return_ext=False):
    """Check whether the argument is a filename."""
    ext = filename.suffixes

    # cases for gzipped/bzipped text files
    if len(ext) == 2 and ext[0][1:] in text_exts and ext[1][1:] in ("gz", "bz2"):
        return ext[0][1:] if return_ext else True
    elif ext and ext[-1][1:] in avail_exts:
        return ext[-1][1:] if return_ext else True
    elif "".join(ext) == ".soft.gz":
        return "soft.gz" if return_ext else True
    elif "".join(ext) == ".mtx.gz":
        return "mtx.gz" if return_ext else True
    elif not return_ext:
        return False
    raise ValueError(
        f"""\
{filename!r} does not end on a valid extension.
Please, provide one of the available extensions.
{avail_exts}
Text files with .gz and .bz2 extensions are also supported.\
"""
    )


def read_mtx(filename: PathLike, dtype: str = "float32") -> anndata.AnnData:
    """\
    Read `.mtx` file.

    Parameters
    ----------
    filename
        The filename.
    dtype
        Numpy data type.
    """

    # could be rewritten accounting for dtype to be more performant
    X = mmread(fspath(filename)).astype(dtype)

    X = csr_matrix(X)
    return anndata.AnnData(X)
