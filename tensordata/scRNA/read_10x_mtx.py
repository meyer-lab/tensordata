from enum import Enum
from pathlib import Path
from typing import Literal

import anndata
import pandas as pd


class Empty(Enum):
    token = 0

    def __repr__(self) -> str:
        return "_empty"


_empty = Empty.token


def read_10x_mtx(
    path: Path | str,
    *,
    var_names: Literal["gene_symbols", "gene_ids"] = "gene_symbols",
    make_unique: bool = True,
    cache: bool = False,
    cache_compression: Literal["gzip", "lzf"] | None | Empty = _empty,
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
        cache=cache,
        cache_compression=cache_compression,
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
    cache: bool = False,
    cache_compression: Literal["gzip", "lzf"] | None | Empty = _empty,
    prefix: str = "",
    is_legacy: bool,
) -> anndata.AnnData:
    """
    Read mex from output from Cell Ranger v2- or v3+
    """
    suffix = "" if is_legacy else ".gz"
    adata = read(
        path / f"{prefix}matrix.mtx{suffix}",
        cache=cache,
        cache_compression=cache_compression,
    ).T  # transpose the data
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


def read(
    filename: Path | str,
    backed: Literal["r", "r+"] | None = None,
    *,
    sheet: str | None = None,
    ext: str | None = None,
    delimiter: str | None = None,
    first_column_names: bool = False,
    backup_url: str | None = None,
    cache: bool = False,
    cache_compression: Literal["gzip", "lzf"] | None | Empty = _empty,
    **kwargs,
) -> anndata.AnnData:
    """\
    Read file and return :class:`~anndata.AnnData` object.

    To speed up reading, consider passing ``cache=True``, which creates an hdf5
    cache file.

    Parameters
    ----------
    filename
        If the filename has no file extension, it is interpreted as a key for
        generating a filename via ``sc.settings.writedir / (filename +
        sc.settings.file_format_data)``.  This is the same behavior as in
        ``sc.read(filename, ...)``.
    backed
        If ``'r'``, load :class:`~anndata.AnnData` in ``backed`` mode instead
        of fully loading it into memory (`memory` mode). If you want to modify
        backed attributes of the AnnData object, you need to choose ``'r+'``.
    sheet
        Name of sheet/table in hdf5 or Excel file.
    ext
        Extension that indicates the file type. If ``None``, uses extension of
        filename.
    delimiter
        Delimiter that separates data within text file. If ``None``, will split at
        arbitrary number of white spaces, which is different from enforcing
        splitting at any single white space ``' '``.
    first_column_names
        Assume the first column stores row names. This is only necessary if
        these are not strings: strings in the first column are automatically
        assumed to be row names.
    backup_url
        Retrieve the file from an URL if not present on disk.
        
    cache
        If `False`, read from source, if `True`, read from fast 'h5ad' cache.
    cache_compression
        See the h5py :ref:`dataset_compression`.
        (Default: `settings.cache_compression`)
    kwargs
        Parameters passed to :func:`~anndata.read_loom`.

    Returns
    -------
    An :class:`~anndata.AnnData` object
    """
    filename = Path(filename)  # allow passing strings
    if is_valid_filename(filename):
        return _read(
            filename,
            backed=backed,
            sheet=sheet,
            ext=ext,
            delimiter=delimiter,
            first_column_names=first_column_names,
            backup_url=backup_url,
            cache=cache,
            cache_compression=cache_compression,
            **kwargs,
        )
    # generate filename and read to dict
    filekey = str(filename)
    filename = settings.writedir / (filekey + "." + settings.file_format_data)
    if not filename.exists():
        raise ValueError(
            f"Reading with filekey {filekey!r} failed, "
            f"the inferred filename {filename!r} does not exist. "
        )
    return anndata.read_h5ad(filename, backed=backed)

def _read(
    filename: Path,
    *,
    backed=None,
    sheet=None,
    ext=None,
    delimiter=None,
    first_column_names=None,
    backup_url=None,
    cache=False,
    cache_compression=None,
    suppress_cache_warning=False,
    **kwargs,
):
    if ext is not None and ext not in avail_exts:
        raise ValueError(
            "Please provide one of the available extensions.\n" f"{avail_exts}"
        )
    else:
        ext = is_valid_filename(filename, return_ext=True)
    is_present = _check_datafile_present_and_download(filename, backup_url=backup_url)
    if not is_present:
        logg.debug(f"... did not find original file {filename}")
    # read hdf5 files
    if ext in {"h5", "h5ad"}:
        if sheet is None:
            return read_h5ad(filename, backed=backed)
        else:
            logg.debug(f"reading sheet {sheet} from file {filename}")
            return read_hdf(filename, sheet)
    # read other file types
    path_cache: Path = settings.cachedir / _slugify(filename).replace(
        f".{ext}", ".h5ad"
    )
    if path_cache.suffix in {".gz", ".bz2"}:
        path_cache = path_cache.with_suffix("")
    if cache and path_cache.is_file():
        logg.info(f"... reading from cache file {path_cache}")
        return read_h5ad(path_cache)

    if not is_present:
        raise FileNotFoundError(f"Did not find file {filename}.")
    logg.debug(f"reading {filename}")
    if not cache and not suppress_cache_warning:
        logg.hint(
            "This might be very slow. Consider passing `cache=True`, "
            "which enables much faster reading from a cache file."
        )

    adata = read_mtx(filename)

    return adata

