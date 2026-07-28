# tensordata

[![Test](https://github.com/meyer-lab/tensordata/actions/workflows/pytest.yml/badge.svg)](https://github.com/meyer-lab/tensordata/actions/workflows/pytest.yml)
[![Code Quality](https://github.com/meyer-lab/tensordata/actions/workflows/code_quality.yml/badge.svg)](https://github.com/meyer-lab/tensordata/actions/workflows/code_quality.yml)

A common repository for tensor structured datasets, developed and maintained by the [Meyer Lab](https://github.com/meyer-lab).

`tensordata` bundles a collection of published systems serology / immunology datasets together with loader functions that reshape each dataset's raw, wide-format CSVs into labeled, multi-dimensional [xarray](https://docs.xarray.dev/) `DataArray`/`Dataset` objects. This makes the data immediately usable for tensor factorization and other multi-way analyses without needing to hand-write reshaping code for every dataset.

## Installation

`tensordata` requires Python 3.14. It is not published on PyPI, so install it directly from GitHub.

With [uv](https://docs.astral.sh/uv/):

```bash
uv add git+https://github.com/meyer-lab/tensordata.git
```

or with pip:

```bash
pip install git+https://github.com/meyer-lab/tensordata.git
```

## Quick start

Each dataset lives in its own module and exposes a `data()` function (or, for a couple of datasets, more specifically named functions) that returns an xarray object indexed by meaningful coordinates such as `Subject`, `Antigen`, and `Receptor`:

```python
from tensordata.zohar import data

dx = data()
print(dx.dims)     # ('Sample', 'Antigen', 'Receptor')
print(dx.shape)

# Select a slice by coordinate label
dx.sel(Antigen="S1", Receptor="IgG3")
```

Convert any `DataArray` to a plain NumPy tensor (plus its axis labels) with the `xr_to_bunch` helper:

```python
from tensordata.utils import xr_to_bunch

bunch = xr_to_bunch(dx)
bunch.tensor   # numpy.ndarray
bunch.mode     # list of dimension names, e.g. ["Sample", "Antigen", "Receptor"]
bunch.axes     # list of coordinate arrays, one per dimension
```

## Available datasets

| Module | Source | Loader(s) | Description |
| --- | --- | --- | --- |
| `tensordata.alter` | Alter et al., *Molecular Systems Biology* (2018) | `data()` | Humoral correlates of HIV control: Fc Array, glycan, and effector-function measurements, returned as an `xarray.Dataset` with `Fc`, `gp120`, and `Functional` variables. |
| `tensordata.atyeo` | Atyeo et al. (2020) | `data()` | COVID-19 antibody profiling data (antigen × receptor). |
| `tensordata.chung` | Chung et al. (2021) | `data()` | Antibody profiling data (antigen × receptor) across subjects. |
| `tensordata.jones` | Jones et al. (2017) | `process_RA_Tensor()`, `make_RA_Tensor()` | Rheumatoid arthritis synovial fibroblast cytokine response data, processed into a stimulant × inhibitor × cytokine × donor tensor. |
| `tensordata.kaplonek` | Kaplonek et al., MGH & SpaceX cohorts (2021) | `MGH4D()`, `SpaceX4D()` | COVID-19 systems serology data (subject × antigen × receptor × time), including functional assay results for the MGH cohort. |
| `tensordata.kaplonekVaccine` | Kaplonek et al., *Science Translational Medicine* (2022) | `data()` | mRNA-1273 / BNT162b2 vaccine antibody Fc-effector function data. |
| `tensordata.kaplonekVaccineSA` | Kaplonek et al., *Nature Immunology* (2023) | `data()` | ChAdOx1 nCoV-19 (AZD1222) vaccine Fc-receptor binding data, returned as an `xarray.Dataset` with `Meta` (subject metadata) and `Luminex` (subject × antigen × receptor) variables. |
| `tensordata.serology` | Combines `kaplonek` (MGH, SpaceX) and `zohar` | `serology_rename()`, `importConcat()`, `concat4D()` | Utilities for aligning antigen names across cohorts and concatenating the MGH, SpaceX, and Zohar serology datasets into combined 3D/4D tensors. |
| `tensordata.zohar` | Zohar et al. (2020) | `data(subtract_baseline=False)` | COVID-19 antibody profiling data (sample × antigen × receptor), with an option to subtract PBS baseline values. |

Each dataset's raw CSV files live in a corresponding directory under `tensordata/` (e.g. `tensordata/zohar2020/`), often alongside a `readme.txt`/`readme.md` describing the original source data and citation.

### Example: combining serology cohorts

```python
from tensordata.serology import concat4D

# Aligns antigen/receptor names across the MGH, SpaceX, and Zohar cohorts
# and combines them into a single tensor.
combined = concat4D()
```

## Development

This project uses [uv](https://docs.astral.sh/uv/) for dependency management.

```bash
git clone git@github.com:meyer-lab/tensordata.git
cd tensordata
uv sync
```

Run the test suite:

```bash
make test
```

or directly with pytest:

```bash
uv run pytest -s -v -x
```

Generate a coverage report:

```bash
make coverage.xml
```

Lint and type-check:

```bash
uv run ruff check .
uv run ruff format --check .
uv run ty check
```

Continuous integration (GitHub Actions) runs the test suite and code quality checks on every push, and uploads coverage results to Codecov.

## License

Released under the [MIT License](LICENSE).
