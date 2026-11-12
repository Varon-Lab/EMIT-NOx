[![DOI](https://zenodo.org/badge/1452827134.svg)](https://doi.org/10.5281/zenodo.24154985)

# EMIT-NOx

Code for the EMIT NO2 retrieval and NOx point-source rate estimates in
*Quantifying Nitrogen Oxide Point Sources With the EMIT Satellite Imaging Spectrometer* (GRL, doi:10.1029/2026GL124107).
Per-scene retrievals, masks and source rates are archived on Zenodo at doi:10.5281/zenodo.24150654

## Setup

```bash
git clone --recurse-submodules <this repo>
conda env create -f environment.yml && conda activate emit_nox_env
export EMIT_NOX_DIR=/path/to/storage   # base directory for all data and results (default: ./data)
```

External input: a CAMPD API key file (`$CAMPD_APIKEY_FILE`), only needed to download CEMS emissions. The AMF look-up table is included at `datasets/differential_amf.nc`.

## Workflow (run from `src/`)

```bash
python EMIT_QUERY.py --loc_name <SITE> --L1B --L2A --download                       # download EMIT granules
python BATCH_RETR.py --loc_name <SITE> --km_boundary 40 --destripe                  # NO2 retrieval
# draw plume masks manually -> <mask_base_dir>/<SITE>/<SITE>_<GRANULE>_retrieval_mask.npy
python source_rate.py batch --input-csv scenes.csv --output-csv out.csv             # source rates (add --riyadh for Riyadh)
python export_scene_netcdf.py --input-csv scenes.csv --output-dir nc_out            # per-scene NetCDFs
```

`scenes.csv` needs `LOC_NAME`, `GRANULE`, `CAMPD_CORR` (CEMS rate, kg/s). Site coordinates are in `REFERENCE_PLANTS.py` and `config.py`.
`error_analysis.py` runs the source-rate uncertainty ensemble.
