#!/usr/bin/env python3
"""
Cross-sectional-flux (CSF) NOx source rate Q for EMIT NO2 plume scenes: one module for U.S. CEMS plants (HRRR
wind) and Riyadh-style sites (GEOS-FP wind), with a single-scene and a batch command line.

This file replaces four scripts; the computation is unchanged and was copied verbatim:

    run_source_rate.py               -> the core below (Params, load_scene, compute_q_from_scene, ...) + `single`
    run_source_rate_riyadh.py        -> geosfp_wind_resolver() + site routing + `single --loc-name RIYADH_PLANT_9`
    batch_run_source_rate.py         -> `batch --input-csv ...`          (CSV columns: the "us" schema)
    batch_run_source_rate_riyadh.py  -> `batch --riyadh ...`             (CSV columns: the "riyadh" schema)

Site handling is chosen per scene, as in the old scripts: a LOC_NAME in REFERENCE_PLANTS uses HRRR wind at 500 m
AGL and that registry; a LOC_NAME found only in config.LOCS (e.g. RIYADH_PLANT_9) uses GEOS-FP wind and LOCS.

Pipeline summary (all defaults reproduce the published numbers):
  saved boolean plume mask -> optional extra dilation/erosion -> rotate so the plume axis is horizontal ->
  integrate cross-sections / AMF -> x NOx:NO2 (1.38) x wind -> Q(x) -> Savitzky-Golay smoothing (51 px) ->
  mean of Q(x) above a length-dependent percentile (90th for short plumes ... 50th for plumes >= 25 km).
Every knob is perturbed away from its own default by EITHER an additive offset OR a multiplicative fraction
(never set to an absolute value) - see resolve_param(). The knobs are meant for sensitivity analysis.

Usage:
    # one scene (was: python run_source_rate.py --loc-name Colstrip --granule G)
    python source_rate.py single --loc-name Colstrip --granule 20250213T190437_2504413_005

    # one Riyadh scene (was: python run_source_rate_riyadh.py --granule G [--lat .. --lon .. --mask-path ..])
    python source_rate.py single --loc-name RIYADH_PLANT_9 --granule 20250730T075549_2521105_014

    # batch over a CSV with LOC_NAME, GRANULE, CAMPD_CORR   (was: python batch_run_source_rate.py ...)
    python source_rate.py batch --input-csv plotdf_data.csv --output-csv plotdf_source_rates.csv

    # batch over every Riyadh granule that has results      (was: python batch_run_source_rate_riyadh.py ...)
    python source_rate.py batch --riyadh --output-csv riyadh_source_rates.csv
    python source_rate.py batch --riyadh --granules 20250730T075549_2521105_014 --output-csv out.csv

    # the same perturbation knobs apply to both commands, e.g.
    python source_rate.py single --loc-name Colstrip --granule ... \
        --percentile-perturb-frac -0.10 --amf-plume-height-offset-m -100

Batch output schemas (kept identical to the old scripts so downstream notebooks keep working):
    us      LOC_NAME, GRANULE, CAMPD, PERCENTILE_THRESHOLD, WIND_SPEED_M_S, AMF, PIX_PITCH_M, Q
    riyadh  LOC_NAME, GRANULE, Q, PERCENTILE_THRESHOLD, WIND_SPEED_M_S, AMF_EFF, PLUME_LENGTH_KM
The schema defaults to "riyadh" for `batch --riyadh` without --input-csv and to "us" otherwise; override with --schema.
"""
import argparse
import glob
import os
import re
import sys
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from functools import lru_cache
from typing import Callable, Optional, Tuple

import numpy as np
import pandas as pd
import xarray as xr
from scipy.ndimage import (
    binary_dilation,
    binary_erosion,
    binary_fill_holes,
    rotate,
    shift,
)
from scipy.integrate import quad
from scipy.signal import savgol_filter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(HERE, "../EMIT-Data-Resources/python/modules/"))
from emit_tools import emit_xarray, ortho_xr  # noqa: E402

sys.path.append(os.path.join(HERE, ".."))
from config import CONFIG, LOCS  # noqa: E402
from REFERENCE_PLANTS import REFERENCE_PLANTS  # noqa: E402

sys.path.append(os.path.join(HERE, "../datasets/"))
import get_geosfp  # noqa: E402
from get_hrrr import get_hrrr_wind_agl  # noqa: E402

get_geosfp_wind = get_geosfp.get_geosfp_wind

PREFIXES = {
    "OBS": "EMIT_L1B_OBS_001_",
    "L1B": "EMIT_L1B_RAD_001_",
    "MASK": "EMIT_L2A_MASK_001_",
    "L2A": "EMIT_L2A_RFL_001_",
}

AVOG = 6.02214076e23
MOLEC_TO_KG_NO2 = 46 / (AVOG * 1000)
NOX_NO2_RATIO = 1.38

DEFAULT_WIND_LAYER_M = (499.0, 501.0)  # point estimate at 500 m AGL, approximated as a 2 m window
WIND_LAYER_HALF_WIDTH_M = 1.0  # a custom height h is approximated as a point value via h +/- 1 m
WIND_SPEED_FLOOR_MS = 0.3  # resolved wind speed is never allowed below this, even after offset/frac perturbation

MASK_BASE_DIR = CONFIG["mask_base_dir"]

# ---- "current operation" defaults, taken directly from the notebook ----
DEFAULT_MASK_DILATE_ITERS = 0
DEFAULT_MASK_ERODE_ITERS = 0
DEFAULT_AMF_PLUME_HEIGHT_M = 500.0
DEFAULT_SMOOTHING_WINDOW = 51
DEFAULT_SMOOTHING_POLYORDER = 3
NATIVE_PIX_PITCH_M = 60.0  # EMIT orthorectified grid spacing; used for the km->pixel smoothing conversion


def default_percentile_for_length(plume_length_km: float) -> float:
    """Length-dependent percentile threshold, matching retained_pct() in
    PAPER_FIGS.ipynb (the actual estimator behind the published Fig 4c):
    90th pct for short plumes (<= 5 km), 50th pct for long plumes
    (>= 25 km), linear in between, rounded to the nearest 10."""
    p = np.clip(100.0 - 2.0 * plume_length_km, 50.0, 90.0)
    return float(np.floor(p / 10.0 + 0.5) * 10.0)


def resolve_param(default, offset=None, frac=None):
    """Perturb `default` by EITHER an additive offset OR a multiplicative
    fraction - never a full override, and never both at once:
        offset=+5   -> default + 5
        frac=0.10   -> default * 1.10   (+10%)
        frac=-0.10  -> default * 0.90   (-10%)
    Returns `default` unchanged if neither is given."""
    if offset is not None and frac is not None:
        raise ValueError("pass either an offset or a frac perturbation, not both")
    if frac is not None:
        return default * (1.0 + frac)
    if offset is not None:
        return default + offset
    return default


@dataclass
class Params:
    mask_dilate_iters: int = DEFAULT_MASK_DILATE_ITERS
    mask_erode_iters: int = DEFAULT_MASK_ERODE_ITERS

    amf_plume_height_offset_m: Optional[float] = None  # additive, e.g. -100 -> 400 m
    amf_plume_height_perturb_frac: Optional[float] = None  # multiplicative, relative to 500 m default

    wind_height_m: Optional[float] = None  # None -> DEFAULT_WIND_LAYER_M (point estimate at 500 m AGL)
    wind_speed_offset_m_s: Optional[float] = None  # additive, applied to the resolved wind speed
    wind_speed_perturb_frac: Optional[float] = None  # multiplicative, applied to the resolved wind speed
    wind_speed_resolver: Optional[Callable[[float, float, datetime, "Params"], float]] = None  # if set, called instead of the built-in HRRR resolver to get the raw (pre-offset/frac) wind speed - e.g. a GEOS-FP-based resolver for a different site. Signature: (lat, lon, obs_time, params) -> m/s

    smoothing_window: int = DEFAULT_SMOOTHING_WINDOW  # pixels; ignored if smoothing_window_km is set
    smoothing_polyorder: int = DEFAULT_SMOOTHING_POLYORDER
    smoothing_window_km: Optional[float] = None  # physical smoothing length; converted to pixels via NATIVE_PIX_PITCH_M (60 m)

    percentile_offset: Optional[float] = None  # additive percentage points, relative to the length-based default
    percentile_perturb_frac: Optional[float] = None  # multiplicative, relative to that length-based default

    nox_no2_ratio: Optional[float] = None  # None -> module NOX_NO2_RATIO default

    mask_path_override: Optional[str] = None
    mask_shift_px: Tuple[int, int] = (0, 0)  # (row, col) shift applied to the loaded mask before use; for sampling nearby background regions

    def resolved_amf_plume_height(self) -> float:
        return resolve_param(DEFAULT_AMF_PLUME_HEIGHT_M, self.amf_plume_height_offset_m, self.amf_plume_height_perturb_frac)

    def resolved_percentile(self, plume_length_km: float) -> float:
        default = default_percentile_for_length(plume_length_km)
        pct = resolve_param(default, self.percentile_offset, self.percentile_perturb_frac)
        return float(np.clip(pct, 1.0, 99.0))


@lru_cache(maxsize=None)
def _cached_hrrr_speed(lat: float, lon: float, obs_time: datetime, layer: tuple, cache_dir: str) -> float:
    return float(get_hrrr_wind_agl(lat, lon, obs_time, layer=layer, cache=cache_dir)["speed_ms"])


def _resolve_wind_speed(lat: float, lon: float, obs_time: datetime, params: Params) -> float:
    if params.wind_height_m is None:
        layer = DEFAULT_WIND_LAYER_M
    else:
        layer = (params.wind_height_m - WIND_LAYER_HALF_WIDTH_M, params.wind_height_m + WIND_LAYER_HALF_WIDTH_M)

    wind_speed = _cached_hrrr_speed(lat, lon, obs_time, layer, f"{CONFIG['hrrr']}")

    resolved = resolve_param(wind_speed, params.wind_speed_offset_m_s, params.wind_speed_perturb_frac)
    return max(resolved, WIND_SPEED_FLOOR_MS)


def _load_amf_fields(loc_name: str, granule_name: str):
    """Per-pixel fields needed for AMF averaging (reads L2A reflectance + OBS
    geometry netCDFs). Independent of any Params - plume mask, AMF plume
    height, etc all only affect how these fields get averaged, not the fields
    themselves - so this is meant to be called once per scene and reused."""
    l2a_fpath = f"{CONFIG['data_folder']}/{loc_name}/{PREFIXES['L2A']}{granule_name}.nc"
    L2A_xarr = emit_xarray(l2a_fpath, ortho=True)
    emit_spec_wlen = L2A_xarr["wavelengths"].to_numpy().astype(np.float64)
    window_sel = (emit_spec_wlen >= 405) & (emit_spec_wlen <= 465)
    gler = L2A_xarr["reflectance"].sel(wavelengths=window_sel, method="nearest").values
    gler = np.where(gler > -8000, gler, np.nan)
    gler_field = np.nanmean(gler, axis=-1)
    del L2A_xarr

    obs_fpath = f"{CONFIG['data_folder']}/{loc_name}/{PREFIXES['OBS']}{granule_name}.nc"
    OBS_xarr = emit_xarray(obs_fpath, ortho=True)
    obs_src = np.where(OBS_xarr["obs"] > -8000, OBS_xarr["obs"], np.nan)
    sza_field = obs_src[:, :, 4]
    vza_field = obs_src[:, :, 2]
    to_sensor_az = np.array(obs_src[:, :, 1])
    to_sun_az = np.array(obs_src[:, :, 3])
    rel_azimuth = np.abs(to_sensor_az - to_sun_az)
    phi_field = np.minimum(rel_azimuth, 360.0 - rel_azimuth)
    del OBS_xarr

    return gler_field, sza_field, vza_field, phi_field


_AMF_LUT_CACHE = None


def _get_amf_lut():
    global _AMF_LUT_CACHE
    if _AMF_LUT_CACHE is None:
        _AMF_LUT_CACHE = xr.load_dataset(CONFIG["AMF_LUT"])
    return _AMF_LUT_CACHE


def _compute_amf(gler_field, sza_field, vza_field, phi_field, plume_mask_filled, geosfp_info, params: Params) -> float:
    gler = np.nanmean(np.where(plume_mask_filled, gler_field, np.nan))
    sza = np.nanmean(np.where(plume_mask_filled, sza_field, np.nan))
    vza = np.nanmean(np.where(plume_mask_filled, vza_field, np.nan))
    phi = np.nanmean(np.where(plume_mask_filled, phi_field, np.nan))

    plume_height_m = params.resolved_amf_plume_height()

    H = 287.05 * geosfp_info["T2M"] / 9.81  # scale height, m
    surface_pressure = np.clip(geosfp_info["PS"], 112, 1012)
    pressure_at_height = np.clip(surface_pressure * np.exp(-plume_height_m / H), 1, 1019)

    amf_lut = _get_amf_lut()
    amf_profile = amf_lut["damf"].interp(
        dspres=surface_pressure,
        dpres=pressure_at_height,
        dalb=gler,
        dphi=phi,
        dvza=vza,
        dsza=sza,
    ).data
    amf_eff = amf_profile + 0  # forces proper scalar cast, per notebook
    return float(amf_eff)


_METADATA_CACHE = None


def _get_metadata() -> pd.DataFrame:
    global _METADATA_CACHE
    if _METADATA_CACHE is None:
        _METADATA_CACHE = pd.read_csv(f"{CONFIG['data_folder']}/metadata_FULL.csv")
    return _METADATA_CACHE


@dataclass
class SceneData:
    """Everything expensive and parameter-independent for one scene: reading
    and orthorectifying the radiance/retrieval/L2A/OBS netCDFs, GEOS-FP and
    metadata lookups. Build once with load_scene() and reuse across many
    compute_q_from_scene() calls with different Params (e.g. an ensemble) -
    each of those only redoes the cheap, perturbation-dependent math."""
    loc_name: str
    granule_name: str
    loc_data: dict
    obs_time: datetime
    campd_e_rate: float
    geosfp_info: dict
    dSCD_nan: np.ndarray
    pix_pitch_m: float
    lon_grid: np.ndarray
    lat_grid: np.ndarray
    Rr: int
    Rc: int
    gler_field: np.ndarray
    sza_field: np.ndarray
    vza_field: np.ndarray
    phi_field: np.ndarray


def load_scene(loc_name: str, granule_name: str, loc_registry: dict = None, loc_data_override: dict = None) -> SceneData:
    """loc_registry: where to look up {'LAT', 'LON'} for loc_name. Defaults to
    REFERENCE_PLANTS (US CEMS plants); pass a different dict (e.g. LOCS) for
    sites not in that registry. loc_data_override: if given, use this
    {'LAT', 'LON'} directly and skip the registry lookup entirely - e.g. to
    target a different source within the same scene/granule (radiance and
    retrieval file paths are still built from loc_name; only the source
    coordinates - the rotation pivot and the point meteorology is sampled at
    - come from here). Pair with Params.mask_path_override to point at that
    other source's own mask file."""
    loc_data = loc_data_override if loc_data_override is not None else (loc_registry if loc_registry is not None else REFERENCE_PLANTS)[loc_name]

    retrieval_fpath = (
        f"{CONFIG['results_folder']}/{CONFIG['retr_subdir']}/{loc_name}/"
        f"dSCD_{PREFIXES['L1B']}{granule_name}.npy"
    )
    radiance_fpath = f"{CONFIG['data_folder']}/{loc_name}/{PREFIXES['L1B']}{granule_name}.nc"

    obs_time = datetime.strptime(granule_name.split("_")[0], "%Y%m%dT%H%M%S").replace(tzinfo=timezone.utc)

    # CAMPD/CEMS ground truth is US-only; sites without a metadata_FULL.csv row (e.g. Riyadh) get NaN, not a crash
    try:
        meta = _get_metadata()
        matches = meta[meta["GRANULE"] == granule_name]
        campd_e_rate = float(matches.iloc[0]["CAMPD_RATE"]) if len(matches) else float("nan")
    except Exception:
        campd_e_rate = float("nan")

    geosfp_info = get_geosfp_wind(loc_data["LAT"], loc_data["LON"], obs_time, cache=f"{CONFIG['geosfp']}/")

    # ---- dSCD field, orthorectified (cell 20) ----
    ds_no_orth = emit_xarray(radiance_fpath, ortho=False)
    dSCD = np.load(retrieval_fpath)
    wl_val = float(ds_no_orth["wavelengths"].isel(wavelengths=0))
    dscd_da = xr.DataArray(
        dSCD.astype("float32")[..., None],
        dims=("downtrack", "crosstrack", "wavelengths"),
        coords={
            "downtrack": ds_no_orth["downtrack"],
            "crosstrack": ds_no_orth["crosstrack"],
            "wavelengths": [wl_val],
        },
        name="dSCD",
    )
    ds_no_orth = ds_no_orth.assign(dSCD=dscd_da)
    ds = ortho_xr(ds_no_orth)
    dy, _dx = ds.rio.resolution()
    pix_pitch_m = 111000 * abs(dy)
    del ds_no_orth

    rad1500 = ds["radiance"].sel(wavelengths=1500, method="nearest").values
    dSCD_nan = np.where(rad1500 <= -8000, np.nan, ds["dSCD"].values[:, :, 0]) * 1e19

    lon_vals, lat_vals = ds["longitude"].values, ds["latitude"].values
    lon_grid = np.linspace(lon_vals.min(), lon_vals.max(), dSCD_nan.shape[1])
    lat_grid = np.linspace(lat_vals.min(), lat_vals.max(), dSCD_nan.shape[0])
    del ds

    ni = lambda v, a: int(np.argmin(np.abs(a - v)))  # noqa: E731
    Rr = len(lat_grid) - ni(loc_data["LAT"], lat_grid)
    Rc = ni(loc_data["LON"], lon_grid)

    gler_field, sza_field, vza_field, phi_field = _load_amf_fields(loc_name, granule_name)

    return SceneData(
        loc_name=loc_name, granule_name=granule_name, loc_data=loc_data, obs_time=obs_time,
        campd_e_rate=campd_e_rate, geosfp_info=geosfp_info, dSCD_nan=dSCD_nan, pix_pitch_m=pix_pitch_m,
        lon_grid=lon_grid, lat_grid=lat_grid, Rr=Rr, Rc=Rc,
        gler_field=gler_field, sza_field=sza_field, vza_field=vza_field, phi_field=phi_field,
    )


def compute_q_from_scene(scene: SceneData, params: Params, save_dir: Optional[str] = None) -> dict:
    mask_path = params.mask_path_override or (
        f"{MASK_BASE_DIR}/{scene.loc_name}/{scene.loc_name}_{scene.granule_name}_retrieval_mask.npy"
    )
    base_mask = np.load(mask_path).astype(bool)
    if params.mask_shift_px != (0, 0):
        shifted = shift(
            base_mask.astype(float), params.mask_shift_px,
            order=0, mode="constant", cval=0.0, prefilter=False,
        )
        base_mask = shifted > 0.5

    if params.wind_speed_resolver is not None:
        raw_wind_speed = params.wind_speed_resolver(scene.loc_data["LAT"], scene.loc_data["LON"], scene.obs_time, params)
        wind_speed = max(resolve_param(raw_wind_speed, params.wind_speed_offset_m_s, params.wind_speed_perturb_frac), WIND_SPEED_FLOOR_MS)
    else:
        wind_speed = _resolve_wind_speed(scene.loc_data["LAT"], scene.loc_data["LON"], scene.obs_time, params)

    dSCD_nan = scene.dSCD_nan
    pix_pitch_m = scene.pix_pitch_m

    # ---- mask perturbation: extra dilation/erosion on top of the saved mask (cell 22) ----
    plume_mask_filled = binary_fill_holes(base_mask)
    if params.mask_dilate_iters:
        plume_mask_filled = binary_dilation(plume_mask_filled, iterations=params.mask_dilate_iters)
        plume_mask_filled = binary_fill_holes(plume_mask_filled)
    if params.mask_erode_iters:
        plume_mask_filled = binary_erosion(plume_mask_filled, iterations=params.mask_erode_iters)
        plume_mask_filled = binary_fill_holes(plume_mask_filled)

    plume_size_init = int(plume_mask_filled.sum())
    if plume_size_init == 0:
        raise ValueError("mask perturbation eroded the plume mask to nothing")

    dSCD_plume_masked = np.where(plume_mask_filled, dSCD_nan, np.nan)

    rows, cols = np.indices(dSCD_plume_masked.shape)
    m = np.isfinite(dSCD_plume_masked)
    weights = np.clip(dSCD_plume_masked[m], 0, None)
    xw = np.nansum(cols[m] * weights) / np.nansum(weights)
    yw = np.nansum(rows[m] * weights) / np.nansum(weights)

    Rr, Rc = scene.Rr, scene.Rc
    Gr, Gc = int(yw), int(xw)

    ym, xm = np.where(np.isfinite(dSCD_plume_masked))
    pad = 50
    r0, r1 = max(0, ym.min() - pad), min(dSCD_plume_masked.shape[0], ym.max() + pad + 1)
    c0, c1 = max(0, xm.min() - pad), min(dSCD_plume_masked.shape[1], xm.max() + pad + 1)
    img = dSCD_plume_masked[r0:r1, c0:c1].copy()
    Rr, Rc, Gr, Gc = Rr - r0, Rc - c0, Gr - r0, Gc - c0

    # ---- rotate about the source so the plume axis is horizontal (cell 23) ----
    dxg, dyg = (Gc - Rc), (Gr - Rr)
    angle = -np.degrees(np.arctan2(-dyg, dxg))

    H, W = img.shape
    P = max(H, W)
    img = np.pad(img, P, constant_values=np.nan)
    Rr, Rc, Gr, Gc = Rr + P, Rc + P, Gr + P, Gc + P
    H, W = img.shape
    cy, cx = H / 2, W / 2

    valid = np.isfinite(img).astype(float)
    vals = np.nan_to_num(img, nan=0.0)
    shift_vec = (cy - Rr, cx - Rc)
    vals = shift(vals, shift_vec, order=1, mode="constant", cval=0.0, prefilter=False)
    valid = shift(valid, shift_vec, order=1, mode="constant", cval=0.0, prefilter=False)

    num = rotate(vals * valid, angle, reshape=True, order=1, mode="constant", cval=0.0, prefilter=False)
    den = rotate(valid, angle, reshape=True, order=1, mode="constant", cval=0.0, prefilter=False)
    with np.errstate(divide="ignore", invalid="ignore"):
        img_rot = num / den
    img_rot[den < 1e-6] = np.nan

    plume_size_rot = int(np.sum(~np.isnan(img_rot)))
    Rx = img_rot.shape[1] / 2
    start_idx = int(Rx)

    # ---- fill within outer per-column boundary (cell 25) ----
    Hh, Ww = img_rot.shape
    A_filled = img_rot.copy()
    for j in range(Ww):
        col = img_rot[:, j]
        validcol = np.isfinite(col)
        if not np.any(validcol):
            continue
        i0 = np.argmax(validcol)
        i1 = Hh - 1 - np.argmax(validcol[::-1])
        col_mean = np.nanmean(col[i0 : i1 + 1])
        fill_idx = np.isnan(col) & (np.arange(Hh) >= i0) & (np.arange(Hh) <= i1)
        A_filled[fill_idx, j] = col_mean
    img_rot = A_filled

    # ---- AMF (cell 26) ----
    amf_eff = _compute_amf(scene.gler_field, scene.sza_field, scene.vza_field, scene.phi_field,
                            plume_mask_filled, scene.geosfp_info, params)

    # ---- integrate to get the along-plume emission-rate profile Q (cell 27) ----
    pix_pitch = pix_pitch_m * np.sqrt(plume_size_rot / plume_size_init)
    integrated_flux = np.nansum(img_rot, axis=0) * pix_pitch / amf_eff
    nox_no2_ratio = params.nox_no2_ratio if params.nox_no2_ratio is not None else NOX_NO2_RATIO
    NO2_kg_m2 = integrated_flux * MOLEC_TO_KG_NO2 * nox_no2_ratio * 1e4  # cm^2 -> m^2

    positive_idx = np.where(NO2_kg_m2 > 0)[0]
    if positive_idx.size == 0:
        raise ValueError("no positive emission signal found downwind of the source")
    end_idx = positive_idx[-1]

    x = np.arange(len(NO2_kg_m2))
    plume_extent = x[start_idx:end_idx] * pix_pitch - start_idx * pix_pitch
    Q_profile = NO2_kg_m2[start_idx:end_idx] * wind_speed

    if Q_profile.size < 5:
        raise ValueError("plume profile too short to smooth/summarize")

    plume_length_km = float(np.max(plume_extent)) / 1000.0

    if params.smoothing_window_km is not None:
        window = int(round(params.smoothing_window_km * 1000.0 / NATIVE_PIX_PITCH_M))
    else:
        window = params.smoothing_window
    window = max(window, 1)
    window = min(window, Q_profile.size if Q_profile.size % 2 == 1 else Q_profile.size - 1)
    if window % 2 == 0:
        window += 1
    window = max(window, params.smoothing_polyorder + 1 + (params.smoothing_polyorder % 2 == 0))
    if window % 2 == 0:
        window += 1

    smoothed = savgol_filter(Q_profile, window_length=window, polyorder=params.smoothing_polyorder)

    pct = params.resolved_percentile(plume_length_km)
    Q = float(np.mean(smoothed[smoothed >= np.percentile(smoothed, pct)]))

    result = {
        "loc_name": scene.loc_name,
        "granule_name": scene.granule_name,
        "Q_kg_s": Q,
        "campd_e_rate_kg_s": scene.campd_e_rate,
        "amf_eff": amf_eff,
        "wind_speed_m_s": wind_speed,
        "plume_length_km": plume_length_km,
        "percentile_used": pct,
        "mask_dilate_iters": params.mask_dilate_iters,
        "mask_erode_iters": params.mask_erode_iters,
        "amf_plume_height_m": params.resolved_amf_plume_height(),
        "smoothing_window": window,
        "smoothing_polyorder": params.smoothing_polyorder,
        "nox_no2_ratio": nox_no2_ratio,
        "plume_pixels_init": plume_size_init,
        "plume_pixels_rotated": plume_size_rot,
        "pix_pitch_m": float(pix_pitch),
    }

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        np.save(os.path.join(save_dir, f"{scene.loc_name}_{scene.granule_name}_profile.npy"), Q_profile)
        np.save(os.path.join(save_dir, f"{scene.loc_name}_{scene.granule_name}_smoothed.npy"), smoothed)
        np.save(os.path.join(save_dir, f"{scene.loc_name}_{scene.granule_name}_extent.npy"), plume_extent)

    return result


def compute_source_rate(loc_name: str, granule_name: str, params: Params, save_dir: Optional[str] = None, loc_registry: dict = None, loc_data_override: dict = None) -> dict:
    """Convenience wrapper: load the scene fresh and compute Q once. For
    repeated calls against the SAME scene with different Params (e.g. an
    ensemble/sensitivity sweep), call load_scene() once and reuse it across
    many compute_q_from_scene() calls instead - that skips re-reading and
    re-orthorectifying the scene's netCDFs on every call."""
    scene = load_scene(loc_name, granule_name, loc_registry=loc_registry, loc_data_override=loc_data_override)
    return compute_q_from_scene(scene, params, save_dir=save_dir)



# =====================================================================================================
# Riyadh / GEOS-FP sites (formerly run_source_rate_riyadh.py and batch_run_source_rate_riyadh.py)
#
# Riyadh Power Plant 9 uses the identical pipeline above except for the location lookup (LOCS instead of
# REFERENCE_PLANTS), the wind (GEOS-FP power-law profile instead of HRRR) and the ground truth (there is no
# CAMPD/CEMS for Riyadh, so campd_e_rate_kg_s is NaN). The nominal wind is a point estimate at 500 m AGL,
# approximated as the integral average over a narrow (499, 501) m window, matching the U.S. convention; a custom
# wind_height_m h instead evaluates the power-law profile as an exact point value at h.
# =====================================================================================================
RIYADH_LOC_NAME = "RIYADH_PLANT_9"

RIYADH_DEFAULT_WIND_LAYER_M = (499.0, 501.0)  # point estimate at 500 m AGL, approximated as a 2 m window


@lru_cache(maxsize=None)
def _cached_geosfp_wind_info(lat: float, lon: float, obs_time: datetime, cache_dir: str):
    return get_geosfp_wind(lat, lon, obs_time, cache=cache_dir)


def geosfp_wind_resolver(lat: float, lon: float, obs_time: datetime, params: Params) -> float:
    """Params.wind_speed_resolver implementation for GEOS-FP. Nominal (h=None)
    is the continuous integral average of the power-law wind profile over
    RIYADH_DEFAULT_WIND_LAYER_M - a narrow (499,501) m window, i.e. a point
    estimate at 500 m; an explicit height is an exact point value evaluated
    directly, no window at all."""
    wind_info = _cached_geosfp_wind_info(lat, lon, obs_time, f"{CONFIG['geosfp']}/")
    power_law = wind_info["power_law"]

    if params.wind_height_m is None:
        lo, hi = RIYADH_DEFAULT_WIND_LAYER_M
        return float(quad(power_law, lo, hi)[0] / (hi - lo))
    return float(power_law(params.wind_height_m))


GRANULE_ID_RE = re.compile(r"\d{8}T\d{6}_\d{7}_\d{3}")  # e.g. 20250730T075549_2521105_014


def discover_granules(loc_name: str = RIYADH_LOC_NAME, root: Optional[str] = None) -> list:
    """Granule ids found under <root>/<loc_name>/ (default root: CONFIG['results_folder']/CONFIG['ps_subdir']).
    That directory holds several files per granule (mask, profile, smoothed, extent, ...); the id is taken from each
    file name by regex and de-duplicated. `root` is an addition: the default behaves exactly like the old script."""
    pattern = f"{root}/{loc_name}/*" if root else f"{CONFIG['results_folder']}/{CONFIG['ps_subdir']}/{loc_name}/*"
    seen = set()
    granules = []
    for path in sorted(glob.glob(pattern)):
        basename = path.split("/")[-1]
        match = GRANULE_ID_RE.search(basename)
        if not match:
            continue
        granule = match.group(0)
        if granule not in seen:  # ps_subdir has many files per granule (mask, profile, smoothed, extent, ...)
            seen.add(granule)
            granules.append(granule)
    return granules


def uses_geosfp(loc_name: str) -> bool:
    """Riyadh-style sites: in LOCS but not a U.S. CEMS plant in REFERENCE_PLANTS (those use HRRR)."""
    return loc_name not in REFERENCE_PLANTS and loc_name in LOCS


def compute_for_site(loc_name: str, granule_name: str, params: Params, save_dir: Optional[str] = None, loc_data_override: dict = None) -> dict:
    """compute_source_rate() with the site-appropriate registry and wind source selected from loc_name - what the
    four old scripts each hard-wired. An explicit params.wind_speed_resolver is respected."""
    if uses_geosfp(loc_name):
        if params.wind_speed_resolver is None:
            params = replace(params, wind_speed_resolver=geosfp_wind_resolver)
        return compute_source_rate(loc_name, granule_name, params, save_dir=save_dir, loc_registry=LOCS, loc_data_override=loc_data_override)
    return compute_source_rate(loc_name, granule_name, params, save_dir=save_dir, loc_data_override=loc_data_override)



# =====================================================================================================
# Command line
# =====================================================================================================
OUTPUT_COLUMNS_US = ("LOC_NAME", "GRANULE", "CAMPD", "PERCENTILE_THRESHOLD", "WIND_SPEED_M_S", "AMF", "PIX_PITCH_M", "Q")
OUTPUT_COLUMNS_RIYADH = ("LOC_NAME", "GRANULE", "Q", "PERCENTILE_THRESHOLD", "WIND_SPEED_M_S", "AMF_EFF", "PLUME_LENGTH_KM")
REQUIRED_INPUT_COLUMNS = ("LOC_NAME", "GRANULE", "CAMPD_CORR")


def add_param_args(p: argparse.ArgumentParser) -> None:
    """Perturbation knobs, shared by `single` and `batch` (applied identically to every scene in a batch)."""
    p.add_argument("--mask-dilate-iters", type=int, default=DEFAULT_MASK_DILATE_ITERS, help="Extra binary_dilation iterations applied on top of the saved mask (default: 0)")
    p.add_argument("--mask-erode-iters", type=int, default=DEFAULT_MASK_ERODE_ITERS, help="Extra binary_erosion iterations applied on top of the saved mask (default: 0)")

    p.add_argument("--amf-plume-height-offset-m", type=float, default=None, help=f"Additive offset (m) on the AMF plume height, e.g. -100 -> 400 m (default: {DEFAULT_AMF_PLUME_HEIGHT_M} m, no offset)")
    p.add_argument("--amf-plume-height-perturb-frac", type=float, default=None, help="Multiplicative offset on plume height, e.g. 0.10 for +10%%, relative to the default. Mutually exclusive with --amf-plume-height-offset-m")

    p.add_argument("--wind-height-m", type=float, default=None, help=f"Wind speed as a point estimate at this AGL height h. HRRR (U.S. plants): approximated via the layer [h-{WIND_LAYER_HALF_WIDTH_M:.0f}, h+{WIND_LAYER_HALF_WIDTH_M:.0f}] m; GEOS-FP (Riyadh): the power-law profile evaluated exactly at h. Default: a point estimate at 500 m via {DEFAULT_WIND_LAYER_M} m")
    p.add_argument("--wind-speed-offset-m-s", type=float, default=None, help="Additive offset (m/s) on the resolved wind speed, e.g. -1.0. Mutually exclusive with --wind-speed-perturb-frac")
    p.add_argument("--wind-speed-perturb-frac", type=float, default=None, help="Multiplicative offset on the resolved wind speed, e.g. -0.10 for -10%%. Mutually exclusive with --wind-speed-offset-m-s")

    p.add_argument("--smoothing-window", type=int, default=DEFAULT_SMOOTHING_WINDOW, help=f"savgol_filter window_length in pixels (default: {DEFAULT_SMOOTHING_WINDOW}). Ignored if --smoothing-window-km is set")
    p.add_argument("--smoothing-window-km", type=float, default=None, help=f"savgol_filter window_length as a physical length (km), converted to pixels assuming a {NATIVE_PIX_PITCH_M:.0f} m EMIT pixel pitch. Takes precedence over --smoothing-window")
    p.add_argument("--smoothing-polyorder", type=int, default=DEFAULT_SMOOTHING_POLYORDER, help=f"savgol_filter polyorder (default: {DEFAULT_SMOOTHING_POLYORDER})")

    p.add_argument("--percentile-offset", type=float, default=None, help="Additive offset (percentage points) on the percentile threshold, e.g. -10, relative to the length-dependent default. Mutually exclusive with --percentile-perturb-frac")
    p.add_argument("--percentile-perturb-frac", type=float, default=None, help="Multiplicative offset on the percentile threshold, e.g. 0.10 for +10%%, -0.10 for -10%%, relative to the length-dependent default. Mutually exclusive with --percentile-offset")

    p.add_argument("--nox-no2-ratio", type=float, default=None, help=f"NOx:NO2 conversion ratio alpha (default: {NOX_NO2_RATIO})")

    p.add_argument("--save-dir", default=None, help="If set, save the raw/smoothed profile and extent arrays here")


def params_from_args(args: argparse.Namespace) -> Params:
    """Params from the shared knobs. The wind resolver is not set here: compute_for_site() adds the GEOS-FP resolver
    for Riyadh-style sites. `--mask-path` only exists on `single`."""
    return Params(
        mask_dilate_iters=args.mask_dilate_iters,
        mask_erode_iters=args.mask_erode_iters,
        amf_plume_height_offset_m=args.amf_plume_height_offset_m,
        amf_plume_height_perturb_frac=args.amf_plume_height_perturb_frac,
        wind_height_m=args.wind_height_m,
        wind_speed_offset_m_s=args.wind_speed_offset_m_s,
        wind_speed_perturb_frac=args.wind_speed_perturb_frac,
        smoothing_window=args.smoothing_window,
        smoothing_window_km=args.smoothing_window_km,
        smoothing_polyorder=args.smoothing_polyorder,
        percentile_offset=args.percentile_offset,
        percentile_perturb_frac=args.percentile_perturb_frac,
        nox_no2_ratio=args.nox_no2_ratio,
        mask_path_override=getattr(args, "mask_path", None),
    )


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("single", help="Source rate for one scene; prints every result field", description="Compute Q for one scene.")
    s.add_argument("--loc-name", required=True, help="Key into REFERENCE_PLANTS (U.S. plants, HRRR wind) or config.LOCS (e.g. RIYADH_PLANT_9, GEOS-FP wind)")
    s.add_argument("--granule", required=True, help="EMIT granule id, e.g. 20250213T190437_2504413_005")
    s.add_argument("--mask-path", default=None, help=f"Path to the saved boolean plume mask .npy (default: {MASK_BASE_DIR}/<LOC_NAME>/<LOC_NAME>_<GRANULE>_retrieval_mask.npy)")
    s.add_argument("--lat", type=float, default=None, help="Override source latitude for this run - use to target a different source within the same scene/granule (e.g. Riyadh PP10, Al Yamama) instead of the site's own coordinates. Must be given together with --lon; pair with --mask-path pointing at that source's own mask")
    s.add_argument("--lon", type=float, default=None, help="Override source longitude for this run. Must be given together with --lat")
    add_param_args(s)

    b = sub.add_parser("batch", help="Source rates for many scenes -> CSV", description="Compute Q for every scene in a CSV and/or every Riyadh granule, one shared parameterization.")
    b.add_argument("--input-csv", default=None, help="CSV with LOC_NAME, GRANULE (and CAMPD_CORR for the 'us' schema)")
    b.add_argument("--riyadh", action="store_true", help=f"Also run {RIYADH_LOC_NAME} scenes: --granules, or auto-discovered from <discover-dir>/{RIYADH_LOC_NAME}")
    b.add_argument("--granules", nargs="+", default=None, help="Explicit granule id list for --riyadh (default: auto-discover)")
    b.add_argument("--discover-dir", default=None, help="Root for --riyadh auto-discovery (default: CONFIG['results_folder']/CONFIG['ps_subdir'], as the old script)")
    b.add_argument("--output-csv", required=True, help="Where to write the results")
    b.add_argument("--schema", choices=("us", "riyadh"), default=None, help="Output columns (see module docstring). Default: riyadh for `--riyadh` without --input-csv, otherwise us")
    add_param_args(b)
    return p


def cmd_single(args: argparse.Namespace) -> None:
    if (args.lat is None) != (args.lon is None):
        raise SystemExit("--lat and --lon must be given together (or not at all)")
    loc_data_override = {"LAT": args.lat, "LON": args.lon} if args.lat is not None else None

    result = compute_for_site(args.loc_name, args.granule, params_from_args(args), save_dir=args.save_dir, loc_data_override=loc_data_override)

    for k, v in result.items():
        print(f"{k}: {v}")
    print(f"\nSource rate Q = {result['Q_kg_s']:.4f} kg/s")


def _collect_scenes(args: argparse.Namespace, schema: str) -> pd.DataFrame:
    frames = []
    if args.input_csv:
        df = pd.read_csv(args.input_csv)
        required = REQUIRED_INPUT_COLUMNS if schema == "us" else REQUIRED_INPUT_COLUMNS[:2]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"input CSV is missing required columns: {missing}")
        frames.append(df)
    if args.riyadh:
        granules = args.granules if args.granules else discover_granules(root=args.discover_dir)
        if not granules:
            root = args.discover_dir or f"{CONFIG['results_folder']}/{CONFIG['ps_subdir']}"
            raise ValueError(f"no granules found for {RIYADH_LOC_NAME} under {root}/{RIYADH_LOC_NAME}")
        frames.append(pd.DataFrame({"LOC_NAME": RIYADH_LOC_NAME, "GRANULE": granules}))
    if not frames:
        raise SystemExit("batch needs --input-csv and/or --riyadh")
    return pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]


def cmd_batch(args: argparse.Namespace) -> None:
    schema = args.schema or ("riyadh" if (args.riyadh and not args.input_csv) else "us")
    columns = OUTPUT_COLUMNS_US if schema == "us" else OUTPUT_COLUMNS_RIYADH
    params = params_from_args(args)
    scenes = _collect_scenes(args, schema)

    rows = []
    n = len(scenes)
    for i, (_, row) in enumerate(scenes.iterrows()):
        loc_name = row["LOC_NAME"]
        granule_name = row["GRANULE"]
        campd_corr = row["CAMPD_CORR"] if "CAMPD_CORR" in scenes.columns else float("nan")

        print(f"[{i + 1}/{n}] {loc_name} {granule_name} ... ", end="", file=sys.stderr, flush=True)
        try:
            result = compute_for_site(loc_name, granule_name, params, save_dir=args.save_dir)
            if schema == "us":
                rows.append({
                    "LOC_NAME": loc_name,
                    "GRANULE": granule_name,
                    "CAMPD": campd_corr,
                    "PERCENTILE_THRESHOLD": result["percentile_used"],
                    "WIND_SPEED_M_S": result["wind_speed_m_s"],
                    "AMF": result["amf_eff"],
                    "PIX_PITCH_M": result["pix_pitch_m"],
                    "Q": result["Q_kg_s"],
                })
            else:
                rows.append({
                    "LOC_NAME": loc_name,
                    "GRANULE": granule_name,
                    "Q": result["Q_kg_s"],
                    "PERCENTILE_THRESHOLD": result["percentile_used"],
                    "WIND_SPEED_M_S": result["wind_speed_m_s"],
                    "AMF_EFF": result["amf_eff"],
                    "PLUME_LENGTH_KM": result["plume_length_km"],
                })
            print(f"Q={result['Q_kg_s']:.4f} kg/s", file=sys.stderr)
        except Exception as exc:
            print(f"FAILED: {exc}", file=sys.stderr)
            nan = float("nan")
            if schema == "us":
                rows.append({"LOC_NAME": loc_name, "GRANULE": granule_name, "CAMPD": campd_corr, "PERCENTILE_THRESHOLD": nan,
                             "WIND_SPEED_M_S": nan, "AMF": nan, "PIX_PITCH_M": nan, "Q": nan})
            else:
                rows.append({"LOC_NAME": loc_name, "GRANULE": granule_name, "Q": nan, "PERCENTILE_THRESHOLD": nan,
                             "WIND_SPEED_M_S": nan, "AMF_EFF": nan, "PLUME_LENGTH_KM": nan})

        # write after every scene so a crash midway doesn't lose finished work
        pd.DataFrame(rows, columns=list(columns)).to_csv(args.output_csv, index=False)

    n_failed = sum(1 for r in rows if pd.isna(r["Q"]))
    print(f"\nWrote {len(rows)} rows to {args.output_csv} ({n_failed} failed)", file=sys.stderr)


def main():
    args = build_arg_parser().parse_args()
    {"single": cmd_single, "batch": cmd_batch}[args.command](args)


if __name__ == "__main__":
    main()
