#!/usr/bin/env python3
"""
Export one NetCDF4 file per EMIT scene containing everything the published
source-rate estimate was computed from: the NO2 retrieval field, the plume
mask, the lat/lon grid, and the wind / AMF / source-rate values that
run_source_rate.py used for that scene (plus the inputs behind them).

Takes the same inputs as batch_run_source_rate.py (US CEMS plants) and
batch_run_source_rate_riyadh.py (Riyadh), and the same perturbation knobs.
Nothing is re-implemented: Q, AMF, wind speed, percentile, plume length and the
Q(x) profile all come straight from run_source_rate.compute_q_from_scene().
The extra quantities (wind direction, AMF input means, effective mask) are
derived here from the same SceneData / Params, and the AMF is recomputed as a
consistency check against the value compute_q_from_scene() returned.

Site handling is chosen per row:
  - LOC_NAME in REFERENCE_PLANTS  -> HRRR wind at 500 m AGL (as batch_run_source_rate.py)
  - LOC_NAME only in LOCS (Riyadh) -> GEOS-FP wind (as batch_run_source_rate_riyadh.py)

Usage:
    # US plants (same CSV batch_run_source_rate.py takes; CAMPD_CORR optional)
    python export_scene_netcdf.py --input-csv plotdf_data.csv --output-dir nc_out

    # Riyadh Power Plant 9, auto-discovering every granule that has a saved mask
    # (<MASK_BASE_DIR>/RIYADH_PLANT_9/RIYADH_PLANT_9_<granule>_retrieval_mask.npy)
    python export_scene_netcdf.py --riyadh --output-dir nc_out

    # Riyadh, explicit granules
    python export_scene_netcdf.py --riyadh --granules 20250613T114019_2516407_025 --output-dir nc_out

Optional CSV columns (per scene):
    CAMPD_CORR  CEMS ground truth, kg NOx(as NO2)/s -> saved as `cems_rate`
    MASK_PATH   explicit mask .npy (default: <MASK_BASE_DIR>/<loc>/<loc>_<granule>_retrieval_mask.npy)
    LAT, LON    override source coordinates (e.g. another Riyadh source in the same granule)

Output: <output-dir>/<LOC_NAME>_<GRANULE>.nc per scene, plus <output-dir>/index.csv.
"""
import argparse
import dataclasses
import glob
import json
import os
import sys
import tempfile
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import xarray as xr
from scipy.ndimage import binary_dilation, binary_erosion, binary_fill_holes, shift

import source_rate as rsr
from config import CONFIG, LOCS
from REFERENCE_PLANTS import REFERENCE_PLANTS
from source_rate import RIYADH_LOC_NAME, _cached_geosfp_wind_info, geosfp_wind_resolver


def discover_riyadh_granules(loc_name: str = RIYADH_LOC_NAME) -> list:
    """Granules that have a default-named mask under MASK_BASE_DIR - exactly the files
    compute_q_from_scene() loads. (batch_run_source_rate_riyadh.discover_granules() looks under
    CONFIG['results_folder'], which is a different tree from where the masks are stored.)"""
    pattern = os.path.join(rsr.MASK_BASE_DIR, loc_name, f"{loc_name}_*_retrieval_mask.npy")
    prefix, suffix = f"{loc_name}_", "_retrieval_mask.npy"
    return sorted(os.path.basename(p)[len(prefix):-len(suffix)] for p in glob.glob(pattern))

CONVENTIONS = "CF-1.8"
INDEX_COLUMNS = (
    "LOC_NAME", "GRANULE", "STATUS", "FILE", "Q_KG_S", "CEMS_KG_S", "WIND_SPEED_M_S",
    "WIND_DIR_FROM_DEG", "AMF", "PERCENTILE", "PLUME_LENGTH_KM", "ERROR",
)


# ---------------------------------------------------------------------------
# derived quantities
# ---------------------------------------------------------------------------
def uses_geosfp(loc_name: str) -> bool:
    """Same split the two batch scripts make: Riyadh-style sites (in LOCS, not a US plant) use GEOS-FP."""
    return loc_name not in REFERENCE_PLANTS and loc_name in LOCS


def load_masks(scene: rsr.SceneData, params: rsr.Params):
    """Returns (saved_mask, used_mask). `used_mask` mirrors the morphology at the top of
    compute_q_from_scene(): optional shift, hole-fill, extra dilation/erosion."""
    mask_path = params.mask_path_override or (
        f"{rsr.MASK_BASE_DIR}/{scene.loc_name}/{scene.loc_name}_{scene.granule_name}_retrieval_mask.npy"
    )
    saved = np.load(mask_path).astype(bool)
    base = saved
    if params.mask_shift_px != (0, 0):
        base = shift(base.astype(float), params.mask_shift_px, order=0, mode="constant", cval=0.0, prefilter=False) > 0.5
    used = binary_fill_holes(base)
    if params.mask_dilate_iters:
        used = binary_fill_holes(binary_dilation(used, iterations=params.mask_dilate_iters))
    if params.mask_erode_iters:
        used = binary_fill_holes(binary_erosion(used, iterations=params.mask_erode_iters))
    return mask_path, saved, used


def wind_details(scene: rsr.SceneData, params: rsr.Params, geosfp: bool) -> dict:
    """Wind speed/direction behind the value compute_q_from_scene() used.
    Direction is the meteorological 'from' direction (deg clockwise from north) and is
    informational: the CSF method takes the plume axis from the mask, not from this."""
    lat, lon, t = scene.loc_data["LAT"], scene.loc_data["LON"], scene.obs_time
    unperturbed = dataclasses.replace(params, wind_speed_offset_m_s=None, wind_speed_perturb_frac=None)

    if geosfp:
        info = _cached_geosfp_wind_info(lat, lon, t, f"{CONFIG['geosfp']}/")
        raw = geosfp_wind_resolver(lat, lon, t, unperturbed)
        return {
            "source": "GEOS-FP",
            "speed_unperturbed": raw,
            # GEOS-FP only provides 2/10/50 m winds; 50 m is the closest level to the 500 m speed used.
            "dir_from": float(info["DIR50"]),
            "dir_level_m": 50.0,
            "height_m": params.wind_height_m if params.wind_height_m is not None else 500.0,
            "extra": {"geosfp_u10_m_s": info["U10"], "geosfp_u50_m_s": info["U50"],
                      "geosfp_dir10_deg": info["DIR10"], "geosfp_dir50_deg": info["DIR50"],
                      "geosfp_power_law_alpha": info["alpha"]},
        }

    if params.wind_height_m is None:
        layer = rsr.DEFAULT_WIND_LAYER_M
    else:
        layer = (params.wind_height_m - rsr.WIND_LAYER_HALF_WIDTH_M, params.wind_height_m + rsr.WIND_LAYER_HALF_WIDTH_M)
    hrrr = rsr.get_hrrr_wind_agl(lat, lon, t, layer=layer, cache=f"{CONFIG['hrrr']}")
    return {
        "source": "HRRR",
        "speed_unperturbed": float(hrrr["speed_ms"]),
        "dir_from": float(hrrr["dir_from_deg"]),
        "dir_level_m": float(np.mean(layer)),
        "height_m": float(np.mean(layer)),
        "extra": {"hrrr_u_m_s": hrrr["u_ms"], "hrrr_v_m_s": hrrr["v_ms"],
                  "hrrr_terrain_m_msl": hrrr["terrain_m_msl"]},
    }


def amf_inputs(scene: rsr.SceneData, used_mask: np.ndarray, params: rsr.Params) -> dict:
    """Plume-mask-averaged inputs to the AMF look-up (same averaging as _compute_amf)."""
    mean = lambda f: float(np.nanmean(np.where(used_mask, f, np.nan)))  # noqa: E731
    info = scene.geosfp_info
    return {
        "gler": mean(scene.gler_field), "sza": mean(scene.sza_field), "vza": mean(scene.vza_field),
        "raa": mean(scene.phi_field),
        "surface_pressure_hpa": float(np.clip(info["PS"], 112, 1012)),
        "t2m_k": float(info["T2M"]),
        "plume_height_m": params.resolved_amf_plume_height(),
    }


# ---------------------------------------------------------------------------
# dataset assembly
# ---------------------------------------------------------------------------
def _scalar(value, units, long_name, dtype="float64"):
    return xr.Variable((), np.asarray(value, dtype=dtype), {"units": units, "long_name": long_name})


def build_dataset(scene, params, result, profile, saved_mask, used_mask, mask_path, wind, amf_in, cems_rate, geosfp):
    amf = result["amf_eff"]
    dscd = scene.dSCD_nan  # molec cm-2, NaN outside the swath / invalid
    vcd = dscd / amf / rsr.AVOG * 1e4  # mol m-2 (scalar plume-mean AMF, as in the published method)
    lat = scene.lat_grid[::-1]  # array rows run north -> south; lat_grid is ascending
    lon = scene.lon_grid

    ds = xr.Dataset(coords={
        "latitude": ("latitude", lat, {"units": "degrees_north", "standard_name": "latitude"}),
        "longitude": ("longitude", lon, {"units": "degrees_east", "standard_name": "longitude"}),
    })
    grid = ("latitude", "longitude")
    ds["no2_slant_column_enhancement"] = (grid, dscd.astype("float32"), {
        "units": "molecule cm-2", "long_name": "Retrieved NO2 slant column enhancement (EMIT DOAS), as used for the source rate"})
    ds["no2_vertical_column_enhancement"] = (grid, vcd.astype("float32"), {
        "units": "mol m-2", "long_name": "NO2 vertical column enhancement = slant column / AMF (single plume-mean AMF)"})
    ds["plume_mask"] = (grid, used_mask.astype("uint8"), {
        "long_name": "Plume mask actually used (saved mask after hole-filling and any extra dilation/erosion)",
        "flag_values": np.array([0, 1], dtype="uint8"), "flag_meanings": "background plume"})
    ds["plume_mask_saved"] = (grid, saved_mask.astype("uint8"), {
        "long_name": "Plume mask as stored on disk, before hole-filling/dilation/erosion",
        "flag_values": np.array([0, 1], dtype="uint8"), "flag_meanings": "background plume"})

    # along-plume profile (source -> end of detectable plume)
    ds = ds.assign_coords(downwind_distance=("downwind", profile["extent"], {"units": "m", "long_name": "Distance downwind of the source"}))
    ds["emission_rate_profile"] = ("downwind", profile["profile"], {
        "units": "kg s-1", "long_name": "Cross-sectional-flux NOx emission rate Q(x), as NO2 mass"})
    ds["emission_rate_smoothed"] = ("downwind", profile["smoothed"], {
        "units": "kg s-1", "long_name": "Q(x) after Savitzky-Golay smoothing"})

    # scalars: the quantities that define the published estimate
    S = ds.__setitem__
    S("source_rate", _scalar(result["Q_kg_s"], "kg s-1", "NOx source rate as NO2-equivalent mass (mean of smoothed Q(x) above percentile_used)"))
    S("cems_rate", _scalar(cems_rate, "kg s-1", "CEMS ground-truth NOx rate (NaN where unavailable)"))
    S("wind_speed", _scalar(result["wind_speed_m_s"], "m s-1", "Wind speed used in the source rate (after any perturbation, floored at 0.3)"))
    S("wind_speed_unperturbed", _scalar(wind["speed_unperturbed"], "m s-1", "Wind speed before any offset/fraction perturbation"))
    S("wind_direction_from", _scalar(wind["dir_from"], "degree", f"Wind direction, meteorological 'from' (clockwise from north), at {wind['dir_level_m']:.0f} m AGL"))
    S("wind_height", _scalar(wind["height_m"], "m", "Height AGL of the wind speed used"))
    S("amf", _scalar(amf, "1", "Scattering air mass factor applied to the slant columns"))
    S("amf_plume_height", _scalar(amf_in["plume_height_m"], "m", "Plume height assumed in the AMF look-up"))
    S("amf_surface_reflectance", _scalar(amf_in["gler"], "1", "Plume-mean Lambertian-equivalent reflectance, 405-465 nm (EMIT L2A)"))
    S("amf_solar_zenith_angle", _scalar(amf_in["sza"], "degree", "Plume-mean solar zenith angle"))
    S("amf_viewing_zenith_angle", _scalar(amf_in["vza"], "degree", "Plume-mean viewing zenith angle"))
    S("amf_relative_azimuth_angle", _scalar(amf_in["raa"], "degree", "Plume-mean relative azimuth angle"))
    S("amf_surface_pressure", _scalar(amf_in["surface_pressure_hpa"], "hPa", "GEOS-FP surface pressure (clipped to the LUT range)"))
    S("amf_temperature_2m", _scalar(amf_in["t2m_k"], "K", "GEOS-FP 2 m temperature, used for the scale height"))
    S("nox_no2_ratio", _scalar(result["nox_no2_ratio"], "1", "NOx-to-NO2 ratio alpha applied to the NO2 flux"))
    S("percentile_used", _scalar(result["percentile_used"], "percent", "Percentile threshold on smoothed Q(x)"))
    S("plume_length", _scalar(result["plume_length_km"], "km", "Downwind plume length"))
    S("smoothing_window", _scalar(result["smoothing_window"], "pixel", "Savitzky-Golay window length", dtype="int32"))
    S("smoothing_polyorder", _scalar(result["smoothing_polyorder"], "1", "Savitzky-Golay polynomial order", dtype="int32"))
    S("pixel_pitch_native", _scalar(scene.pix_pitch_m, "m", "Native orthorectified pixel pitch"))
    S("pixel_pitch_integration", _scalar(result["pix_pitch_m"], "m", "Pixel pitch used in the cross-section integral (rescaled for rotation)"))
    S("plume_pixels", _scalar(result["plume_pixels_init"], "pixel", "Pixels in the plume mask used", dtype="int64"))
    S("source_latitude", _scalar(scene.loc_data["LAT"], "degrees_north", "Source latitude"))
    S("source_longitude", _scalar(scene.loc_data["LON"], "degrees_east", "Source longitude"))
    for k, v in wind["extra"].items():
        S(k, _scalar(v, "1", f"Wind-product diagnostic: {k}"))

    p = {k: v for k, v in dataclasses.asdict(params).items() if v is not None and k != "wind_speed_resolver"}
    p["mask_shift_px"] = list(p["mask_shift_px"])
    ds.attrs.update({
        "Conventions": CONVENTIONS,
        "title": f"EMIT NO2 plume retrieval and NOx source rate: {scene.loc_name} {scene.granule_name}",
        "source_name": scene.loc_name,
        "emit_granule": scene.granule_name,
        "observation_time_utc": scene.obs_time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "wind_source": wind["source"],
        "mask_file": os.path.basename(mask_path),
        "retrieval_field_note": "Retrieval is on the EMIT orthorectified grid; NaN marks invalid pixels. Rows run north to south.",
        "processing_parameters": json.dumps(p, sort_keys=True),
        "history": f"{datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')} created by export_scene_netcdf.py",
    })
    return ds


def encoding_for(ds: xr.Dataset) -> dict:
    enc = {}
    for name, var in ds.data_vars.items():
        if var.ndim == 0:
            continue
        enc[name] = {"zlib": True, "complevel": 4}
        if var.dtype.kind == "f":
            enc[name]["_FillValue"] = np.nan
    return enc


# ---------------------------------------------------------------------------
# per-scene driver
# ---------------------------------------------------------------------------
def export_scene(loc_name, granule, params, out_path, cems_rate=float("nan"), mask_path=None, lat=None, lon=None) -> dict:
    geosfp = uses_geosfp(loc_name)
    params = dataclasses.replace(params, mask_path_override=mask_path)
    if geosfp:
        params = dataclasses.replace(params, wind_speed_resolver=geosfp_wind_resolver)

    registry = LOCS if geosfp else None
    override = {"LAT": lat, "LON": lon} if lat is not None and lon is not None else None
    scene = rsr.load_scene(loc_name, granule, loc_registry=registry, loc_data_override=override)

    with tempfile.TemporaryDirectory() as tmp:
        result = rsr.compute_q_from_scene(scene, params, save_dir=tmp)
        stem = f"{loc_name}_{granule}"
        profile = {k: np.load(os.path.join(tmp, f"{stem}_{k}.npy")) for k in ("profile", "smoothed", "extent")}

    mask_file, saved_mask, used_mask = load_masks(scene, params)
    amf_in = amf_inputs(scene, used_mask, params)
    check = rsr._compute_amf(scene.gler_field, scene.sza_field, scene.vza_field, scene.phi_field,
                             used_mask, scene.geosfp_info, params)
    if not np.isclose(check, result["amf_eff"], rtol=1e-6):
        raise RuntimeError(f"recomputed AMF {check} != AMF used for Q {result['amf_eff']}; saved mask/AMF inputs would be inconsistent")

    wind = wind_details(scene, params, geosfp)
    ds = build_dataset(scene, params, result, profile, saved_mask, used_mask, mask_file, wind, amf_in, cems_rate, geosfp)
    ds.to_netcdf(out_path, encoding=encoding_for(ds))

    return {"Q_KG_S": result["Q_kg_s"], "CEMS_KG_S": cems_rate, "WIND_SPEED_M_S": result["wind_speed_m_s"],
            "WIND_DIR_FROM_DEG": wind["dir_from"], "AMF": result["amf_eff"],
            "PERCENTILE": result["percentile_used"], "PLUME_LENGTH_KM": result["plume_length_km"]}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output-dir", required=True, help="Directory for the per-scene .nc files and index.csv")
    p.add_argument("--input-csv", default=None, help="CSV with LOC_NAME, GRANULE (optional: CAMPD_CORR, MASK_PATH, LAT, LON)")
    p.add_argument("--riyadh", action="store_true", help=f"Add {RIYADH_LOC_NAME} scenes (from --granules, or auto-discovered under results)")
    p.add_argument("--granules", nargs="+", default=None, help="Explicit granule ids for --riyadh")
    p.add_argument("--skip-done", action="store_true", help="Skip scenes whose .nc already exists")

    # same perturbation knobs as the batch scripts; defaults reproduce the published settings
    p.add_argument("--mask-dilate-iters", type=int, default=rsr.DEFAULT_MASK_DILATE_ITERS)
    p.add_argument("--mask-erode-iters", type=int, default=rsr.DEFAULT_MASK_ERODE_ITERS)
    p.add_argument("--amf-plume-height-offset-m", type=float, default=None)
    p.add_argument("--amf-plume-height-perturb-frac", type=float, default=None)
    p.add_argument("--wind-height-m", type=float, default=None)
    p.add_argument("--wind-speed-offset-m-s", type=float, default=None)
    p.add_argument("--wind-speed-perturb-frac", type=float, default=None)
    p.add_argument("--smoothing-window", type=int, default=rsr.DEFAULT_SMOOTHING_WINDOW)
    p.add_argument("--smoothing-window-km", type=float, default=None)
    p.add_argument("--smoothing-polyorder", type=int, default=rsr.DEFAULT_SMOOTHING_POLYORDER)
    p.add_argument("--percentile-offset", type=float, default=None)
    p.add_argument("--percentile-perturb-frac", type=float, default=None)
    p.add_argument("--nox-no2-ratio", type=float, default=None)
    return p


def params_from_args(a: argparse.Namespace) -> rsr.Params:
    return rsr.Params(
        mask_dilate_iters=a.mask_dilate_iters, mask_erode_iters=a.mask_erode_iters,
        amf_plume_height_offset_m=a.amf_plume_height_offset_m, amf_plume_height_perturb_frac=a.amf_plume_height_perturb_frac,
        wind_height_m=a.wind_height_m, wind_speed_offset_m_s=a.wind_speed_offset_m_s, wind_speed_perturb_frac=a.wind_speed_perturb_frac,
        smoothing_window=a.smoothing_window, smoothing_window_km=a.smoothing_window_km, smoothing_polyorder=a.smoothing_polyorder,
        percentile_offset=a.percentile_offset, percentile_perturb_frac=a.percentile_perturb_frac, nox_no2_ratio=a.nox_no2_ratio,
    )


def collect_scenes(args) -> pd.DataFrame:
    frames = []
    if args.input_csv:
        df = pd.read_csv(args.input_csv)
        missing = [c for c in ("LOC_NAME", "GRANULE") if c not in df.columns]
        if missing:
            raise ValueError(f"input CSV is missing required columns: {missing}")
        frames.append(df)
    if args.riyadh:
        granules = args.granules or discover_riyadh_granules()
        if not granules:
            raise ValueError(f"no *_retrieval_mask.npy files found under {rsr.MASK_BASE_DIR}/{RIYADH_LOC_NAME}")
        frames.append(pd.DataFrame({"LOC_NAME": RIYADH_LOC_NAME, "GRANULE": granules}))
    if not frames:
        raise SystemExit("give --input-csv and/or --riyadh")
    return pd.concat(frames, ignore_index=True)


def _opt(row, key):
    v = row.get(key)
    return None if v is None or pd.isna(v) else v


def main():
    args = build_arg_parser().parse_args()
    params = params_from_args(args)
    scenes = collect_scenes(args)
    os.makedirs(args.output_dir, exist_ok=True)
    index_path = os.path.join(args.output_dir, "index.csv")

    rows, n = [], len(scenes)
    for i, row in scenes.iterrows():
        loc_name, granule = row["LOC_NAME"], row["GRANULE"]
        fname = f"{loc_name}_{granule}.nc"
        out_path = os.path.join(args.output_dir, fname)
        entry = {"LOC_NAME": loc_name, "GRANULE": granule, "FILE": fname}
        print(f"[{i + 1}/{n}] {loc_name} {granule} ... ", end="", file=sys.stderr, flush=True)

        if args.skip_done and os.path.exists(out_path):
            entry["STATUS"] = "skipped"
            print("exists, skipped", file=sys.stderr)
            rows.append(entry)
            continue
        try:
            cems = _opt(row, "CAMPD_CORR")
            lat, lon = _opt(row, "LAT"), _opt(row, "LON")
            summary = export_scene(
                loc_name, granule, params, out_path,
                cems_rate=float(cems) if cems is not None else float("nan"),
                mask_path=_opt(row, "MASK_PATH"),
                lat=float(lat) if lat is not None else None, lon=float(lon) if lon is not None else None,
            )
            entry.update(summary, STATUS="ok")
            print(f"Q={summary['Q_KG_S']:.4f} kg/s", file=sys.stderr)
        except Exception as exc:  # keep going; a failed scene is recorded, not fatal
            entry.update(STATUS="failed", ERROR=str(exc), FILE="")
            print(f"FAILED: {exc}", file=sys.stderr)
        rows.append(entry)
        pd.DataFrame(rows, columns=list(INDEX_COLUMNS)).to_csv(index_path, index=False)  # survive a mid-run crash

    pd.DataFrame(rows, columns=list(INDEX_COLUMNS)).to_csv(index_path, index=False)
    n_bad = sum(r["STATUS"] == "failed" for r in rows)
    print(f"\nWrote {sum(r['STATUS'] == 'ok' for r in rows)} files to {args.output_dir} ({n_bad} failed); index: {index_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
