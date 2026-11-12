#!/usr/bin/env python3
"""
Ensemble sensitivity / error analysis for CSF source rates, per the SI
recipe in "Error Sources SI Inventory.pdf": for each scene, build a ~200
member joint Monte Carlo ensemble by drawing one level from each of 8 error
sources and rerunning compute_source_rate() (the same function
batch_run_source_rate.py drives), then also vary each source one-at-a-time
to see which one moves Q the most.

For plume i, member k:
    Q_i^(k) = f(dOmega_i^(k), mask^(k), AMF^(k), U^(k), alpha^(k), L_smooth^(k), P_Q^(k))
Report per scene:
    Q_i    = median_k(Q_i^(k))
    u_i    = (Q84_i - Q16_i) / (2 * Q50_i)      [joint ensemble]
    u_i,j  = (Q84_i,j - Q16_i,j) / (2 * Q50_i)  [source j varied alone, others nominal]

THE EIGHT ERROR SOURCES, mapped onto run_source_rate.Params:

  1. NO2 retrieval/background noise (Random)
     No independent-Gaussian-per-pixel noise is added (the PDF explicitly warns
     against that - retrieval errors are spatially correlated). Instead we use
     its suggested simpler proxy: shift the saved plume mask to several nearby
     source-free locations in the same scene and rerun the full pipeline. Each
     "pseudo-Q" from a source-free patch is a direct, same-units (kg/s)
     estimate of the background noise floor. One pseudo-Q is added to each
     joint-ensemble draw as the noise realization for that member.

  2. Plume mask (Algorithmic)
     The PDF's nominal test is "recompute Q using several nearby acceptable
     masks: threshold p-2/p/p+2, morphology n-1/n/n+1". This script starts
     from an already-finalized saved mask (see run_source_rate.py), so
     "nearby acceptable masks" is implemented as: the saved mask as-is,
     extra-dilated (looser/larger), and extra-eroded (tighter/smaller), via
     Params.mask_dilate_iters / mask_erode_iters.

  3. AMF / plume altitude (Structural): 250 / 500 / 750 m, via
     Params.amf_plume_height_offset_m in {-250, 0, +250}.

  4. Wind height (Structural/random): HRRR wind as a point estimate (not a
     layer average) at 300 / 500 / 700 m, via Params.wind_height_m in
     {300, None, 700} - a +/-200 m perturbation around the 500 m default.
     Each height is approximated via a narrow (h-1, h+1) m window
     (run_source_rate.WIND_LAYER_HALF_WIDTH_M=1), since HRRR only exposes a
     layer-average API. "nominal" is None (run_source_rate.py's own default,
     the (499,501) m point-estimate window) rather than an explicit 500, so
     this always tracks the real default even if it changes.

  5. Wind measurement uncertainty (Random/structural): HRRR's own intrinsic
     error, independent of which height is sampled - a genuinely different
     error source from #4 above (that one asks "which height is right", this
     one asks "how accurate is HRRR even at the right height"). +/- 2.0 m/s
     additive offset on the resolved wind speed, via
     Params.wind_speed_offset_m_s - the error SD Varon et al. (2018, their
     Fig. 9) find for GEOS-FP U10 vs MesoWest at large plume lifetimes
     (tau=1h): 2.0 m/s (vs 2.5 m/s for tau=5 min); team chose the more
     conservative/larger-plume value. run_source_rate.WIND_SPEED_FLOOR_MS
     (0.3 m/s) prevents this from ever pushing wind speed to zero or
     negative.

  6. NOx:NO2 ratio alpha (Structural): nominal 1.38, +/- 0.1 (Beirle et al.,
     2023), giving ALPHA_LOW=1.28, ALPHA_HIGH=1.48 below.

  7. Savitzky-Golay smoothing length (Algorithmic): 2 / 3 / 4 km, via
     Params.smoothing_window_km (converted to pixels assuming the fixed
     60 m EMIT pixel pitch, run_source_rate.NATIVE_PIX_PITCH_M). Nominal is
     3 km, matching run_source_rate.py's actual default (window_length=51 px
     @ 60 m/px = 3.06 km - confirmed by reverse-engineering a saved
     profile/smoothed pair, max_abs_err=0.0). Low/high keep the same +/-33%
     relative spacing as the original 4/6/8 km levels.

  8. Final-Q percentile threshold (Algorithmic): the length-dependent adaptive
     default, +/- 10 percentage points, via Params.percentile_offset.

Usage:
    python error_analysis.py --input-csv plotdf_data.csv \
        --output-csv error_analysis_per_scene.csv \
        --summary-csv error_analysis_summary.csv

Performance: each scene's expensive, parameter-independent work (reading and
orthorectifying the radiance/L2A/OBS netCDFs, GEOS-FP lookup, metadata lookup)
is done exactly once via run_source_rate.load_scene(), then reused across all
n_background + n_joint + OAT draws for that scene via
run_source_rate.compute_q_from_scene() - so the per-scene call count
(229 by default: 8 background + 200 joint + 21 OAT, 7 axes x 3 levels) no longer multiplies the
netCDF I/O cost, only the cheap mask/AMF/rotate/integrate math. The AMF LUT
and the metadata CSV are also cached process-wide (loaded once, not once per
call), and HRRR wind lookups are memoized by (lat, lon, obs_time, layer) so
the small number of distinct wind heights actually used gets fetched once
each rather than once per draw. Still use --n-ensemble to shrink a test run.

Parallelism: --workers N runs N scenes concurrently in separate processes
(each scene's full ensemble - background + joint + OAT - is independent of
every other scene's, so this parallelizes cleanly at the scene level). With
only 42-ish scenes, --workers beyond the number of PENDING scenes buys
nothing further - there's no finer-grained parallelism within a single
scene's own draws. Each worker process gets its own AMF LUT / metadata /
HRRR caches (they don't share across processes), so memory scales roughly
with --workers, not just with data size. Reproducibility: with --workers > 1,
scenes finish in whatever order the OS schedules them, but each scene's own
random draws are unaffected by that - every scene gets an independent seed
derived from --seed via numpy's SeedSequence.spawn(), keyed to that scene's
position in the input CSV, not to when it happens to run or how many other
scenes were skipped via --resume. So results are bit-for-bit reproducible
per scene regardless of --workers or --resume.
"""
import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

# Must happen before numpy/scipy are imported (BLAS reads these at init, not
# per-call). Without this, each of the N worker processes independently tries
# to multi-thread its own linear algebra (OpenBLAS/MKL default to using ALL
# visible cores per process) - with --workers 42 on a 64-core node that's up
# to 42x oversubscription of the same 64 cores, which is why --workers can
# fail to speed anything up (or even slow things down) without this. Uses
# setdefault so an explicit shell-level override (e.g. for a --workers 1 run
# where you actually want BLAS multithreading) still wins.
for _env_var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_env_var, "1")

import numpy as np
import pandas as pd

import source_rate as rsr

REQUIRED_INPUT_COLUMNS = ("LOC_NAME", "GRANULE", "CAMPD_CORR")

# ---- 1. NO2 retrieval / background noise: nearby source-free mask shifts ----
N_BACKGROUND_SHIFTS_DEFAULT = 8
BACKGROUND_SHIFT_RANGE_PX = (100, 300)  # magnitude range; direction is random

# ---- 2. Plume mask: nearby acceptable masks (proxy via dilate/erode) ----
MASK_LEVELS = {
    "tighter": {"mask_dilate_iters": 0, "mask_erode_iters": 2},
    "nominal": {"mask_dilate_iters": 0, "mask_erode_iters": 0},
    "looser": {"mask_dilate_iters": 2, "mask_erode_iters": 0},
}

# ---- 3. AMF / plume altitude: 250 / 500 / 750 m ----
AMF_HEIGHT_OFFSETS_M = {"low_250m": -250.0, "nominal_500m": 0.0, "high_750m": 250.0}

# ---- 4a. Wind height-sampling sensitivity: HRRR wind as a point estimate
# (not a layer average) at 300 / 500 / 700 m, i.e. +/-200 m around the 500 m
# default. "nominal" must be None (run_source_rate.Params' own default - the
# (499,501) m point-estimate window), not an explicit 500.0, so this always
# tracks the real default even if it changes. This axis answers "if the true
# injection/plume height were different, how much would Q change" - it is NOT
# a test of HRRR's own accuracy at a given height (see WIND_UNCERTAINTY_MS).
WIND_HEIGHTS_M = {"low_300m": 300.0, "nominal_500m": None, "high_700m": 700.0}

# ---- 4b. Wind measurement uncertainty: HRRR's own intrinsic error,
# independent of which layer is sampled - this is a genuinely different error
# source from WIND_HEIGHTS_M above (which is about *choice* of layer, not
# accuracy within a given layer), so it is drawn as its own axis rather than
# folded into it. +/- 2.0 m/s additive offset on the resolved wind speed -
# the error SD Varon et al. (2018, their Fig. 9) find for GEOS-FP U10 vs
# MesoWest airport observations at large plume lifetimes (tau=1h; 2.5 m/s at
# tau=5 min). run_source_rate.py floors the result at WIND_SPEED_FLOOR_MS
# (0.3 m/s) so this can never push wind speed to zero or negative.
WIND_UNCERTAINTY_MS = {"low": -2.0, "nominal": 0.0, "high": 2.0}

# ---- 5. NOx:NO2 ratio alpha: 1.38 +/- 0.1 (Beirle et al., 2023) ----
ALPHA_LOW = 1.28
ALPHA_HIGH = 1.48
ALPHA_LEVELS = {"low": ALPHA_LOW, "nominal": rsr.NOX_NO2_RATIO, "high": ALPHA_HIGH}

# ---- 6. Savitzky-Golay smoothing length: 2 / 3 / 4 km (nominal 3 km matches
# run_source_rate.py's actual default of window_length=51 px @ 60 m/px) ----
SMOOTHING_KM_LEVELS = {"low_2km": 2.0, "nominal_3km": 3.0, "high_4km": 4.0}

# ---- 7. Final-Q percentile threshold: nominal +/- 10 percentage points ----
PERCENTILE_OFFSETS = {"low": -10.0, "nominal": 0.0, "high": 10.0}

GRID_AXES = {
    "PLUME_MASK": MASK_LEVELS,
    "AMF_HEIGHT": AMF_HEIGHT_OFFSETS_M,
    "WIND_SPEED": WIND_HEIGHTS_M,
    "WIND_UNCERTAINTY": WIND_UNCERTAINTY_MS,
    "ALPHA": ALPHA_LEVELS,
    "SMOOTHING": SMOOTHING_KM_LEVELS,
    "PERCENTILE": PERCENTILE_OFFSETS,
}

ERROR_COMPONENTS = ("NO2_BACKGROUND",) + tuple(GRID_AXES.keys())


def _overrides_for_level(axis: str, level_value) -> dict:
    if axis == "PLUME_MASK":
        return dict(level_value)
    if axis == "AMF_HEIGHT":
        return {"amf_plume_height_offset_m": level_value}
    if axis == "WIND_SPEED":
        return {"wind_height_m": level_value}
    if axis == "WIND_UNCERTAINTY":
        return {"wind_speed_offset_m_s": level_value}
    if axis == "ALPHA":
        return {"nox_no2_ratio": level_value}
    if axis == "SMOOTHING":
        return {"smoothing_window_km": level_value}
    if axis == "PERCENTILE":
        return {"percentile_offset": level_value}
    raise ValueError(f"unknown axis {axis}")


def sample_background_shift(rng: np.random.Generator) -> tuple:
    lo, hi = BACKGROUND_SHIFT_RANGE_PX
    magnitude_r = rng.uniform(lo, hi) * rng.choice([-1, 1])
    magnitude_c = rng.uniform(lo, hi) * rng.choice([-1, 1])
    return (int(round(magnitude_r)), int(round(magnitude_c)))


def build_background_pool(scene, rng: np.random.Generator, n_shifts: int) -> list:
    pool = []
    for _ in range(n_shifts):
        shift_px = sample_background_shift(rng)
        params = rsr.Params(mask_shift_px=shift_px)
        try:
            result = rsr.compute_q_from_scene(scene, params)
        except Exception:
            continue
        pool.append(result["Q_kg_s"])
    return pool


def sample_joint_overrides(rng: np.random.Generator) -> dict:
    overrides = {}
    for axis, levels in GRID_AXES.items():
        level_value = levels[rng.choice(list(levels.keys()))]
        overrides.update(_overrides_for_level(axis, level_value))
    return overrides


def percentile_spread(values, q50_override=None):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan"), float("nan"), float("nan"), float("nan")
    q50 = q50_override if q50_override is not None else float(np.median(values))
    q16, q84 = np.percentile(values, [16, 84])
    denom = 2.0 * q50
    u = float(q84 - q16) / denom if denom else float("nan")
    return q50, float(q16), float(q84), u


def _std_mean(values):
    """std/mean ingredients for a set of Q draws (e.g. one axis's OAT levels,
    or the background pool) - same std/mean convention as Q_STD/Q_MEAN for
    the joint ensemble, just applied per-component so each row in Table S2
    can be reported on a std/mean basis too, not just percentile-based."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan"), float("nan")
    return float(np.std(values)), float(np.mean(values))


def run_scene(loc_name: str, granule_name: str, campd_corr: float, n_joint: int, n_background: int,
              rng: np.random.Generator) -> dict:
    row = {"LOC_NAME": loc_name, "GRANULE": granule_name, "CAMPD": campd_corr}

    # load once; every draw below reuses this instead of re-reading/re-orthorectifying the scene's netCDFs
    try:
        scene = rsr.load_scene(loc_name, granule_name)
    except Exception as exc:
        print(f"  FAILED to load scene: {exc}", file=sys.stderr)
        row["N_BACKGROUND_OK"] = 0
        row["N_JOINT_OK"] = 0
        row["Q_NOMINAL"] = row["Q_STD"] = row["Q_MEAN"] = row["STD_OVER_MEAN"] = float("nan")
        row["Q50"] = row["Q16"] = row["Q84"] = row["U_JOINT"] = float("nan")
        for comp in ERROR_COMPONENTS:
            row[f"U_{comp}"] = float("nan")
        return row

    # the single deterministic default-Params() output - this is what Fig 4c
    # actually plots and what CAMPD should be differenced against, distinct
    # from Q50 (the ensemble MEDIAN, a different quantity - see below)
    try:
        row["Q_NOMINAL"] = rsr.compute_q_from_scene(scene, rsr.Params())["Q_kg_s"]
    except Exception:
        row["Q_NOMINAL"] = float("nan")

    background_pool = build_background_pool(scene, rng, n_background)
    row["N_BACKGROUND_OK"] = len(background_pool)

    joint_qs = []
    for _ in range(n_joint):
        overrides = sample_joint_overrides(rng)
        params = rsr.Params(**overrides)
        try:
            result = rsr.compute_q_from_scene(scene, params)
        except Exception:
            continue
        q = result["Q_kg_s"]
        if background_pool:
            q += rng.choice(background_pool)
        joint_qs.append(q)
    row["N_JOINT_OK"] = len(joint_qs)

    q50, q16, q84, u_joint = percentile_spread(joint_qs)
    row["Q50"] = q50
    row["Q16"] = q16
    row["Q84"] = q84
    row["U_JOINT"] = u_joint
    # real sample std of the joint ensemble (NOT the (Q84-Q16)/2 percentile
    # proxy U_JOINT uses) - this is the quantity that's actually comparable,
    # std-for-std, to the manuscript's std/mean CEMS precision figure
    row["Q_STD"] = float(np.std(joint_qs)) if joint_qs else float("nan")
    # Daniel's recipe (Slack, 2026-09): per scene, estimate a bunch of
    # possible source rates (the ensemble), take THEIR OWN mean and std, save
    # std/mean. Q_MEAN is the ensemble's arithmetic mean - NOT Q_NOMINAL (a
    # single deterministic default-Params() run) and NOT Q50 (the ensemble
    # MEDIAN) - those are three genuinely different quantities.
    row["Q_MEAN"] = float(np.mean(joint_qs)) if joint_qs else float("nan")
    row["STD_OVER_MEAN"] = row["Q_STD"] / row["Q_MEAN"] if row["Q_MEAN"] else float("nan")

    if not np.isfinite(q50):
        for comp in ERROR_COMPONENTS:
            row[f"U_{comp}"] = float("nan")
            row[f"STD_{comp}"] = float("nan")
            row[f"MEAN_{comp}"] = float("nan")
        return row

    # NO2 retrieval/background: OAT uncertainty directly from the pool itself.
    # background_pool is sampled from SOURCE-FREE regions, so it's ~0-centered
    # by construction - it isn't a perturbation of the real plume signal the
    # way the other 7 axes are, so its OWN mean is a degenerate (near-zero)
    # denominator: dividing by it makes the ratio balloon, as Kevin diagnosed
    # (Slack, 2026-09) and Daniel confirmed the fix for - normalize the
    # background noise's magnitude against the SCENE's actual signal
    # (Q_NOMINAL), not against the noise's own near-zero mean. This matches
    # how U_NO2_BACKGROUND already handled it (q50_override=q50 below, not
    # the background pool's own median) - only the newer STD_/MEAN_ columns
    # had the bug.
    _, _, _, u_bg = percentile_spread(background_pool, q50_override=q50)
    row["U_NO2_BACKGROUND"] = u_bg
    bg_std, _bg_own_mean = _std_mean(background_pool)
    row["STD_NO2_BACKGROUND"] = bg_std
    row["MEAN_NO2_BACKGROUND"] = row["Q_NOMINAL"]

    # remaining 7 axes: vary this source alone, everything else at nominal
    for axis, levels in GRID_AXES.items():
        axis_qs = []
        for level_value in levels.values():
            overrides = _overrides_for_level(axis, level_value)
            params = rsr.Params(**overrides)
            try:
                result = rsr.compute_q_from_scene(scene, params)
            except Exception:
                continue
            axis_qs.append(result["Q_kg_s"])
        _, _, _, u_axis = percentile_spread(axis_qs, q50_override=q50)
        row[f"U_{axis}"] = u_axis
        row[f"STD_{axis}"], row[f"MEAN_{axis}"] = _std_mean(axis_qs)

    return row


def _run_scene_with_seed(loc_name: str, granule_name: str, campd_corr: float, n_joint: int,
                          n_background: int, seed_seq: np.random.SeedSequence) -> dict:
    """Module-level (picklable) wrapper for ProcessPoolExecutor: builds this
    scene's own independent RNG from its seed and runs it. Never share a
    single np.random.Generator instance across worker processes - pickling it
    to multiple workers would hand them identical state and produce
    correlated (not independent) draws; a SeedSequence per scene avoids that."""
    rng = np.random.default_rng(seed_seq)
    return run_scene(loc_name, granule_name, campd_corr, n_joint, n_background, rng)


def _failed_row(loc_name: str, granule_name: str, campd_corr: float) -> dict:
    row = {"LOC_NAME": loc_name, "GRANULE": granule_name, "CAMPD": campd_corr,
           "Q_NOMINAL": float("nan"), "Q50": float("nan"), "Q16": float("nan"), "Q84": float("nan"),
           "Q_STD": float("nan"), "Q_MEAN": float("nan"), "STD_OVER_MEAN": float("nan"),
           "U_JOINT": float("nan"), "N_JOINT_OK": 0, "N_BACKGROUND_OK": 0}
    for comp in ERROR_COMPONENTS:
        row[f"U_{comp}"] = float("nan")
        row[f"STD_{comp}"] = float("nan")
        row[f"MEAN_{comp}"] = float("nan")
    return row


def build_summary_table(out_df: pd.DataFrame) -> pd.DataFrame:
    """Component-level summary table (matches the PDF's suggested SI table):
    median/IQR of the percentile-based U_<component>, plus the std/mean
    statistics (pooled and mean-of-ratios) for every component, not just
    JOINT. Pure function of an already-computed per-scene DataFrame - no
    pipeline calls - so this can be rerun instantly on a saved per-scene CSV
    without recomputing the ensemble. Robust to older CSVs that predate the
    STD_<comp>/MEAN_<comp> columns: those rows just get NaN for the
    std/mean columns rather than raising."""
    component_std_mean_cols = {"JOINT": ("Q_STD", "Q_MEAN")}
    for c in ERROR_COMPONENTS:
        component_std_mean_cols[c] = (f"STD_{c}", f"MEAN_{c}")

    summary_rows = []
    for comp, (std_col, mean_col) in component_std_mean_cols.items():
        u_col = "U_JOINT" if comp == "JOINT" else f"U_{comp}"
        u_vals = out_df[u_col].dropna() if u_col in out_df.columns else pd.Series(dtype=float)

        std_mean_valid = None
        if std_col in out_df.columns and mean_col in out_df.columns:
            std_mean_valid = out_df.dropna(subset=[std_col, mean_col])
            std_mean_valid = std_mean_valid[std_mean_valid[mean_col] != 0]

        if std_mean_valid is not None and len(std_mean_valid):
            pooled = float(np.sqrt(np.mean(std_mean_valid[std_col] ** 2))) / float(std_mean_valid[mean_col].mean())
            mean_of_ratios = float((std_mean_valid[std_col] / std_mean_valid[mean_col]).mean())
            n_scenes = len(std_mean_valid)
        else:
            pooled = mean_of_ratios = float("nan")
            n_scenes = 0

        summary_rows.append({
            "COMPONENT": comp,
            "MEDIAN_FRACTIONAL_UNCERTAINTY": u_vals.median() if len(u_vals) else float("nan"),
            "IQR_LOW": u_vals.quantile(0.25) if len(u_vals) else float("nan"),
            "IQR_HIGH": u_vals.quantile(0.75) if len(u_vals) else float("nan"),
            "STD_OVER_MEAN_POOLED": pooled,
            "STD_OVER_MEAN_MEAN_OF_RATIOS": mean_of_ratios,
            "N_SCENES": n_scenes,
        })
    return pd.DataFrame(summary_rows)


def print_precision_comparison(out_df: pd.DataFrame, file=sys.stderr) -> None:
    """Internal-vs-external precision comparison, per PDF section 9. Pure
    function of an already-computed per-scene DataFrame - rerun instantly on
    a saved CSV. Requires Q_STD/Q_MEAN/STD_OVER_MEAN (added partway through
    this project); prints a note and returns if an older CSV lacks them,
    rather than raising.

    UNRESOLVED (as of 2026-09 team discussion): whether the manuscript's own
    Fig 4c "43% precision" is (a) a single POOLED std(residuals)/mean(CEMS)
    over all 42 points at once, or (b) the MEAN of 42 individual per-plume
    (std/mean) ratios. Confirm against the actual Fig 4c code before trusting
    either comparison below - the two conventions generally give different
    numbers, because mean-of-ratios weights every scene equally regardless of
    its magnitude, while pooling weights larger-magnitude scenes more
    heavily, the same way a real pooled regression-residual std would.
    """
    required = ["Q_STD", "Q_MEAN", "STD_OVER_MEAN"]
    if not all(c in out_df.columns for c in required):
        print(
            "\n(skipping internal/external precision comparison: this CSV predates the "
            "Q_STD/Q_MEAN/STD_OVER_MEAN columns - rerun error_analysis.py to get them)",
            file=file,
        )
        return

    valid = out_df.dropna(subset=required)
    if not len(valid):
        return

    internal_pooled = float(np.sqrt(np.mean(valid["Q_STD"] ** 2))) / float(valid["Q_MEAN"].mean())
    internal_mean_of_ratios = float(valid["STD_OVER_MEAN"].mean())

    external_precision = None
    valid_ext = out_df.iloc[0:0]
    if "Q_NOMINAL" in out_df.columns and "CAMPD" in out_df.columns:
        valid_ext = out_df.dropna(subset=["Q_NOMINAL", "CAMPD"])
        valid_ext = valid_ext[valid_ext["CAMPD"] != 0]
        if len(valid_ext):
            residuals = valid_ext["Q_NOMINAL"] - valid_ext["CAMPD"]
            external_precision = float(np.std(residuals)) / float(valid_ext["CAMPD"].mean())

    median_u_joint = out_df["U_JOINT"].median() if "U_JOINT" in out_df.columns else float("nan")
    msg = (
        f"\nInternal precision, pooled (sqrt(mean(std_i^2))/mean(mean_i)), n={len(valid)}: {internal_pooled:.1%}\n"
        f"Internal precision, mean-of-per-scene-ratios (Daniel's recipe), n={len(valid)}: {internal_mean_of_ratios:.1%}\n"
    )
    if external_precision is not None:
        msg += (
            f"External precision (pooled std of Q_NOMINAL-CAMPD residuals / mean CAMPD), n={len(valid_ext)}: {external_precision:.1%}\n"
            "Compare against whichever internal number above matches how your Fig 4c code actually computes "
            "43% - pooled vs mean-of-ratios are NOT interchangeable, confirm the definition before quoting.\n"
        )
    msg += (
        f"(for reference, the median-of-per-scene-(Q84-Q16)/2 statistic was: {median_u_joint:.1%} - "
        "a different, percentile-based statistic, kept here only for continuity with earlier runs)"
    )
    print(msg, file=file)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input-csv", required=True, help="CSV with LOC_NAME, GRANULE, CAMPD_CORR columns")
    p.add_argument("--output-csv", required=True, help="Where to write per-scene Q50/Q16/Q84 and per-component u_ij")
    p.add_argument("--summary-csv", default=None, help="Optional: where to write the across-scene median-uncertainty-per-component table")
    p.add_argument("--n-ensemble", type=int, default=200, help="Joint ensemble members per scene (default: 200, per the SI recipe)")
    p.add_argument("--n-background", type=int, default=N_BACKGROUND_SHIFTS_DEFAULT, help=f"Nearby source-free mask shifts per scene used to estimate retrieval/background noise (default: {N_BACKGROUND_SHIFTS_DEFAULT})")
    p.add_argument("--seed", type=int, default=None, help="RNG seed for reproducibility")
    p.add_argument("--resume", action="store_true", help="If --output-csv already has rows for a (LOC_NAME, GRANULE), skip recomputing them and append only the missing scenes. Does NOT verify that --n-ensemble/--n-background/--seed match the run that produced the existing file - that's on you.")
    p.add_argument("--workers", type=int, default=1, help="Run this many scenes concurrently in separate processes (default: 1, serial). Beyond the number of pending scenes this buys nothing further - parallelism here is per-scene, not per-draw.")
    return p


def main():
    args = build_arg_parser().parse_args()

    scenes = pd.read_csv(args.input_csv)
    missing = [c for c in REQUIRED_INPUT_COLUMNS if c not in scenes.columns]
    if missing:
        raise ValueError(f"input CSV is missing required columns: {missing}")

    out_columns = (
        ["LOC_NAME", "GRANULE", "CAMPD", "Q_NOMINAL", "Q50", "Q16", "Q84", "Q_STD", "Q_MEAN", "STD_OVER_MEAN", "U_JOINT"]
        + [f"U_{c}" for c in ERROR_COMPONENTS]
        + [f"STD_{c}" for c in ERROR_COMPONENTS]
        + [f"MEAN_{c}" for c in ERROR_COMPONENTS]
        + ["N_JOINT_OK", "N_BACKGROUND_OK"]
    )

    # keyed by (LOC_NAME, GRANULE) rather than a plain list, so results can be
    # written back out in the ORIGINAL input CSV order regardless of what
    # order parallel workers finish in
    rows_by_key = {}
    completed = set()
    if args.resume and os.path.exists(args.output_csv):
        existing_df = pd.read_csv(args.output_csv)
        ok_mask = existing_df["Q50"].notna()
        existing_ok = existing_df[ok_mask]
        n_failed = int((~ok_mask).sum())
        for _, r in existing_ok.iterrows():
            rows_by_key[(r["LOC_NAME"], r["GRANULE"])] = r.to_dict()
        completed = set(rows_by_key.keys())
        print(
            f"--resume: found {len(completed)} already-completed scene(s) in {args.output_csv}; skipping them."
            + (f" {n_failed} previously-failed scene(s) will be retried." if n_failed else "")
            + "\nThis does not check that --n-ensemble/--n-background/--seed match the run that produced "
            "those rows - mixing parameterizations across a resumed run silently produces an inconsistent table.",
            file=sys.stderr,
        )

    n = len(scenes)
    # one independent seed per scene, keyed to its position in the ORIGINAL
    # (full, unfiltered) scene list - reproducible per scene regardless of
    # --workers or which scenes --resume skips
    child_seeds = np.random.SeedSequence(args.seed).spawn(n)

    pending = []
    for i, scene_row in scenes.iterrows():
        key = (scene_row["LOC_NAME"], scene_row["GRANULE"])
        if key in completed:
            print(f"[{i + 1}/{n}] {key[0]} {key[1]} ... skipped (already in {args.output_csv})", file=sys.stderr)
            continue
        pending.append((i, scene_row))

    def write_csv():
        ordered = [rows_by_key[(r["LOC_NAME"], r["GRANULE"])] for _, r in scenes.iterrows() if (r["LOC_NAME"], r["GRANULE"]) in rows_by_key]
        pd.DataFrame(ordered, columns=out_columns).to_csv(args.output_csv, index=False)

    def log_result(idx, done, total, loc_name, granule_name, row):
        print(
            f"[{done}/{total}] (row {idx + 1}/{n}) {loc_name} {granule_name}: "
            f"Q50={row['Q50']:.4f} kg/s  u_joint={row['U_JOINT']:.3f}  "
            f"(n_joint_ok={row['N_JOINT_OK']}/{args.n_ensemble}, n_bg_ok={row['N_BACKGROUND_OK']}/{args.n_background})",
            file=sys.stderr,
        )

    if args.workers <= 1:
        for done, (i, scene_row) in enumerate(pending, start=1):
            loc_name, granule_name, campd_corr = scene_row["LOC_NAME"], scene_row["GRANULE"], scene_row["CAMPD_CORR"]
            print(f"[{done}/{len(pending)}] (row {i + 1}/{n}) {loc_name} {granule_name} ...", file=sys.stderr, flush=True)
            try:
                row = _run_scene_with_seed(loc_name, granule_name, campd_corr, args.n_ensemble, args.n_background, child_seeds[i])
            except Exception as exc:
                print(f"[{done}/{len(pending)}] (row {i + 1}/{n}) {loc_name} {granule_name} FAILED: {exc}", file=sys.stderr)
                row = _failed_row(loc_name, granule_name, campd_corr)
            rows_by_key[(loc_name, granule_name)] = row
            log_result(i, done, len(pending), loc_name, granule_name, row)
            write_csv()  # after every scene, so a crash/interruption doesn't lose finished work
    else:
        print(f"Running {len(pending)} scene(s) with up to {args.workers} parallel workers ...", file=sys.stderr)
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            future_to_scene = {
                executor.submit(
                    _run_scene_with_seed, r["LOC_NAME"], r["GRANULE"], r["CAMPD_CORR"],
                    args.n_ensemble, args.n_background, child_seeds[i],
                ): (i, r["LOC_NAME"], r["GRANULE"], r["CAMPD_CORR"])
                for i, r in pending
            }
            done = 0
            for future in as_completed(future_to_scene):
                i, loc_name, granule_name, campd_corr = future_to_scene[future]
                done += 1
                try:
                    row = future.result()
                except Exception as exc:
                    print(f"[{done}/{len(pending)}] (row {i + 1}/{n}) {loc_name} {granule_name} FAILED (worker exception): {exc}", file=sys.stderr)
                    row = _failed_row(loc_name, granule_name, campd_corr)
                rows_by_key[(loc_name, granule_name)] = row
                log_result(i, done, len(pending), loc_name, granule_name, row)
                write_csv()  # after every scene, so a crash/interruption doesn't lose finished work

    out_df = pd.DataFrame(
        [rows_by_key[(r["LOC_NAME"], r["GRANULE"])] for _, r in scenes.iterrows() if (r["LOC_NAME"], r["GRANULE"]) in rows_by_key],
        columns=out_columns,
    )

    summary_df = build_summary_table(out_df)

    if args.summary_csv:
        summary_df.to_csv(args.summary_csv, index=False)

    print("\n" + summary_df.to_string(index=False), file=sys.stderr)

    print_precision_comparison(out_df, file=sys.stderr)

    print(f"\nWrote {len(rows_by_key)} rows to {args.output_csv}", file=sys.stderr)
    if args.summary_csv:
        print(f"Wrote component summary to {args.summary_csv}", file=sys.stderr)


if __name__ == "__main__":
    main()
