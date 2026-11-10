import os
import logging
import numpy as np
import xarray as xr
from scipy.optimize import curve_fit
import os
import logging
from datetime import datetime
import cfgrib

def zero_pad_number(n):
    return f"{int(n):02d}"

def download_hrrr(date_time, forecast_hour=0, product="wrfsfc", cache="./../dat/_hrrr/" ):
    """
    Download HRRR data from the NOAA public S3 bucket.

    Parameters
    ----------
    date_time : datetime
        Initialization time (UTC)
    forecast_hour : int
        Forecast hour (0 = analysis)
    product : str
        'wrfsfc' (surface) or 'wrfprs' (pressure levels)
    cache : str
        Local cache directory
    """

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s"
    )

    os.makedirs(cache, exist_ok=True)

    ymd = date_time.strftime("%Y%m%d")
    hh = zero_pad_number(date_time.hour)
    fh = zero_pad_number(forecast_hour)

    fname = f"hrrr.t{hh}z.{product}f{fh}.grib2"
    url = (
        f"https://noaa-hrrr-bdp-pds.s3.amazonaws.com/"
        f"hrrr.{ymd}/conus/{fname}"
    )

    out = os.path.join(cache, f"{ymd}.{fname}")

    if not os.path.isfile(out):
        logging.info(f"Downloading {url}")
        os.system(f"wget -q -O {out} {url}")
    else:
        logging.info(f"{fname} already downloaded")

    return out


def _wind_dir_from_uv(u, v):
    return (np.degrees(np.arctan2(-u, -v)) + 360.0) % 360.0

def _nearest_ij(lat2d, lon2d, lat, lon):
    lon2 = ((lon2d + 180) % 360) - 180
    lon1 = ((lon + 180) % 360) - 180
    d2 = (lat2d - lat)**2 + (lon2 - lon1)**2
    j, i = np.unravel_index(np.argmin(d2), d2.shape)
    return int(j), int(i)

def _open_isobaric(grib_path, shortName, stepType="instant"):
    return xr.open_dataset(
        grib_path,
        engine="cfgrib",
        backend_kwargs={
            "filter_by_keys": {
                "typeOfLevel": "isobaricInhPa",
                "stepType": stepType,
                "shortName": shortName,
            }
        },
    )

def get_hrrr_wind_agl(lat, lon, date_time, z_agl=400, layer=None, cache="./../dat/_hrrr/"):
    """
    HRRR wind at target height above ground using wrfprs isobaric levels.

    Parameters
    ----------
    z_agl : float
        Target height AGL in meters (e.g., 300, 400, 500).
    layer : tuple[float,float] or None
        If provided, compute layer-mean wind between (z0, z1) meters AGL.
        Example: layer=(300, 500)

    Returns
    -------
    dict with speed_ms, dir_from_deg, u_ms, v_ms, z_used_m,
    plus some debug fields.
    """
    # You want pressure levels here:
    grib_path = download_hrrr(date_time, product="wrfprs", forecast_hour=0, cache=cache)

    # Winds on isobaric levels
    ds_u = _open_isobaric(grib_path, "u")
    ds_v = _open_isobaric(grib_path, "v")

    # Geopotential height on isobaric levels is usually shortName 'gh' in NCEP GRIB
    # If this fails, we can swap to 'z' depending on your file.
    ds_gh = _open_isobaric(grib_path, "gh")

    lat2d = ds_u["latitude"].values
    lon2d = ds_u["longitude"].values
    j, i = _nearest_ij(lat2d, lon2d, lat, lon)

    p = ds_u["isobaricInhPa"].values.astype(float)  # hPa levels

    u_prof = ds_u["u"].isel(y=j, x=i).values.astype(float)
    v_prof = ds_v["v"].isel(y=j, x=i).values.astype(float)
    z_msl = ds_gh["gh"].isel(y=j, x=i).values.astype(float)  # meters (MSL)

    # Need terrain/surface height (MSL) to convert MSL -> AGL.
    # Many HRRR files include "orog" (terrain height). If not, we can approximate
    # terrain as the minimum z_msl in the column (near-surface isobaric level),
    # but better is to open a surface/terrain field.
    terrain_msl = None
    try:
        ds_orog = xr.open_dataset(
            grib_path,
            engine="cfgrib",
            backend_kwargs={"filter_by_keys": {"shortName": "orog", "stepType": "instant"}},
        )
        terrain_msl = float(ds_orog["orog"].isel(y=j, x=i).values)
    except Exception:
        # fallback: crude terrain estimate from the lowest level height
        # (works OK-ish if the lowest isobaric level is close to the surface)
        terrain_msl = float(np.nanmin(z_msl))

    z_agl_levels = z_msl - terrain_msl  # meters AGL for each pressure level

    # Sort by increasing AGL height (important for interpolation)
    order = np.argsort(z_agl_levels)
    z_agl_levels = z_agl_levels[order]
    u_prof = u_prof[order]
    v_prof = v_prof[order]
    p_sorted = p[order]

    def interp_uv_at(z_target):
        # nearest fallback if outside
        if z_target <= z_agl_levels.min():
            k = int(np.argmin(np.abs(z_agl_levels - z_target)))
            return float(u_prof[k]), float(v_prof[k]), float(z_agl_levels[k])
        if z_target >= z_agl_levels.max():
            k = int(np.argmin(np.abs(z_agl_levels - z_target)))
            return float(u_prof[k]), float(v_prof[k]), float(z_agl_levels[k])
        u_t = float(np.interp(z_target, z_agl_levels, u_prof))
        v_t = float(np.interp(z_target, z_agl_levels, v_prof))
        return u_t, v_t, float(z_target)

    if layer is not None:
        z0, z1 = layer
        # sample a few points and average u/v (simple, robust)
        zs = np.linspace(z0, z1, 9)
        us, vs = [], []
        for zz in zs:
            uu, vv, _ = interp_uv_at(zz)
            us.append(uu); vs.append(vv)
        u = float(np.mean(us))
        v = float(np.mean(vs))
        z_used = float(0.5 * (z0 + z1))
    else:
        u, v, z_used = interp_uv_at(z_agl)

    speed = float(np.hypot(u, v))
    wdir = float(_wind_dir_from_uv(u, v))

    return {
        "speed_ms": speed,
        "dir_from_deg": wdir,
        "u_ms": u,
        "v_ms": v,
        "z_used_m_agl": z_used,
        "terrain_m_msl": terrain_msl,
        "levels_z_agl_m": z_agl_levels.tolist(),
        "levels_p_hpa": p_sorted.tolist(),
        "grib_path": grib_path,
    }

def open_hrrr_10m_group(grib_path):
    groups = cfgrib.open_datasets(grib_path, indexpath="")
    for ds in groups:
        if "u10" in ds.data_vars and "v10" in ds.data_vars:
            return ds
    raise RuntimeError("No HRRR 10 m wind group found")

def get_hrrr_wind_10m(lat, lon, date_time, cache="./../dat/_hrrr/"):
    """
    Download HRRR (wrfsfc f00) and return 10m wind at nearest grid point.

    Returns dict:
      u10_ms, v10_ms, speed_ms, dir_from_deg, grid_lat, grid_lon, grib_path
    """
    grib_path = download_hrrr(date_time, product="wrfsfc", forecast_hour=0, cache=cache)

    ds = open_hrrr_10m_group(grib_path)

    # lat/lon are coordinates in the selected group
    lat2d = ds.coords["latitude"].values
    lon2d = ds.coords["longitude"].values

    j, i = _nearest_ij(lat2d, lon2d, lat, lon)

    u10 = float(ds["u10"].isel(y=j, x=i).values)
    v10 = float(ds["v10"].isel(y=j, x=i).values)

    speed = float(np.hypot(u10, v10))
    wdir  = float(_wind_dir_from_uv(u10, v10))

    return {
        "u10_ms": u10,
        "v10_ms": v10,
        "speed_ms": speed,
        "dir_from_deg": wdir,  # meteorological: wind is FROM this direction
        "grid_lat": float(lat2d[j, i]),
        "grid_lon": float(lon2d[j, i]),
        "grib_path": grib_path,
    }



if __name__ == "__main__":
    usecache='/orcd/data/dvaron/001/kgauld/HRRR'
    # dt = datetime(2025, 1, 21, 18)
    dt = datetime(2024, 10, 1, 19)
    # w = get_hrrr_wind_agl(42.34648458544964, -71.11735491561392, dt, layer=(300, 500))
    w = get_hrrr_wind_agl(39.50764637214064, -112.57803027397549, dt, layer=(300,500), cache=usecache)
    print(w)
    print()
    # list_wind_messages()
    w = get_hrrr_wind_10m(39.50764637214064, -112.57803027397549, dt, cache=usecache)
    print(w)
    # get_hrrr_wind(42.34648458544964, -71.11735491561392, dt, cache=usecache)
    