from EMIT_NOX import run_retrieval

import argparse
import numpy as np
import glob
import os, sys

from config import CONFIG, LOCS
from REFERENCE_PLANTS import REFERENCE_PLANTS

sys.path.append('../datasets/')
from get_geosfp import get_geosfp_wind
from get_hrrr import get_hrrr_wind_10m, get_hrrr_wind_agl

sys.path.append('../EMIT-Data-Resources/python/modules/')
from emit_tools import emit_xarray

from datetime import datetime, timedelta, timezone
from scipy.ndimage import rotate

def max_center_crop_indices(ds, dSCD, clat, clon):
    """
    Returns:
      (y0, y1, x0, x1, y_c, x_c)
    where (y_c, x_c) is the pixel closest to (clat, clon),
    and [y0:y1+1, x0:x1+1] is the largest crop with that pixel at the center.
    """
    lat = ds["lat"].values   # (downtrack, crosstrack)
    lon = ds["lon"].values

    # distance in degrees; good enough for "closest pixel" selection
    # (if you want a bit better: multiply lon diff by cos(lat))
    dlat = lat - clat
    dlon = (lon - clon) * np.cos(np.deg2rad(clat))
    dist2 = dlat*dlat + dlon*dlon

    # ignore invalid geolocation pixels if present
    dist2 = np.where(np.isfinite(dist2), dist2, np.inf)

    y_c, x_c = np.unravel_index(np.argmin(dist2), dist2.shape)

    ny, nx = dSCD.shape  # should match (downtrack, crosstrack)

    # largest symmetric crop around (y_c, x_c)
    half_h = min(y_c, ny - 1 - y_c)
    half_w = min(x_c, nx - 1 - x_c)

    y0, y1 = y_c - half_h, y_c + half_h
    x0, x1 = x_c - half_w, x_c + half_w

    return y0, y1, x0, x1, y_c, x_c

def wind_to_rotation_deg_in_image(ds, yc, xc, wind_from_deg):
    """
    ds: xarray Dataset with lat/lon coords on (downtrack, crosstrack)
    (yc, xc): center pixel indices in (downtrack, crosstrack)
    wind_from_deg: meteorological direction wind is coming FROM (clockwise from North)
    
    Returns rot_deg suitable for scipy.ndimage.rotate(..., rot_deg, reshape=False)
    so that wind points to +x (right) in the rotated image.
    """
    lat = ds["lat"].values
    lon = ds["lon"].values

    # pick safe finite-difference neighbors
    ny, nx = lat.shape
    y0 = np.clip(yc-1, 0, ny-1); y1 = np.clip(yc+1, 0, ny-1)
    x0 = np.clip(xc-1, 0, nx-1); x1 = np.clip(xc+1, 0, nx-1)

    lat_c = lat[yc, xc]
    if not np.isfinite(lat_c):
        raise ValueError("Center pixel lat is not finite; choose another center or mask invalid geo.")

    # local meters-per-degree factors (good locally)
    m_per_deg_lat = 111_320.0
    m_per_deg_lon = 111_320.0 * np.cos(np.deg2rad(lat_c))

    # image basis vectors expressed in local East/North meters:
    # e_x: one step in crosstrack (to the right)
    dlon_dx = lon[yc, x1] - lon[yc, x0]
    dlat_dx = lat[yc, x1] - lat[yc, x0]
    e_x = np.array([dlon_dx * m_per_deg_lon, dlat_dx * m_per_deg_lat])  # [E, N]

    # e_y: one step in downtrack (downwards)
    dlon_dy = lon[y1, xc] - lon[y0, xc]
    dlat_dy = lat[y1, xc] - lat[y0, xc]
    e_y = np.array([dlon_dy * m_per_deg_lon, dlat_dy * m_per_deg_lat])  # [E, N]

    # Normalize (avoid degenerate cases)
    exn = np.linalg.norm(e_x); eyn = np.linalg.norm(e_y)
    if exn < 1e-12 or eyn < 1e-12:
        raise ValueError("Degenerate local geo gradients; can't define image axes here.")
    e_x /= exn
    e_y /= eyn

    # Wind "to" vector in local East/North
    wind_to = (wind_from_deg + 180.0) % 360.0
    # met convention: 0=N, 90=E
    w_EN = np.array([np.sin(np.deg2rad(wind_to)), np.cos(np.deg2rad(wind_to))])  # [E, N]

    # Project wind onto image axes => components in (x=crosstrack, y=downtrack)
    w_x = np.dot(w_EN, e_x)
    w_y = np.dot(w_EN, e_y)

    # Angle of wind in image coordinates, with +x right, +y down.
    # arctan2 gives angle CCW from +x if y is "up", but our y is "down".
    # So use arctan2(w_y, w_x) and interpret as screen coords.
    angle_deg = np.rad2deg(np.arctan2(w_y, w_x))  # wind pointing at this angle in image plane

    # We want wind to be angle 0 (pointing right). ndimage.rotate is CCW-positive,
    # so rotate by -angle_deg.
    rot_deg = -angle_deg
    return rot_deg, dict(wind_to=wind_to, w_x=w_x, w_y=w_y, angle_deg=angle_deg)

def rotate_nanaware(a, rot_deg, order=1):
    a = a.astype(np.float32)

    valid = np.isfinite(a).astype(np.float32)
    a0 = np.where(np.isfinite(a), a, 0.0).astype(np.float32)

    # rotate both with SAME settings
    a_rot = rotate(a0, rot_deg, reshape=False, order=order, mode="constant", cval=0.0)
    v_rot = rotate(valid, rot_deg, reshape=False, order=0,   mode="constant", cval=0.0)

    # normalize where we have support
    out = np.full_like(a_rot, np.nan, dtype=np.float32)
    m = v_rot > 0.5
    out[m] = a_rot[m] / v_rot[m]
    return out

def center_crop(a, out_h, out_w):
    """Center-crop 2D array a to (out_h, out_w). Returns cropped view."""
    h, w = a.shape
    cy, cx = h // 2, w // 2
    hh, hw = out_h // 2, out_w // 2
    return a[cy - hh: cy + hh + 1, cx - hw: cx + hw + 1]

def time_avg_retrieval(target_name, loc_source, samples, km_boundary=None, min_pix = 100):
    clat, clon = loc_source[target_name]['LAT'], loc_source[target_name]['LON']
    
    granules = [k.split('/')[-1][5:-4] for k in glob.glob(f"{CONFIG['results_folder']}/{CONFIG['retr_subdir']}/{target_name}/*_RAD_*")]
    
    rotated_dscds = []
    granule_status = []
    
    for granule in granules:
        print(f"Starting {len(rotated_dscds)+1}: {granule}")
        emit_radfn = f"{CONFIG['data_folder']}/{target_name}/{granule}.nc"
        retr_fn = f"{CONFIG['results_folder']}/{CONFIG['retr_subdir']}/{target_name}/dSCD_{granule}.npy"
    
        ds = emit_xarray(emit_radfn, ortho=False)
        dSCD = np.load(retr_fn)
        ys, xs = np.where(~np.isnan(dSCD))
        if len(ys) == 0:
            print(f"{granule} FAILED: No pixels found inside requested box!")
            granule_status.append('NO_PIX')
            continue
        
        y0, y1 = ys.min(), ys.max()
        x0, x1 = xs.min(), xs.max()
    
        y0, y1, x0, x1, yc, xc = max_center_crop_indices(ds, dSCD, clat, clon)
        ds_crop = ds.isel(downtrack=slice(y0, y1+1), crosstrack=slice(x0, x1+1))
        dSCD_crop = dSCD[y0:y1+1, x0:x1+1]
    
        obs_time = datetime.strptime(granule[-27:-12], "%Y%m%dT%H%M%S").replace(tzinfo=timezone.utc)
    
        geosfp_info = get_geosfp_wind(clat, clon, obs_time, cache=f'{CONFIG["geosfp"]}/')
        geosfp_dir, geosfp_sp = float(geosfp_info["DIR50"]), float(geosfp_info["U50"])
        
        hrrr_agl_info = get_hrrr_wind_agl(clat, clon, obs_time, layer=(200,600), cache=f'{CONFIG["hrrr"]}')
        hrrr_agl_dir, hrrr_agl_sp = float(hrrr_agl_info["dir_from_deg"]), float(hrrr_agl_info["speed_ms"])
        
        hrrr_10m_info = get_hrrr_wind_10m(clat, clon, obs_time, cache=f'{CONFIG["hrrr"]}')
        hrrr_10m_dir, hrrr_10m_sp = float(hrrr_10m_info["dir_from_deg"]), float(hrrr_10m_info["speed_ms"])
    
        avg_speed = (geosfp_sp+hrrr_agl_sp+hrrr_10m_sp)/3
        if avg_speed < 1.5:
            granule_status.append('LOW_WSP')
            print(f"{granule} FAILED: wind speed too low {avg_speed=}")
            continue
        
        rot_deg, dbg = wind_to_rotation_deg_in_image(ds, yc, xc, wind_from_deg=hrrr_agl_dir)  # or geosfp_dir
        dSCD_rot = rotate_nanaware(dSCD_crop, rot_deg, order=1)

        if (dSCD_rot.shape[0] < min_pix or dSCD_rot.shape[1] < min_pix):
            granule_status.append("TOO_SMALL")
            print(f"{granule} FAILED: area too small")
            continue
        
        print(f"{granule} SUCCESS! appending")
        rotated_dscds.append(dSCD_rot)
        granule_status.append("GOOD")
    
    H = int(np.min([a.shape[0] for a in rotated_dscds]))
    W = int(np.min([a.shape[1] for a in rotated_dscds]))
    
    # force odd so there is an exact center pixel
    H = H if (H % 2 == 1) else (H - 1)
    W = W if (W % 2 == 1) else (W - 1)
    
    print("stack size:", H, W)

    stack = np.stack(
        [center_crop(a, H, W) for a in rotated_dscds],
        axis=0
    )
    mean_stack = np.nanmean(stack, axis=0)

    save_dir = f"{CONFIG['results_folder']}/{CONFIG['tavg_subdir']}/{target_name}/"
    os.makedirs(save_dir, exist_ok=True)

    mean_fn = os.path.join(save_dir, f"{target_name}_mean_stack.npy")
    np.save(mean_fn, mean_stack)

    status_fn = os.path.join(save_dir, f"{target_name}_granule_status.txt")

    with open(status_fn, "w") as f:
        for granule, status in zip(granules, granule_status):
            f.write(f"{granule}\t{status}\n")
    

def singlepass_retrieval(target_name, loc_source, fn, save_file=True, km_boundary=None, skip_done=False, include_mask=False, destripe=False):
    clat, clon = loc_source[target_name]['LAT'], loc_source[target_name]['LON']
    
    granule_name = fn.split('/')[-1]
    save_path = f"{CONFIG['results_folder']}/{CONFIG['retr_subdir']}/{target_name}"
    result_savefn = f"{save_path}/dSCD_{granule_name.split('.')[0]}.npy"
    
    if skip_done and os.path.exists(result_savefn):
        print("already computed, skipping...")
        return
    result_dSCD, lat, lon, mask, envi_masks, plume_masks = run_retrieval(fn, clat=clat, clon=clon, km_boundary=km_boundary, include_mask=include_mask, destripe=destripe)
    
    if save_file:
        os.makedirs(save_path, exist_ok=True)
        np.save(result_savefn, result_dSCD)
        
        print(f"Saved product to {save_path}/dSCD_{granule_name.split('.')[0]}.npy !")

    return result_dSCD, lat, lon, mask, envi_masks, plume_masks

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--loc_name", type=str, required=True, help="Location name")
    parser.add_argument("--time_avg", action="store_true", help="Flag for plant mode")
    parser.add_argument("--skip_done", action="store_true", help="Flag to skip already processed imgs")
    parser.add_argument("--include_mask", action="store_true", help="Flag for environmental masks")
    parser.add_argument("--no_rerun", action="store_true", help="Flag for environmental masks")
    parser.add_argument("--destripe", action="store_true",
        help="Apply the row-wise destriping (EMIT_NOX.get_destriped_field, as in noisefix.ipynb) to each retrieved dSCD field before saving it. "
             "Saves under CONFIG['results_folder'], so point that at the *_destriped folder to keep raw and destriped retrievals apart.")

    parser.add_argument("--km_boundary", type=int, default=None,
        help="Optional kilometer bound (integer). Defaults to None if not provided.")
    args = parser.parse_args()
    
    target_name = args.loc_name
    km_boundary = args.km_boundary
    skip_done = args.skip_done
    include_mask = args.include_mask

    print(f"Running with {args=}")
    samples = glob.glob(f"{CONFIG['data_folder']}/{target_name}/*_RAD_*.nc")
    
    if args.time_avg:
        skip_done=True
    loc_src = REFERENCE_PLANTS if target_name in REFERENCE_PLANTS.keys() else LOCS
    if not args.no_rerun:
        for fn in samples:
            try:
                singlepass_retrieval(target_name, 
                                     loc_src, 
                                     fn, 
                                     km_boundary=km_boundary, 
                                     skip_done=skip_done, 
                                     include_mask=include_mask,
                                     destripe=args.destripe)
            except ValueError as e:
                print(f"[SKIP] {target_name} | {fn}")
                print(f"        {e}")
    
    if args.time_avg:
        time_avg_retrieval(target_name, REFERENCE_PLANTS, samples)
        
