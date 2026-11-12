import earthaccess
import os
import warnings
import csv
import numpy as np
import math
import xarray as xr
import holoviews as hv
import hvplot.xarray
import netCDF4 as nc
#import cv2 as cv

import glob
import sys
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
from scipy import linalg
from scipy.ndimage import gaussian_filter, binary_erosion, binary_dilation
from skimage.transform import hough_line, hough_line_peaks

from config import CONFIG, LOCS
from REFERENCE_PLANTS import REFERENCE_PLANTS

sys.path.append('../EMIT-Data-Resources/python/modules/')
from emit_tools import emit_xarray, ortho_xr

RING = None
wav_min = 403#417#405#418
wav_max = 493#465#492
polydeg = 3

####################################################################################################
############################## NOX CROSS SECTION ###################################################
####################################################################################################
def gaussian_kernel(x, mu, fwhm_nm):
    """
    Create normalized Gaussian kernel with standard deviation sigma (nm) based on sigma = FWHM/sqrt(2log2)
    """
    # Define standard deviation sigma
    sigma = fwhm_nm / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    # Create kernel
    k = np.exp(-0.5 * ((x - mu) / sigma)**2)
    # Normalize kernel to area = 1
    k /= np.trapz(k, x, dx=0.01)
    return k

def convolve_to_sensor_grid(xs_wav_nm, xs, sensor_wav_nm, sensor_fwhm_nm):
    """
    Convolve high-res cross-section xs_sigma(xs_wav_nm) to sensor grid using per-band Gaussian SRF.
    Returns array of length len(sensor_wav_nm).
    """
    out = np.zeros_like(sensor_wav_nm, dtype=float)
    for i, (mu, fwhm) in enumerate(zip(sensor_wav_nm, sensor_fwhm_nm)):
        # Create normalized Gaussian kernel
        k = gaussian_kernel(xs_wav_nm, mu, fwhm)
        # Apply kernel to absorption cross-section spectrum
        out[i] = np.trapz(k * xs, xs_wav_nm, dx=0.01)
    return out

def build_design_matrix(emit, xs_conv, include_ring=False, include_inverse_spectrum=False, jj=None, ii=None, fit_shift=False, fit_stretch=False, poly_degree=polydeg):
    wav = emit['wavelengths']/1e3
    # Stack species columns
    cols = []
    names = []
    
    cols.append(xs_conv*1e19)
    names.append("NO2")

    if include_ring:
        cols.append(RING if RING is not None else np.zeros_like(wav))
        names.append('RING')

    # Polynomial baseline
    Xp = np.vstack([wav**i for i in range(poly_degree+1)]).T  # (nband, poly_degree+1)
    for i in range(poly_degree+1):
        cols.append(Xp[:, i])
        names.append(f'poly_{i}')

    if include_inverse_spectrum:
        cols.append(1/emit["radiance"][:,jj,ii])
        names.append("inverse_spectrum")

    A = np.vstack(cols).T

    return A, names

def get_NOX_cross_sec(emit_ds):
    # Get NO2 absorption cross-sections @ 220K from Vandaele et al. (1998)
    # Following TROPOMI NO2 ATBD: https://sentinel.esa.int/documents/247904/2476257/sentinel-5p-tropomi-atbd-no2-data-products
    # Data reference: http://spectrolab.aeronomie.be/no2.htm
    no2_cross_sections = pd.read_csv(
        CONFIG['NOX_CSEC'],
        sep=" ",
        usecols=[3, 4],
        names=["vacuum_wavenumber_cm-1", "cross_section_cm^2/molecule_@220K"],
    )

    # add wavelength column (nm)
    no2_cross_sections["vacuum_wavelength_nm"] = 1e7 / no2_cross_sections["vacuum_wavenumber_cm-1"].values
    
    emit_spec_wlen = emit_ds["wavelengths"].to_numpy().astype(np.float64)
    emit_spec_fwhm = emit_ds["fwhm"].to_numpy().astype(np.float64)
    
    no2_cross_sections_conv = convolve_to_sensor_grid(no2_cross_sections["vacuum_wavelength_nm"], 
                                                      no2_cross_sections["cross_section_cm^2/molecule_@220K"],
                                                      emit_spec_wlen,
                                                      emit_spec_fwhm)
    
    # Select bands and convolved cross-sections in fitting window
    window_sel = (emit_spec_wlen >= wav_min) & (emit_spec_wlen <= wav_max)
    emit_vnir_spectral_bands_fitting_window = emit_spec_wlen[window_sel].copy()
    no2_cross_sections_conv_fitting_window = no2_cross_sections_conv[window_sel].copy()

    return no2_cross_sections_conv_fitting_window, emit_vnir_spectral_bands_fitting_window

####################################################################################################
############################## DOAS FIT ############################################################
####################################################################################################
def doas_fit_pixel(I_ref, I_meas, A, weight=None):
    # Differential optical depth
    D = np.log(I_ref / I_meas)
    if weight is None:
        W = np.eye(len(D))
    else:
        w = np.clip(weight, 1e-12, None)
        W = np.diag(w)
    # Solve normal equations (A^T W A) x = A^T W D
    AtW = A.T @ W
    N = AtW @ A
    y = AtW @ D
    x = linalg.lstsq(N, y)[0]
    # Residuals & covariance estimate
    D_fit = A @ x
    r = D - D_fit
    dof = max(len(D) - A.shape[1], 1)
    s2 = (r @ r) / dof
    cov = s2 * linalg.pinv(N)
    return x, cov, r, s2

def precompute_column_stats(radiance_yxn, eligible_mask=None):
    """
    radiance_yxn: (ny, nx, nband)
    eligible_mask: (ny, nx) True => include; default = finite across all bands
    Returns dict with per-column sums (nband, nx) and counts (nx,).
    """
    if eligible_mask is None:
        eligible_mask = np.isfinite(radiance_yxn).all(axis=2)  # (ny, nx)

    m = eligible_mask[..., None]                              # (ny, nx, 1)
    sum_per_col = np.nansum(radiance_yxn * m, axis=0).T            # (nband, nx)
    cnt_per_col = eligible_mask.sum(axis=0).astype(np.int32)  # (nx,)

    return {
        "sum_per_col":   sum_per_col,
        "cnt_per_col":   cnt_per_col,
        "eligible_mask": eligible_mask
    }

def column_ref_for_pixel(stats, radiance, j, i, exclude_self=True):
    """
    Mean spectrum for column i (optionally excluding pixel (j,i) if it was eligible).
    """
    sum_c = stats["sum_per_col"][:, i]
    cnt_c = int(stats["cnt_per_col"][i])

    if exclude_self and stats["eligible_mask"][j, i]:
        return (sum_c - radiance[j, i, :]) / max(cnt_c - 1, 1)
    else:
        return sum_c / max(cnt_c, 1)

def run_doas_scene_vertical_striping(emit, A, names, plume_mask=None, userad=False):
    if not userad:
        rad = emit['radiance'].to_numpy()
    else:
        rad = emit

    idxs = np.where(~np.isnan(rad[:,:,0]))
    npoints = len(idxs[0])
    ny, nx, nband = rad.shape

    no2_cols = [i for i, n in enumerate(names) if n.startswith('NO2')]
    stats = precompute_column_stats(rad, eligible_mask=plume_mask)
    
    dscd = np.full((ny, nx), np.nan, dtype=float)
    dscd_err = np.full((ny, nx), np.nan, dtype=float)
    rms = np.full((ny, nx), np.nan, dtype=float)

    for p_idx in range(npoints):
        j = idxs[0][p_idx]
        i = idxs[1][p_idx]

        row = rad[j, :, :]
        I_meas = row[i, :]

        ref_spec = column_ref_for_pixel(stats, rad, j, i)
        
        if not np.isfinite(I_meas).any() or np.isnan(I_meas).any():
            continue
        
        # try:
        x, cov, r, s2 = doas_fit_pixel(ref_spec, I_meas, A)
        # Sum NO2 components if multiple temps are included
        dscd_val = np.nansum([x[k] for k in no2_cols])
        # Error as sqrt of sum of variances (approx), ignoring covariance between NO2 temps
        dscd_var = np.nansum([cov[k, k] for k in no2_cols])
        dscd[j, i] = dscd_val
        dscd_err[j, i] = np.sqrt(max(dscd_var, 0))
        rms[j, i] = np.sqrt(np.mean(r**2))
        # except Exception as e:
            # print(e)
            # continue

    out = {
        "dSCD": dscd,
        "dSCD_err": dscd_err,
        "rms": rms,
        "names": names
    }
    return out

def get_plume_mask(DOAS0):
    # Threshold mask
    threshold = np.nanpercentile(DOAS0['dSCD'], 95)
    plume_mask = (DOAS0['dSCD'] > threshold).astype(np.uint8)
    
    plume_mask = binary_dilation(plume_mask, iterations=2)
    plume_mask = binary_erosion(plume_mask, iterations=4)
    plume_mask = binary_dilation(plume_mask, iterations=2)
    
    return 1-plume_mask.astype(np.uint8)

def crop_about_loc(ds, clat, clon, km_boundary=None, pix_boundary=None):
    if km_boundary is None and pix_boundary is None:
        return np.ones_like(ds['radiance'])
    
    lat, lon = ds['lat'], ds['lon']
    
    if km_boundary is not None:
        dlat = (km_boundary/2)/111
        dlon = (km_boundary/2)/(111*np.cos(np.radians(clat)))

        lat_min = clat - dlat
        lat_max = clat + dlat
        lon_min = clon - dlon
        lon_max = clon + dlon

        mask = (
            (lat >= lat_min) & (lat <= lat_max) &
            (lon >= lon_min) & (lon <= lon_max)
        )
        
        ys, xs = np.where(mask)
        if len(ys) == 0:
            raise ValueError("No pixels found inside requested box!")

        y0, y1 = ys.min(), ys.max()
        x0, x1 = xs.min(), xs.max()
    elif pix_boundary is not None:
        plat, plot = np.argmin(np.abs(lat-clat)), np.argmin(np.abs(lon-clon))
        
        y0, y1 = plat - pix_boundary//2, plat + pix_boundary//2
        x0, x1 = plon - pix_boundary//2, plon + pix_boundary//2

    Ny, Nx = ds["radiance"].shape[:2]
        
    mask = np.zeros((Ny, Nx), dtype=bool)
    mask[y0:y1+1, x0:x1+1] = True

    # Expand to (Ny, Nx, Nlam)
    mask3d = mask[..., None]  # shape (Ny, Nx, 1)
    mask3d = np.broadcast_to(mask3d, ds["radiance"].shape)
    
    ds["radiance"] = ds["radiance"].where(mask3d)
    # ds["radiance"] = ds["radiance"].where(mask)

    return mask

def get_envi_mask(fn):
    mask_fn = fn.replace("L1B_RAD", "L2A_MASK")
    mask_ds = emit_xarray(mask_fn, ortho=False)
    cloud_mask = mask_ds['mask'].values[...,0]
    cloud_mask = binary_dilation(cloud_mask, iterations=2)
    
    cirrus_mask = mask_ds['mask'].values[...,1]
    water_mask = mask_ds['mask'].values[...,2]
    sc_mask = mask_ds['mask'].values[...,3]
    
    agg_mask = np.clip(cloud_mask+cirrus_mask+water_mask+sc_mask, 0, 1)
    del mask_ds
    return 1-agg_mask.astype(np.uint8)


# Row-wise destriping of a retrieved dSCD field (optional post-processing step, run_retrieval(..., destripe=True)).
# Copied verbatim from noisefix.ipynb, the step that produced results6_destriped/Retrievals for the paper: for each
# along-track line, subtract the median of that line's "background" pixels (below the scene's 90th percentile, not NaN);
# lines with 10 or fewer background pixels are left unchanged.
def get_destriped_field(field):
    mask = (field > np.nanpercentile(field, 90)).astype(np.uint8)
    mask2 = np.isnan(field)
    mask = np.clip(mask+mask2, 0, 1).astype(np.uint8)

    field_destriped = np.zeros_like(field)
    for row in range(len(field)):
        bg_pixels = field[row, mask[row, :] == 0]
        if len(bg_pixels) > 10:
            field_destriped[row, :] = field[row, :] - np.nanmedian(bg_pixels)
        else:
            field_destriped[row, :] = field[row, :]
    return field_destriped


def run_retrieval(fn, clat=None, clon=None, km_boundary=None, pix_boundary=None, include_mask=False, bin_size=1, destripe=False):
    print(f"Starting {fn}...")
    ds = emit_xarray(fn)
    mask = crop_about_loc(ds, clat=clat, clon=clon, km_boundary=km_boundary, pix_boundary=pix_boundary)
    nox_window, desired_bands = get_NOX_cross_sec(ds)

    emit_windowed = ds.sel(wavelengths=desired_bands, method='nearest')
    emit_windowed = emit_windowed.coarsen(downtrack=bin_size, crosstrack=bin_size, boundary="trim").mean()

    A, names = build_design_matrix(
        emit_windowed, nox_window,
    )
    print("NOX cross section created! Computing DOAS0...")
    if include_mask:
        initial_mask = get_envi_mask(fn)
    else:
        initial_mask = None
    
    DOAS0 = run_doas_scene_vertical_striping(emit_windowed, A, names, plume_mask=initial_mask)
    print("Found DOAS0! Computing plume mask...")
    
    plume_mask = get_plume_mask(DOAS0)
    if include_mask:
        plume_mask = 1-np.clip(2-plume_mask-initial_mask, 0, 1)
    
    print("Found plume mask! Computing DOAS...")
    
    DOAS = run_doas_scene_vertical_striping(emit_windowed, A, names, plume_mask=plume_mask)
    print("DOAS Retrieval done!")

    lat, lon = ds['lat'].copy(), ds['lon'].copy()
    del ds

    if include_mask:
        doas_out = np.where(1-initial_mask, np.nan, DOAS['dSCD'])
    else:
        doas_out = DOAS['dSCD']
    if destripe:
        doas_out = get_destriped_field(doas_out)
    return doas_out, lat, lon, mask, initial_mask, plume_mask


if __name__ == "__main__":
    # imfns = glob.glob(f"{CONFIG['data_folder']}/SPACEX_PAD/*RAD*") ## CHANGE THIS
    # fn = imfns[1]
    # loc_name = "New_Madrid_Power_Plant"#"RIYADH_PLANT_9"
    # granule_name = "EMIT_L1B_RAD_001_20250401T172306_2509111_025.nc" #"EMIT_L1B_RAD_001_20250613T114019_2516407_025.nc"

    loc_name='QTS_Richmond'
    granule_name='EMIT_L1B_RAD_001_20260604T145450_2615510_017'
    
    loc_data = LOCS[loc_name]
    
    fn = f"{CONFIG['data_folder']}/{loc_name}/{granule_name}.nc"

    km_boundary=20
    
    result_dSCD = run_retrieval(fn, clat=loc_data['LAT'], clon=loc_data['LON'], km_boundary=km_boundary, include_mask=True)


    print(result_dSCD)
    save_path = f"{CONFIG['results_folder']}/{CONFIG['retr_subdir']}/{loc_name}"
    os.makedirs(save_path, exist_ok=True)
    np.save(f"{save_path}/{granule_name}.npy", result_dSCD)

    print(f"Saved product to {save_path}/dSCD_EMIT_L1B_RAD_001_{granule_name}.npy !")
    # plt.figure()
    # plt.imshow(ortho_ds.sel(wavelengths=1500, method='nearest')['radiance'], cmap='gray')
    # plt.imshow(np.array(ortho_ds['dSCD'])[:,:,0]*1e19, alpha=0.5, vmin=-2e17, vmax=2e17, origin='upper', cmap="RdBu_r", aspect='auto')
    # plt.show()


    
    
