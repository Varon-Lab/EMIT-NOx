#!/usr/bin/env python3
"""
Get MODIS MCD43A3 BRDF/Albedo averaged over an OMI-like pixel footprint.

- Uses earthaccess to find/download the correct MCD43A3.061 granule
- Averages white-sky albedo (WSA) for a chosen band over a polygon footprint

Dependencies:
    pip install earthaccess rasterio pyproj shapely

Set up Earthdata login before running.
"""

import os
from datetime import datetime, timedelta

import numpy as np
import earthaccess
import rasterio
from rasterio.features import geometry_mask
from shapely.geometry import Polygon, mapping
from pyproj import Transformer


# ----------------------------------------------------------------------
# Earthdata search + download
# ----------------------------------------------------------------------

def login_earthdata():
    """Login using ~/.netrc or env vars; falls back to interactive prompt."""
    earthaccess.login()


def find_mcd43a3_granule(lat, lon, date_str,
                         version="061",
                         max_results=10):
    """
    Search for cloud-hosted MCD43A3 granules that intersect a small
    bounding box around the point and the given date.

    Returns a single granule (first match).
    """
    target_date = datetime.strptime(date_str, "%Y-%m-%d").date()
    t0 = target_date.isoformat()
    t1 = (target_date + timedelta(days=1)).isoformat()

    # Small search box around the point
    bb = (lon - 0.5, lat - 0.5, lon + 0.5, lat + 0.5)

    results = earthaccess.search_data(
        short_name="MCD43A3",
        version=version,
        cloud_hosted=True,
        temporal=(t0, t1),
        bounding_box=bb,
        count=max_results,
    )

    if not results:
        raise RuntimeError(
            f"No MCD43A3.{version} granules found for {date_str} at "
            f"lat={lat}, lon={lon}"
        )

    return results[0]


def download_granule(granule, out_dir="data"):
    """
    Download the selected granule locally and return the local file path.
    """
    os.makedirs(out_dir, exist_ok=True)
    files = earthaccess.download([granule], out_dir)
    if not files:
        raise RuntimeError("Download failed – check Earthdata credentials.")
    return files[0]


# ----------------------------------------------------------------------
# Sampling over a footprint
# ----------------------------------------------------------------------

def sample_mcd43a3_footprint(hdf_path,
                             footprint_lons,
                             footprint_lats,
                             datafield="Albedo_WSA_Band1"):
    """
    Sample an MCD43A3 data field (e.g., white-sky albedo Band1)
    over an OMI pixel footprint defined by polygon vertices (lon/lat).

    Returns:
        mean_val : float (NaN if everything is fill)
        meta     : SDS-level metadata dict
        n_valid  : number of contributing MCD43 pixels
    """
    if len(footprint_lons) != len(footprint_lats):
        raise ValueError("footprint_lons and footprint_lats must have same length")

    # Ensure polygon is closed
    if footprint_lons[0] != footprint_lons[-1] or footprint_lats[0] != footprint_lats[-1]:
        footprint_lons = list(footprint_lons) + [footprint_lons[0]]
        footprint_lats = list(footprint_lats) + [footprint_lats[0]]

    poly_ll = Polygon(zip(footprint_lons, footprint_lats))

    grid_name = "MOD_Grid_BRDF"
    subdataset = f'HDF4_EOS:EOS_GRID:"{hdf_path}":{grid_name}:{datafield}'

    with rasterio.open(subdataset) as ds:
        # Reproject polygon from WGS84 to MODIS sinusoidal
        transformer = Transformer.from_crs("EPSG:4326", ds.crs, always_xy=True)
        xs, ys = transformer.transform(*poly_ll.exterior.xy)
        poly_proj = Polygon(zip(xs, ys))

        # Build mask (True inside footprint)
        geom = [mapping(poly_proj)]
        mask = geometry_mask(
            geometries=geom,
            out_shape=(ds.height, ds.width),
            transform=ds.transform,
            invert=True,  # True for pixels inside the geometry
        )

        arr = ds.read(1)  # 2D
        meta = ds.tags()

        fill = float(meta.get("_FillValue", -28672))
        scale = float(meta.get("scale_factor", 1.0))
        offset = float(meta.get("add_offset", 0.0))

        # Restrict to pixels inside footprint
        vals = arr[mask]

        # Filter out fill
        valid = vals != fill
        if not np.any(valid):
            return np.nan, meta, 0

        raw_valid = vals[valid]
        # Apply scale/offset
        albedo_vals = scale * (raw_valid - offset)

        mean_val = float(np.nanmean(albedo_vals))
        n_valid = int(valid.sum())

    return mean_val, meta, n_valid


def get_mcd43_albedo_for_footprint(center_lat,
                                   center_lon,
                                   date_str,
                                   footprint_lons,
                                   footprint_lats,
                                   band="Band1",
                                   sky="WSA",
                                   out_dir="data"):
    """
    High-level convenience function.

    - Finds & downloads matching MCD43A3 granule near center point/date
    - Averages albedo over the given footprint polygon

    sky  : 'WSA' (white-sky) or 'BSA' (black-sky)
    band : 'Band1', 'Band2', ..., 'Band7', 'Visible', 'NIR', 'Shortwave', etc.
    """
    if sky not in ("WSA", "BSA"):
        raise ValueError("sky must be 'WSA' or 'BSA'")

    datafield = f"Albedo_{sky}_{band}"

    login_earthdata()
    granule = find_mcd43a3_granule(center_lat, center_lon, date_str)
    local_path = download_granule(granule, out_dir)

    mean_val, meta, n_valid = sample_mcd43a3_footprint(
        local_path,
        footprint_lons,
        footprint_lats,
        datafield=datafield,
    )

    return mean_val, meta, n_valid, os.path.basename(local_path)


# ----------------------------------------------------------------------
# Example usage
# ----------------------------------------------------------------------

if __name__ == "__main__":
    # Example: one OMI pixel footprint
    # Replace these with your actual OMI pixel polygon vertices
    footprint_lons = [-105.2, -104.8, -104.7, -105.1]  # arbitrary example
    footprint_lats = [  39.9,   40.0,   40.3,   40.2]

    # Rough center (for search only)
    center_lon = np.mean(footprint_lons)
    center_lat = np.mean(footprint_lats)

    DATE = "2020-07-15"

    mean_alb, meta, n_valid, fname = get_mcd43_albedo_for_footprint(
        center_lat=center_lat,
        center_lon=center_lon,
        date_str=DATE,
        footprint_lons=footprint_lons,
        footprint_lats=footprint_lats,
        band="Band1",   # ignore wavelength/LUT nuance for now
        sky="WSA",      # white-sky albedo for Lambertian AMF LUT
        out_dir="modis_brdf",
    )

    print(f"Granule file      : {fname}")
    print(f"Center            : lat={center_lat:.4f}, lon={center_lon:.4f}")
    print(f"Date              : {DATE}")
    print(f"Field             : {meta.get('long_name', 'Albedo_WSA_Band1')}")
    print(f"Units             : {meta.get('units', '1')}")
    print(f"Mean albedo       : {mean_alb!r}")
    print(f"# valid MCD43 px  : {n_valid}")
