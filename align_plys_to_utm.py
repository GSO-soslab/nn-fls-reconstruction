#!/usr/bin/env python3
"""
Convert two PLY files from different local ENU frames to a common UTM frame and save aligned PLYs.

Each local ENU frame is defined by a GPS datum (lat, lon).
Points in ENU (x=East, y=North, z=Up) relative to datum are converted to UTM.

Usage:
    python3 align_plys_to_utm.py
"""

import numpy as np
from pathlib import Path

# ---- datums (from bag /gps/datum topics) ----
WAMV_DATUM_LAT  =  41.43958396
WAMV_DATUM_LON  = -71.42822211

ALPHA_DATUM_LAT =  41.44525823
ALPHA_DATUM_LON = -71.42515509

# ---- input/output files ----
WAMV_PLY   = "wamv_mbes.ply"
ALPHA_PLY  = "fls_output_field.ply"
OUT_WAMV   = "wamv_mbes_utm.ply"
OUT_ALPHA  = "fls_output_utm.ply"


# ---------------------------------------------------------------------------
# Minimal WGS84 -> UTM conversion (no external deps)
# ---------------------------------------------------------------------------

def latlon_to_utm(lat_deg, lon_deg):
    """Convert WGS84 lat/lon to UTM (easting, northing). Returns (e, n, zone)."""
    a  = 6378137.0
    f  = 1 / 298.257223563
    b  = a * (1 - f)
    e2 = 1 - (b/a)**2
    e  = np.sqrt(e2)

    lat = np.radians(lat_deg)
    lon = np.radians(lon_deg)

    zone = int((lon_deg + 180) / 6) + 1
    lon0 = np.radians((zone - 1) * 6 - 180 + 3)

    N = a / np.sqrt(1 - e2 * np.sin(lat)**2)
    T = np.tan(lat)**2
    C = e2 / (1 - e2) * np.cos(lat)**2
    A = np.cos(lat) * (lon - lon0)

    M = a * ((1 - e2/4 - 3*e2**2/64 - 5*e2**3/256) * lat
             - (3*e2/8 + 3*e2**2/32 + 45*e2**3/1024) * np.sin(2*lat)
             + (15*e2**2/256 + 45*e2**3/1024) * np.sin(4*lat)
             - (35*e2**3/3072) * np.sin(6*lat))

    k0 = 0.9996
    easting = k0 * N * (A + (1-T+C)*A**3/6 + (5-18*T+T**2+72*C-58*e2/(1-e2))*A**5/120) + 500000
    northing = k0 * (M + N*np.tan(lat)*(A**2/2 + (5-T+9*C+4*C**2)*A**4/24
                                         + (61-58*T+T**2+600*C-330*e2/(1-e2))*A**6/720))
    if lat_deg < 0:
        northing += 10000000

    return easting, northing, zone


def enu_to_utm(points_enu, datum_lat, datum_lon):
    """
    Convert points in local ENU (relative to datum) to UTM.
    points_enu: (N, 3) array of (east, north, up)
    Returns (N, 3) UTM (easting, northing, up).
    """
    datum_e, datum_n, zone = latlon_to_utm(datum_lat, datum_lon)
    utm = points_enu.copy()
    utm[:, 0] += datum_e   # east
    utm[:, 1] += datum_n   # north
    # z (up) stays the same
    return utm, zone


# ---------------------------------------------------------------------------
# PLY I/O
# ---------------------------------------------------------------------------

def read_ply(path):
    """Read ASCII PLY, return (N,3) points and (N,3) colors (0-255 uint8)."""
    path = Path(path)
    lines = path.read_text().splitlines()

    # Parse header
    i = 0
    n_verts = 0
    has_color = False
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith('element vertex'):
            n_verts = int(line.split()[-1])
        if 'red' in line:
            has_color = True
        if line == 'end_header':
            i += 1
            break
        i += 1

    points = np.zeros((n_verts, 3), dtype=np.float64)
    colors = np.full((n_verts, 3), 128, dtype=np.uint8)

    for j in range(n_verts):
        vals = lines[i + j].split()
        points[j] = [float(vals[0]), float(vals[1]), float(vals[2])]
        if has_color and len(vals) >= 6:
            colors[j] = [int(vals[3]), int(vals[4]), int(vals[5])]

    return points, colors


def write_ply(path, points, colors):
    path = Path(path)
    with open(path, 'w') as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for (x, y, z), (r, g, b) in zip(points, colors):
            f.write(f"{x:.4f} {y:.4f} {z:.4f} {r} {g} {b}\n")
    print(f"Saved {len(points)} points -> {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Reading PLYs...")
    wamv_pts,  wamv_col  = read_ply(WAMV_PLY)
    alpha_pts, alpha_col = read_ply(ALPHA_PLY)

    print(f"wamv:  {len(wamv_pts)} points")
    print(f"alpha: {len(alpha_pts)} points")

    print("Converting to UTM...")
    wamv_utm,  zone_w = enu_to_utm(wamv_pts,  WAMV_DATUM_LAT,  WAMV_DATUM_LON)
    alpha_utm, zone_a = enu_to_utm(alpha_pts, ALPHA_DATUM_LAT, ALPHA_DATUM_LON)

    print(f"wamv  UTM zone {zone_w}: E={wamv_utm[:,0].mean():.1f}, N={wamv_utm[:,1].mean():.1f}")
    print(f"alpha UTM zone {zone_a}: E={alpha_utm[:,0].mean():.1f}, N={alpha_utm[:,1].mean():.1f}")

    # Shift both to a common local origin (wamv centroid) for easier viewing
    origin = wamv_utm[:, :2].mean(axis=0)
    wamv_utm[:,  0] -= origin[0];  wamv_utm[:,  1] -= origin[1]
    alpha_utm[:, 0] -= origin[0];  alpha_utm[:, 1] -= origin[1]

    print(f"Common origin (UTM): E={origin[0]:.1f}, N={origin[1]:.1f}")
    print(f"wamv  after shift:  X={wamv_utm[:,0].min():.1f}..{wamv_utm[:,0].max():.1f}, "
          f"Y={wamv_utm[:,1].min():.1f}..{wamv_utm[:,1].max():.1f}")
    print(f"alpha after shift:  X={alpha_utm[:,0].min():.1f}..{alpha_utm[:,0].max():.1f}, "
          f"Y={alpha_utm[:,1].min():.1f}..{alpha_utm[:,1].max():.1f}")

    write_ply(OUT_WAMV,  wamv_utm,  wamv_col)
    write_ply(OUT_ALPHA, alpha_utm, alpha_col)
    print("Done. Load both in CloudCompare to compare.")


if __name__ == '__main__':
    main()
