#!/usr/bin/env python3
"""Compare FLS vs MBES PLY point clouds: side-by-side visualization with error metrics."""

import argparse
import glob
import os
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def load_ply(path):
    pcd = o3d.io.read_point_cloud(path)
    return np.asarray(pcd.points)


def compute_error(fls_pts, mbes_pts, bin_size=0.5):
    """Roman & Singh (2006) bin-based map-to-map error."""
    fls_min = np.min(fls_pts[:, :2], axis=0)
    fls_max = np.max(fls_pts[:, :2], axis=0)
    mbes_min = np.min(mbes_pts[:, :2], axis=0)
    mbes_max = np.max(mbes_pts[:, :2], axis=0)

    xy_min = np.maximum(fls_min, mbes_min)
    xy_max = np.minimum(fls_max, mbes_max)

    if np.any(xy_min >= xy_max):
        return None

    nx = int(np.ceil((xy_max[0] - xy_min[0]) / bin_size))
    ny = int(np.ceil((xy_max[1] - xy_min[1]) / bin_size))
    if nx < 1 or ny < 1:
        return None

    fls_binned = {}
    for pt in fls_pts:
        bx = int((pt[0] - xy_min[0]) / bin_size)
        by = int((pt[1] - xy_min[1]) / bin_size)
        if 0 <= bx < nx and 0 <= by < ny:
            fls_binned.setdefault((bx, by), []).append(pt[2])

    mbes_binned = {}
    for pt in mbes_pts:
        bx = int((pt[0] - xy_min[0]) / bin_size)
        by = int((pt[1] - xy_min[1]) / bin_size)
        if 0 <= bx < nx and 0 <= by < ny:
            mbes_binned.setdefault((bx, by), []).append(pt[2])

    errors = []
    for key in fls_binned:
        if key in mbes_binned:
            z_error = abs(np.mean(fls_binned[key]) - np.mean(mbes_binned[key]))
            errors.append(z_error)

    if not errors:
        return None

    errors = np.array(errors)
    return {
        'mean': np.mean(errors),
        'std': np.std(errors),
        'median': np.median(errors),
        'min': np.min(errors),
        'max': np.max(errors),
        'rmse': np.sqrt(np.mean(errors**2)),
        'bins': len(errors),
        'total_bins': nx * ny,
        'coverage': len(errors) / (nx * ny),
    }


def plot_pair(fls_pts, mbes_pts, name, metrics, subsample=1):
    """Plot FLS and MBES side by side with metrics annotation."""
    fig = plt.figure(figsize=(16, 6))

    fls_sub = fls_pts[::subsample]
    mbes_sub = mbes_pts[::subsample]

    # FLS
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    ax1.scatter(fls_sub[:, 0], fls_sub[:, 1], fls_sub[:, 2], s=0.3, c=fls_sub[:, 2], cmap='viridis')
    ax1.set_title(f'FLS — {name}')
    ax1.set_xlabel('X'); ax1.set_ylabel('Y'); ax1.set_zlabel('Z')

    # MBES
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    ax2.scatter(mbes_sub[:, 0], mbes_sub[:, 1], mbes_sub[:, 2], s=0.3, c=mbes_sub[:, 2], cmap='viridis')
    ax2.set_title(f'MBES — {name}')
    ax2.set_xlabel('X'); ax2.set_ylabel('Y'); ax2.set_zlabel('Z')

    # Match axes limits
    all_pts = np.vstack([fls_pts, mbes_pts])
    for ax in [ax1, ax2]:
        ax.set_xlim(all_pts[:, 0].min(), all_pts[:, 0].max())
        ax.set_ylim(all_pts[:, 1].min(), all_pts[:, 1].max())
        ax.set_zlim(all_pts[:, 2].min(), all_pts[:, 2].max())

    # Metrics text
    if metrics:
        text = (f"Mean:   {metrics['mean']:.4f} m\n"
                f"Median: {metrics['median']:.4f} m\n"
                f"RMSE:   {metrics['rmse']:.4f} m\n"
                f"Std:    {metrics['std']:.4f} m\n"
                f"Min:    {metrics['min']:.4f} m\n"
                f"Max:    {metrics['max']:.4f} m\n"
                f"Bins:   {metrics['bins']}/{metrics['total_bins']} ({metrics['coverage']*100:.1f}%)")
    else:
        text = "No overlapping region"

    fig.text(0.5, 0.02, text, ha='center', va='bottom', fontsize=10, family='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    fig.suptitle(f'{name}  —  Map-to-Map Error (Roman & Singh)', fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0.12, 1, 0.95])
    return fig


def main():
    parser = argparse.ArgumentParser(description='Compare FLS vs MBES PLY files with visualization')
    parser.add_argument('--dir', type=str,
                        default=os.path.expanduser('~/Documents/CloudCompare'),
                        help='Directory containing PLY files')
    parser.add_argument('--bin-size', type=float, default=0.5)
    parser.add_argument('--subsample', type=int, default=1,
                        help='Point subsampling factor for plotting')
    parser.add_argument('--save', type=str, default=None,
                        help='Save directory for figures (default: show interactively)')
    parser.add_argument('--fls', type=str, help='Single FLS PLY file')
    parser.add_argument('--mbes', type=str, help='Single MBES PLY file')
    args = parser.parse_args()

    # Build pairs
    if args.fls and args.mbes:
        pairs = [(args.fls, args.mbes, os.path.basename(args.fls).replace('fls-', '').replace('.ply', ''))]
    else:
        fls_files = sorted(glob.glob(os.path.join(args.dir, 'fls-*.ply')))
        pairs = []
        for fls_path in fls_files:
            name = os.path.basename(fls_path).replace('fls-', '')
            mbes_path = os.path.join(args.dir, f'mbes-{name}')
            if os.path.exists(mbes_path):
                pairs.append((fls_path, mbes_path, name.replace('.ply', '')))

    if not pairs:
        print("No matching FLS/MBES pairs found!")
        return

    if args.save:
        os.makedirs(args.save, exist_ok=True)

    for fls_path, mbes_path, name in pairs:
        print(f"Processing {name}...")
        fls_pts = load_ply(fls_path)
        mbes_pts = load_ply(mbes_path)

        if len(fls_pts) < 10 or len(mbes_pts) < 10:
            print(f"  Skipping — not enough points (FLS={len(fls_pts)}, MBES={len(mbes_pts)})")
            continue

        metrics = compute_error(fls_pts, mbes_pts, args.bin_size)
        fig = plot_pair(fls_pts, mbes_pts, name, metrics, args.subsample)

        if args.save:
            out_path = os.path.join(args.save, f'{name}_comparison.png')
            fig.savefig(out_path, dpi=150, bbox_inches='tight')
            print(f"  Saved: {out_path}")
            plt.close(fig)

    if not args.save:
        plt.show()


if __name__ == '__main__':
    main()
