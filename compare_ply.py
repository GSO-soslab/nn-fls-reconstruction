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


def compute_chamfer(fls_pts, mbes_pts, f_thresholds=(0.3, 0.3, 0.3)):
    """Chamfer Distance, Hausdorff, and F-Score (full 3D, point-to-point)."""
    fls_pcd = o3d.geometry.PointCloud()
    fls_pcd.points = o3d.utility.Vector3dVector(fls_pts)
    mbes_pcd = o3d.geometry.PointCloud()
    mbes_pcd.points = o3d.utility.Vector3dVector(mbes_pts)

    d_fls = np.asarray(fls_pcd.compute_point_cloud_distance(mbes_pcd))   # FLS→MBES
    d_mbes = np.asarray(mbes_pcd.compute_point_cloud_distance(fls_pcd))  # MBES→FLS

    f_scores = {}
    for tau in f_thresholds:
        p = float(np.mean(d_fls < tau))
        r = float(np.mean(d_mbes < tau))
        f_scores[tau] = {
            'precision': p,
            'recall': r,
            'f': 2 * p * r / (p + r) if (p + r) > 0 else 0.0,
        }

    # Inlier stats per threshold: how many points actually fell within tau
    inliers = {}
    for tau in f_thresholds:
        n_fls_in = int(np.sum(d_fls < tau))
        n_mbes_in = int(np.sum(d_mbes < tau))
        inliers[tau] = {
            'fls_inliers': n_fls_in,
            'fls_total': len(d_fls),
            'fls_pct': 100.0 * n_fls_in / len(d_fls),
            'mbes_inliers': n_mbes_in,
            'mbes_total': len(d_mbes),
            'mbes_pct': 100.0 * n_mbes_in / len(d_mbes),
        }

    return {
        'chamfer': float(np.mean(d_fls ** 2) + np.mean(d_mbes ** 2)),
        'hausdorff': float(max(np.max(d_fls), np.max(d_mbes))),
        'mean_fls2mbes': float(np.mean(d_fls)),
        'mean_mbes2fls': float(np.mean(d_mbes)),
        'f_scores': f_scores,
        'inliers': inliers,
    }


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


def plot_pair(fls_pts, mbes_pts, name, chamfer, subsample=1):
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

    # # Bin-based metrics (left)
    # if metrics:
    #     bin_text = (f"── Bin-based (Z only) ──\n"
    #                 f"Mean:   {metrics['mean']:.4f} m\n"
    #                 f"Median: {metrics['median']:.4f} m\n"
    #                 f"RMSE:   {metrics['rmse']:.4f} m\n"
    #                 f"Std:    {metrics['std']:.4f} m\n"
    #                 f"Bins:   {metrics['bins']}/{metrics['total_bins']} ({metrics['coverage']*100:.1f}%)")
    # else:
    #     bin_text = "── Bin-based (Z only) ──\nNo overlapping region"

    # fig.text(0.25, 0.02, bin_text, ha='center', va='bottom', fontsize=9, family='monospace',
    #          bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    # 3D metrics + inlier counts — centred below the plots
    # inlier_lines = ''.join(
    #     f"  τ={tau:.2f}m — FLS→MBES (noise):    {iv['fls_inliers']}/{iv['fls_total']} ({iv['fls_pct']:.1f}%)\n"
    #     f"           MBES→FLS (coverage): {iv['mbes_inliers']}/{iv['mbes_total']} ({iv['mbes_pct']:.1f}%)\n"
    #     for tau, iv in sorted(chamfer['inliers'].items())
    # )

    inlier_lines = ''.join(
    f"  τ={tau:.2f}m — MBES→FLS (coverage): {iv['mbes_inliers']}/{iv['mbes_total']} ({iv['mbes_pct']:.1f}%)\n"
    for tau, iv in sorted(chamfer['inliers'].items())
    )
    # guide = ("── Interpretation Guide ──\n"
    #          "High FLS→MBES % → many FLS points are noise / ghosts\n"
    #          "Low  FLS→MBES % → FLS reconstruction is clean\n"
    #          "High MBES→FLS % → FLS covered the real surface well  ✓\n"
    #          "Low  MBES→FLS % → FLS missed parts of the real surface ✗")

    chamfer_text = (f"── 3D Metrics ──\n"
                    f"Mean FLS→MBES (precision/noise): {chamfer['mean_fls2mbes']:.4f} m\n"
                    f"Mean MBES→FLS (recall/coverage): {chamfer['mean_mbes2fls']:.4f} m\n"
                    f"\n── Inliers within threshold ──\n"
                    f"{inlier_lines}")
                    # f"\n{guide}")

    fig.text(0.5, 0.02, chamfer_text, ha='center', va='bottom', fontsize=9, family='monospace',
             bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.8))

    fig.suptitle(f'{name}  —  FLS vs MBES Error', fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0.28, 1, 0.95])
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

        # metrics = compute_error(fls_pts, mbes_pts, args.bin_size)
        chamfer = compute_chamfer(fls_pts, mbes_pts)
        print(f"  Chamfer:   {chamfer['chamfer']:.6f} m²  |  Hausdorff: {chamfer['hausdorff']:.4f} m")
        print(f"  Mean FLS→MBES: {chamfer['mean_fls2mbes']:.4f} m  |  Mean MBES→FLS: {chamfer['mean_mbes2fls']:.4f} m")
        for tau, v in sorted(chamfer['f_scores'].items()):
            print(f"  F@{tau:.2f}m: {v['f']:.3f}  (P={v['precision']:.2f} R={v['recall']:.2f})")
        print(f"  Inlier counts (points within threshold):")
        for tau, iv in sorted(chamfer['inliers'].items()):
            print(f"    τ={tau:.2f}m — FLS: {iv['fls_inliers']}/{iv['fls_total']} ({iv['fls_pct']:.1f}%)  "
                  f"MBES: {iv['mbes_inliers']}/{iv['mbes_total']} ({iv['mbes_pct']:.1f}%)")
        fig = plot_pair(fls_pts, mbes_pts, name, chamfer, args.subsample)

        if args.save:
            out_path = os.path.join(args.save, f'{name}_comparison.png')
            fig.savefig(out_path, dpi=150, bbox_inches='tight')
            print(f"  Saved: {out_path}")
            plt.close(fig)

    if not args.save:
        plt.show()


if __name__ == '__main__':
    main()
