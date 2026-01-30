#!/usr/bin/env python3
"""
Open3D PLY viewer with colormap selection.

Usage:
    python view_ply.py fls_output.ply
    python view_ply.py fls_output.ply --colormap jet --axis z
    python view_ply.py fls_output.ply mbes_output.ply --colormap viridis
"""

import argparse
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt


def apply_colormap(pcd, colormap='viridis', axis='z'):
    """
    Apply matplotlib colormap to point cloud based on coordinate axis.

    Args:
        pcd: Open3D PointCloud
        colormap: matplotlib colormap name (viridis, jet, plasma, inferno, magma, etc.)
        axis: 'x', 'y', or 'z' - which axis to use for coloring

    Returns:
        pcd: Modified point cloud
        v_min, v_max: Value range for colorbar
    """
    points = np.asarray(pcd.points)

    if len(points) == 0:
        return pcd, 0, 0

    axis_map = {'x': 0, 'y': 1, 'z': 2}
    axis_idx = axis_map.get(axis.lower(), 2)

    values = points[:, axis_idx]

    # Normalize to [0, 1]
    v_min, v_max = values.min(), values.max()
    if v_max - v_min > 1e-10:
        normalized = (values - v_min) / (v_max - v_min)
    else:
        normalized = np.zeros_like(values)

    # Apply colormap
    cmap = plt.get_cmap(colormap)
    colors = cmap(normalized)[:, :3]  # RGB only, no alpha

    pcd.colors = o3d.utility.Vector3dVector(colors)

    print(f"Applied '{colormap}' colormap on {axis}-axis")
    print(f"  Range: {v_min:.3f} to {v_max:.3f}")

    return pcd, v_min, v_max


def apply_solid_color(pcd, color):
    """Apply a solid color to all points."""
    points = np.asarray(pcd.points)
    colors = np.tile(color, (len(points), 1))
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd


def save_matplotlib_figure(pcd, colormap='viridis', axis='z', output_path='output.png',
                           elev=30, azim=45, point_size=1):
    """
    Save a publication-ready 3D scatter plot with colorbar using matplotlib.
    """

    points = np.asarray(pcd.points)
    axis_map = {'x': 0, 'y': 1, 'z': 2}
    axis_idx = axis_map.get(axis.lower(), 2)
    values = points[:, axis_idx]

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    scatter = ax.scatter(points[:, 0], points[:, 1], points[:, 2],
                        c=values, cmap=colormap, s=point_size, alpha=0.8)

    cbar = fig.colorbar(scatter, ax=ax, shrink=0.6, pad=0.1)
    cbar.set_label(f'{axis.upper()} (m)', fontsize=12)

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.view_init(elev=elev, azim=azim)

    ax.set_box_aspect([1, 1, 0.5])

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved figure to: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='View PLY files with colormap')
    parser.add_argument('files', nargs='+', help='PLY file(s) to view')
    parser.add_argument('--colormap', '-c', default='viridis',
                       help='Colormap name (viridis, jet, plasma, inferno, turbo, etc.)')
    parser.add_argument('--axis', '-a', default='z', choices=['x', 'y', 'z'],
                       help='Axis to use for coloring (default: z)')
    parser.add_argument('--point-size', '-p', type=float, default=3.0,
                       help='Point size (default: 1.0)')
    parser.add_argument('--bg-color', '-b', default='black',
                       choices=['black', 'white', 'gray'],
                       help='Background color')
    parser.add_argument('--no-color', action='store_true',
                       help='Show without colormap (original colors or gray)')
    parser.add_argument('--overlay', '-o', action='store_true',
                       help='Overlay mode: use solid colors (red/blue) to distinguish point clouds')
    parser.add_argument('--save-figure', '-s', type=str, default=None,
                       help='Save a publication-ready matplotlib figure with colorbar (e.g., output.png)')

    args = parser.parse_args()

    geometries = []

    # Solid colors for overlay mode (easily distinguishable)
    overlay_colors = [
        [0.0, 0.5, 1.0],   # Blue (FLS / first file)
        [1.0, 0.3, 0.0],   # Orange-Red (MBES / second file)
        [0.0, 1.0, 0.3],   # Green (third file)
        [1.0, 1.0, 0.0],   # Yellow (fourth file)
        [1.0, 0.0, 1.0],   # Magenta (fifth file)
    ]

    for i, filepath in enumerate(args.files):
        print(f"\nLoading: {filepath}")
        pcd = o3d.io.read_point_cloud(filepath)
        print(f"  Points: {len(pcd.points)}")

        if args.overlay:
            # Use solid colors for overlay comparison
            color = overlay_colors[i % len(overlay_colors)]
            pcd = apply_solid_color(pcd, color)
            color_names = ['Blue', 'Orange-Red', 'Green', 'Yellow', 'Magenta']
            print(f"  Color: {color_names[i % len(color_names)]}")
        elif not args.no_color:
            # Use different colormaps for multiple files
            if len(args.files) > 1:
                cmaps = ['viridis', 'plasma', 'inferno', 'magma', 'cividis']
                cmap = cmaps[i % len(cmaps)]
            else:
                cmap = args.colormap

            pcd, _, _ = apply_colormap(pcd, colormap=cmap, axis=args.axis)

        geometries.append(pcd)

    # Visualization settings
    bg_colors = {
        'black': [0.0, 0.0, 0.0],
        'white': [1.0, 1.0, 1.0],
        'gray': [0.2, 0.2, 0.2]
    }

    print("\n" + "="*50)
    print("Controls:")
    print("  Mouse drag: Rotate")
    print("  Scroll: Zoom")
    print("  Shift+drag: Pan")
    print("  R: Reset view")
    print("  +/-: Increase/decrease point size")
    print("  Q/Esc: Quit")
    if args.overlay and len(args.files) >= 2:
        print("-"*50)
        print("Legend:")
        color_names = ['Blue', 'Orange-Red', 'Green', 'Yellow', 'Magenta']
        for i, f in enumerate(args.files):
            print(f"  {color_names[i % len(color_names)]}: {f}")
    print("="*50)

    # Save matplotlib figure if requested
    if args.save_figure:
        pcd = geometries[0]
        save_matplotlib_figure(pcd, colormap=args.colormap, axis=args.axis,
                              output_path=args.save_figure, point_size=args.point_size)
        return

    # Create visualizer with custom settings
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="PLY Viewer", width=1280, height=720)

    for geom in geometries:
        vis.add_geometry(geom)

    # Set render options
    opt = vis.get_render_option()
    opt.background_color = np.array(bg_colors[args.bg_color])
    opt.point_size = args.point_size

    vis.run()
    vis.destroy_window()


if __name__ == "__main__":
    main()
