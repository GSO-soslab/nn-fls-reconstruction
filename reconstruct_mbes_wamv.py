#!/usr/bin/env python3
"""
Reconstruct MBES point cloud from wamv_rise bag.
Reads /wamv_rise/norbit_mbes/cloud (PointCloud2), transforms each frame
to world frame via TF, and saves as PLY.

Usage:
    python3 reconstruct_mbes_wamv.py \
        /home/farhang/Documents/Alpha_bags/wamv/rosbag2_2025_10_10-18_09_18/rosbag2_2025_10_10-18_09_18_0.mcap \
        --output wamv_mbes.ply
"""

import argparse
import struct
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import deque
from mcap_ros2.reader import read_ros2_messages


# ---------------------------------------------------------------------------
# TF helpers (same convention as reconstruct_from_rosbag_with_inference.py)
# ---------------------------------------------------------------------------

def quaternion_to_rotation_matrix(qx, qy, qz, qw):
    return np.array([
        [1 - 2*(qy**2 + qz**2), 2*(qx*qy - qw*qz), 2*(qx*qz + qw*qy)],
        [2*(qx*qy + qw*qz),     1 - 2*(qx**2 + qz**2), 2*(qy*qz - qw*qx)],
        [2*(qx*qz - qw*qy),     2*(qy*qz + qw*qx),     1 - 2*(qx**2 + qy**2)]
    ])


def slerp(q0, q1, t):
    q0 = q0 / np.linalg.norm(q0)
    q1 = q1 / np.linalg.norm(q1)
    dot = np.dot(q0, q1)
    if dot < 0:
        q1, dot = -q1, -dot
    if dot > 0.9995:
        return (q0 + t * (q1 - q0)) / np.linalg.norm(q0 + t * (q1 - q0))
    theta0 = np.arccos(dot)
    theta = theta0 * t
    q2 = (q1 - q0 * dot) / np.linalg.norm(q1 - q0 * dot)
    return q0 * np.cos(theta) + q2 * np.sin(theta)


def lookup_single(tf_tree, parent, child, timestamp):
    key = f"{parent}->{child}"
    if key not in tf_tree:
        return None, None
    tfs = tf_tree[key]
    before = after = None
    for tf in tfs:
        if tf['timestamp'] <= timestamp:
            before = tf
        if tf['timestamp'] >= timestamp:
            after = tf
            break
    if before is None and after is None:
        return None, None
    entry = before if after is None else (after if before is None else None)
    if entry is None:
        if before['timestamp'] == after['timestamp']:
            entry = before
        else:
            dt = after['timestamp'] - before['timestamp']
            alpha = (timestamp - before['timestamp']) / dt
            t = (1 - alpha) * before['translation'] + alpha * after['translation']
            q = slerp(before['rotation'], after['rotation'], alpha)
            return t, quaternion_to_rotation_matrix(*q)
    return entry['translation'], quaternion_to_rotation_matrix(*entry['rotation'])


def get_transform(tf_tree, from_frame, to_frame, timestamp):
    """Returns (t, R) such that p_world = R @ p_sensor + t"""
    if from_frame == to_frame:
        return np.zeros(3), np.eye(3)

    edges = {}
    for key in tf_tree:
        if '->' not in key:
            continue
        parent, child = key.split('->', 1)
        edges.setdefault(parent, []).append((child, True))
        edges.setdefault(child,  []).append((parent, False))

    queue = deque([(from_frame, [], [])])
    visited = {from_frame}
    path_keys = path_fwds = None
    while queue:
        frame, keys, fwds = queue.popleft()
        if frame == to_frame:
            path_keys, path_fwds = keys, fwds
            break
        for nb, fwd in edges.get(frame, []):
            if nb not in visited:
                visited.add(nb)
                k = f"{frame}->{nb}" if fwd else f"{nb}->{frame}"
                queue.append((nb, keys + [k], fwds + [fwd]))

    if path_keys is None:
        return None, None

    R_combined = np.eye(3)
    t_combined = np.zeros(3)
    for key, forward in zip(path_keys, path_fwds):
        parent, child = key.split('->', 1)
        t, R = lookup_single(tf_tree, parent, child, timestamp)
        if t is None:
            return None, None
        if forward:
            R = R.T
            t = -R @ t
        R_combined = R @ R_combined
        t_combined = R @ t_combined + t

    return t_combined, R_combined


# ---------------------------------------------------------------------------
# PointCloud2 parser
# ---------------------------------------------------------------------------

def parse_pointcloud2(msg):
    """Extract (x, y, z, intensity) arrays from a PointCloud2 message."""
    fields = {f.name: f for f in msg.fields}
    point_step = msg.point_step
    n_points = msg.width * msg.height
    data = bytes(msg.data)

    offsets = {name: fields[name].offset for name in fields}
    dtype_map = {1: 'b', 2: 'B', 3: 'h', 4: 'H', 5: 'i', 6: 'I', 7: 'f', 8: 'd'}

    xs, ys, zs, ins = [], [], [], []
    for i in range(n_points):
        base = i * point_step
        x = struct.unpack_from('f', data, base + offsets['x'])[0]
        y = struct.unpack_from('f', data, base + offsets['y'])[0]
        z = struct.unpack_from('f', data, base + offsets['z'])[0]
        intensity = struct.unpack_from('f', data, base + offsets.get('intensity', offsets['x']))[0] if 'intensity' in offsets else 1.0
        if np.isfinite(x) and np.isfinite(y) and np.isfinite(z):
            xs.append(x); ys.append(y); zs.append(z); ins.append(intensity)

    return np.array(xs), np.array(ys), np.array(zs), np.array(ins)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('bag', help='Path to .mcap bag file')
    parser.add_argument('--output', default='wamv_mbes.ply', help='Output PLY file')
    parser.add_argument('--sensor-frame', default='wamv_rise/norbit')
    parser.add_argument('--world-frame', default='wamv_rise/world')
    parser.add_argument('--topic', default='/wamv_rise/norbit_mbes/cloud')
    parser.add_argument('--plot-odom', action='store_true',
                        help='Plot sensor and base_link trajectories after processing')
    parser.add_argument('--save-odom-plot', type=str, default=None,
                        help='Save trajectory plot to PNG file')
    parser.add_argument('--base-link-frame', type=str, default=None,
                        help='base_link frame for trajectory comparison (default: auto-derived from sensor-frame)')
    args = parser.parse_args()

    bag = Path(args.bag)
    tf_tree = {}

    # --- Pass 1: build TF tree ---
    print("Building TF tree...")
    with open(bag, 'rb') as f:
        for msg in read_ros2_messages(f):
            if msg.channel.topic not in ['/tf', '/tf_static']:
                continue
            for t in msg.ros_msg.transforms:
                stamp = t.header.stamp
                ts_ns = int(stamp.sec * 1e9 + stamp.nanosec)
                if ts_ns == 0:
                    ts_ns = msg.log_time_ns
                key = f"{t.header.frame_id}->{t.child_frame_id}"
                tf_tree.setdefault(key, []).append({
                    'timestamp': ts_ns,
                    'translation': np.array([t.transform.translation.x,
                                             t.transform.translation.y,
                                             t.transform.translation.z]),
                    'rotation': np.array([t.transform.rotation.x,
                                          t.transform.rotation.y,
                                          t.transform.rotation.z,
                                          t.transform.rotation.w]),
                })

    for key in tf_tree:
        tf_tree[key].sort(key=lambda x: x['timestamp'])

    print(f"TF tree: {len(tf_tree)} chains")
    for k in tf_tree:
        print(f"  {k}: {len(tf_tree[k])} transforms")

    # --- Pass 2: accumulate MBES points ---
    print(f"Processing {args.topic}...")
    all_points = []
    all_intensities = []
    sensor_positions = []   # sensor frame origin in world frame
    bl_positions = []       # base_link origin in world frame
    bl_frame = args.base_link_frame or (args.sensor_frame.rsplit('/', 1)[0] + '/base_link' if '/' in args.sensor_frame else 'base_link')
    count = 0
    skipped = 0

    with open(bag, 'rb') as f:
        for msg in read_ros2_messages(f):
            if msg.channel.topic != args.topic:
                continue

            stamp = msg.ros_msg.header.stamp
            ts_ns = int(stamp.sec * 1e9 + stamp.nanosec)
            if ts_ns == 0:
                ts_ns = msg.log_time_ns

            t_world, R_world = get_transform(tf_tree, args.sensor_frame, args.world_frame, ts_ns)
            if t_world is None:
                skipped += 1
                continue

            xs, ys, zs, ins = parse_pointcloud2(msg.ros_msg)
            if len(xs) == 0:
                continue

            pts_sensor = np.stack([xs, ys, zs], axis=1)  # (N, 3)
            pts_world = (R_world @ pts_sensor.T).T + t_world  # (N, 3)

            all_points.append(pts_world)
            all_intensities.append(ins)
            sensor_positions.append(t_world.copy())
            t_bl, _ = get_transform(tf_tree, bl_frame, args.world_frame, ts_ns)
            if t_bl is not None:
                bl_positions.append(t_bl.copy())
            count += 1

            if count % 500 == 0:
                print(f"  {count} frames, {sum(len(p) for p in all_points)} points so far")

    print(f"Done: {count} frames processed, {skipped} skipped (no TF)")

    if not all_points:
        print("ERROR: No points collected.")
        return

    pts = np.concatenate(all_points, axis=0)
    ints = np.concatenate(all_intensities, axis=0)
    print(f"Total points: {len(pts)}")
    print(f"X: {pts[:,0].min():.2f} to {pts[:,0].max():.2f}")
    print(f"Y: {pts[:,1].min():.2f} to {pts[:,1].max():.2f}")
    print(f"Z: {pts[:,2].min():.2f} to {pts[:,2].max():.2f}")

    # Normalize intensity to 0-255 for PLY
    if ints.max() > ints.min():
        ints_norm = ((ints - ints.min()) / (ints.max() - ints.min()) * 255).astype(np.uint8)
    else:
        ints_norm = np.full(len(ints), 128, dtype=np.uint8)

    # Write PLY
    out = Path(args.output)
    with open(out, 'w') as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(pts)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for (x, y, z), c in zip(pts, ints_norm):
            f.write(f"{x:.4f} {y:.4f} {z:.4f} {c} {c} {c}\n")

    print(f"Saved: {out}")

    # Trajectory plot
    if args.plot_odom or args.save_odom_plot:
        if len(sensor_positions) < 2:
            print("WARNING: Not enough poses to plot trajectory.")
        else:
            _, ax = plt.subplots(figsize=(8, 8))
            pts_s = np.array(sensor_positions)
            ax.plot(pts_s[:, 0], pts_s[:, 1], linewidth=1.5, label=f'{args.sensor_frame} (sensor)')
            ax.scatter(pts_s[0, 0], pts_s[0, 1], c='green', s=60, zorder=5, label='Start')
            ax.scatter(pts_s[-1, 0], pts_s[-1, 1], c='red', s=60, zorder=5, label='End')
            if len(bl_positions) >= 2:
                pts_bl = np.array(bl_positions)
                ax.plot(pts_bl[:, 0], pts_bl[:, 1], linewidth=1.5, linestyle='--', label=bl_frame)
            ax.set_xlabel('X (m)')
            ax.set_ylabel('Y (m)')
            ax.set_title('Trajectory: sensor vs base_link (world frame)')
            ax.legend()
            ax.set_aspect('equal')
            ax.grid(True)
            plt.tight_layout()
            if args.save_odom_plot:
                plt.savefig(args.save_odom_plot, dpi=150)
                print(f"Saved trajectory plot to {args.save_odom_plot}")
            plt.show()


if __name__ == '__main__':
    main()
