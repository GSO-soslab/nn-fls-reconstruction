#!/usr/bin/env python3
"""
Parses a URDF file and injects all fixed joint transforms into a TF tree dict.

Usage (standalone):
    python3 urdf_tf_injector.py /path/to/robot.urdf --prefix alpha_rise

Usage (as a module):
    from urdf_tf_injector import inject_urdf_static_tfs
    inject_urdf_static_tfs(reconstructor.tf_tree, urdf_path, tf_prefix='alpha_rise')
"""

import xml.etree.ElementTree as ET
import numpy as np
import argparse


def rpy_to_quaternion(roll, pitch, yaw):
    """Convert RPY (radians) to quaternion [qx, qy, qz, qw]."""
    cr, sr = np.cos(roll / 2),  np.sin(roll / 2)
    cp, sp = np.cos(pitch / 2), np.sin(pitch / 2)
    cy, sy = np.cos(yaw / 2),   np.sin(yaw / 2)
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy
    qw = cr * cp * cy + sr * sp * sy
    return np.array([qx, qy, qz, qw])


def inject_urdf_static_tfs(tf_tree, urdf_path, tf_prefix=''):
    """
    Parse all fixed joints from a URDF and inject them into tf_tree.

    tf_tree: dict used by FLSPointCloudReconstructor (modified in-place)
    urdf_path: path to the .urdf file
    tf_prefix: namespace prefix to prepend to frame names (e.g. 'alpha_rise')
               Use '' for no prefix. A '/' separator is added automatically.
    """
    tree = ET.parse(urdf_path)
    root = tree.getroot()

    prefix = f"{tf_prefix}/" if tf_prefix else ''
    injected = 0

    for joint in root.findall('joint'):
        if joint.get('type') != 'fixed':
            continue

        parent_el = joint.find('parent')
        child_el  = joint.find('child')
        origin_el = joint.find('origin')

        if parent_el is None or child_el is None:
            continue

        parent = prefix + parent_el.get('link')
        child  = prefix + child_el.get('link')

        # Default to identity if no origin element
        xyz = np.array([0.0, 0.0, 0.0])
        quat = np.array([0.0, 0.0, 0.0, 1.0])

        if origin_el is not None:
            xyz_str = origin_el.get('xyz', '0 0 0')
            rpy_str = origin_el.get('rpy', '0 0 0')
            xyz  = np.array([float(v) for v in xyz_str.split()])
            rpy  = [float(v) for v in rpy_str.split()]
            quat = rpy_to_quaternion(*rpy)

        key = f"{parent}->{child}"
        tf_tree[key] = [{
            'timestamp': 0,  # static — valid at all times
            'translation': xyz,
            'rotation': quat,
        }]
        injected += 1

    print(f"Injected {injected} static TF(s) from URDF: {urdf_path}")
    for key in tf_tree:
        if tf_tree[key][0]['timestamp'] == 0:
            print(f"  [static] {key}")

    return injected


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parse URDF and print static TF frames')
    parser.add_argument('urdf', type=str, help='Path to URDF file')
    parser.add_argument('--prefix', type=str, default='',
                        help='TF namespace prefix (e.g. alpha_rise)')
    args = parser.parse_args()

    tf_tree = {}
    inject_urdf_static_tfs(tf_tree, args.urdf, tf_prefix=args.prefix)
