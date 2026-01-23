#!/usr/bin/env python3
"""
Trajectory Error Metrics for Odometry Evaluation.

Implements standard metrics from:
    Sturm et al., "A Benchmark for the Evaluation of RGB-D SLAM Systems",
    IROS 2012. (TUM RGB-D Benchmark)

Metrics:
    - ATE (Absolute Trajectory Error): Global trajectory consistency
    - RPE (Relative Pose Error): Local drift / odometry consistency

References:
    [1] J. Sturm, N. Engelhard, F. Endres, W. Burgard, D. Cremers,
        "A Benchmark for the Evaluation of RGB-D SLAM Systems", IROS 2012.
    [2] Geiger et al., "Are we ready for Autonomous Driving? The KITTI
        Vision Benchmark Suite", CVPR 2012.
    [3] M. Grupp, "evo: Python package for the evaluation of odometry and SLAM"
        https://github.com/MichaelGrupp/evo
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict, Union
from enum import Enum
import warnings


class PoseRelation(Enum):
    """
    Defines which component of the pose error to extract.

    Following evo/Sturm et al. conventions:
        - TRANSLATION_PART: Euclidean norm of translation error (meters)
        - ROTATION_ANGLE_DEG: Rotation angle in degrees
        - ROTATION_ANGLE_RAD: Rotation angle in radians
        - FULL_TRANSFORMATION: Frobenius norm of full SE(3) difference (unitless)
    """
    TRANSLATION_PART = "translation_part"
    ROTATION_ANGLE_DEG = "rotation_angle_deg"
    ROTATION_ANGLE_RAD = "rotation_angle_rad"
    FULL_TRANSFORMATION = "full_transformation"


class DeltaUnit(Enum):
    """Units for RPE delta parameter."""
    FRAMES = "frames"
    METERS = "meters"
    SECONDS = "seconds"


@dataclass
class TrajectoryErrorStatistics:
    """Container for error statistics."""
    rmse: float
    mean: float
    std: float
    median: float
    min: float
    max: float
    sse: float  # Sum of squared errors
    num_samples: int

    def __str__(self) -> str:
        return (
            f"RMSE:   {self.rmse:.6f}\n"
            f"Mean:   {self.mean:.6f}\n"
            f"Std:    {self.std:.6f}\n"
            f"Median: {self.median:.6f}\n"
            f"Min:    {self.min:.6f}\n"
            f"Max:    {self.max:.6f}\n"
            f"SSE:    {self.sse:.6f}\n"
            f"Samples: {self.num_samples}"
        )


@dataclass
class Trajectory:
    """
    Represents a trajectory as a sequence of SE(3) poses with timestamps.

    Attributes:
        timestamps: Array of timestamps (N,) in nanoseconds or seconds
        positions: Array of positions (N, 3) as [x, y, z]
        orientations: Array of quaternions (N, 4) as [qx, qy, qz, qw]
    """
    timestamps: np.ndarray
    positions: np.ndarray
    orientations: np.ndarray  # quaternions [qx, qy, qz, qw]

    def __post_init__(self):
        """Validate trajectory data."""
        n = len(self.timestamps)
        assert self.positions.shape == (n, 3), f"positions shape {self.positions.shape} != ({n}, 3)"
        assert self.orientations.shape == (n, 4), f"orientations shape {self.orientations.shape} != ({n}, 4)"

        # Normalize quaternions
        norms = np.linalg.norm(self.orientations, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self.orientations = self.orientations / norms

    def __len__(self) -> int:
        return len(self.timestamps)

    def get_poses_se3(self) -> np.ndarray:
        """Return poses as (N, 4, 4) SE(3) matrices."""
        n = len(self)
        poses = np.zeros((n, 4, 4))
        poses[:, 3, 3] = 1.0
        poses[:, :3, 3] = self.positions
        for i in range(n):
            poses[i, :3, :3] = quaternion_to_rotation_matrix(self.orientations[i])
        return poses

    def get_distances(self) -> np.ndarray:
        """Compute cumulative distance traveled along trajectory."""
        if len(self) < 2:
            return np.array([0.0])
        deltas = np.linalg.norm(np.diff(self.positions, axis=0), axis=1)
        return np.concatenate([[0.0], np.cumsum(deltas)])


# =============================================================================
# SE(3) / SO(3) Operations
# =============================================================================

def quaternion_to_rotation_matrix(q: np.ndarray) -> np.ndarray:
    """
    Convert quaternion [qx, qy, qz, qw] to 3x3 rotation matrix.

    Args:
        q: Quaternion as [qx, qy, qz, qw]

    Returns:
        3x3 rotation matrix
    """
    qx, qy, qz, qw = q

    # Normalize
    n = np.sqrt(qx*qx + qy*qy + qz*qz + qw*qw)
    if n > 0:
        qx, qy, qz, qw = qx/n, qy/n, qz/n, qw/n

    R = np.array([
        [1 - 2*(qy**2 + qz**2), 2*(qx*qy - qw*qz), 2*(qx*qz + qw*qy)],
        [2*(qx*qy + qw*qz), 1 - 2*(qx**2 + qz**2), 2*(qy*qz - qw*qx)],
        [2*(qx*qz - qw*qy), 2*(qy*qz + qw*qx), 1 - 2*(qx**2 + qy**2)]
    ])
    return R


def rotation_matrix_to_quaternion(R: np.ndarray) -> np.ndarray:
    """
    Convert 3x3 rotation matrix to quaternion [qx, qy, qz, qw].

    Uses Shepperd's method for numerical stability.
    """
    trace = np.trace(R)

    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        qw = 0.25 / s
        qx = (R[2, 1] - R[1, 2]) * s
        qy = (R[0, 2] - R[2, 0]) * s
        qz = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        qw = (R[2, 1] - R[1, 2]) / s
        qx = 0.25 * s
        qy = (R[0, 1] + R[1, 0]) / s
        qz = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        qw = (R[0, 2] - R[2, 0]) / s
        qx = (R[0, 1] + R[1, 0]) / s
        qy = 0.25 * s
        qz = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        qw = (R[1, 0] - R[0, 1]) / s
        qx = (R[0, 2] + R[2, 0]) / s
        qy = (R[1, 2] + R[2, 1]) / s
        qz = 0.25 * s

    q = np.array([qx, qy, qz, qw])
    return q / np.linalg.norm(q)


def se3_inverse(T: np.ndarray) -> np.ndarray:
    """
    Compute inverse of SE(3) transformation matrix.

    For T = [R | t], T^{-1} = [R^T | -R^T @ t]
    """
    T_inv = np.eye(4)
    R = T[:3, :3]
    t = T[:3, 3]
    T_inv[:3, :3] = R.T
    T_inv[:3, 3] = -R.T @ t
    return T_inv


def se3_compose(T1: np.ndarray, T2: np.ndarray) -> np.ndarray:
    """Compose two SE(3) transformations: T1 @ T2."""
    return T1 @ T2


def relative_se3(T_ref: np.ndarray, T_est: np.ndarray) -> np.ndarray:
    """
    Compute relative transformation: T_ref^{-1} @ T_est

    This gives the error transformation E such that:
        T_est = T_ref @ E
        E = T_ref^{-1} @ T_est

    If T_est == T_ref, E == I (identity).
    """
    return se3_inverse(T_ref) @ T_est


def so3_rotation_angle(R: np.ndarray) -> float:
    """
    Extract rotation angle from rotation matrix using Rodrigues' formula.

    theta = arccos((trace(R) - 1) / 2)

    Returns angle in radians [0, pi].
    """
    trace = np.trace(R)
    # Clamp for numerical stability
    cos_theta = np.clip((trace - 1.0) / 2.0, -1.0, 1.0)
    return np.arccos(cos_theta)


# =============================================================================
# Trajectory Alignment (Umeyama's Method)
# =============================================================================

def umeyama_alignment(
    src: np.ndarray,
    dst: np.ndarray,
    with_scale: bool = False
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Umeyama's method for finding optimal rigid-body transformation.

    Finds R, t, s that minimizes: sum_i || dst_i - s * R @ src_i - t ||^2

    Reference:
        S. Umeyama, "Least-Squares Estimation of Transformation Parameters
        Between Two Point Patterns", IEEE PAMI, 1991.

    Args:
        src: Source points (N, 3)
        dst: Destination points (N, 3) - typically ground truth
        with_scale: If True, also estimate scale factor

    Returns:
        R: 3x3 rotation matrix
        t: 3x1 translation vector
        s: Scale factor (1.0 if with_scale=False)
    """
    assert src.shape == dst.shape
    n, dim = src.shape

    # Compute centroids
    src_mean = np.mean(src, axis=0)
    dst_mean = np.mean(dst, axis=0)

    # Center the points
    src_centered = src - src_mean
    dst_centered = dst - dst_mean

    # Compute variances
    src_var = np.sum(src_centered ** 2) / n

    # Compute cross-covariance matrix
    cov = (dst_centered.T @ src_centered) / n

    # SVD
    U, D, Vt = np.linalg.svd(cov)

    # Handle reflection case
    S = np.eye(dim)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[dim - 1, dim - 1] = -1

    # Rotation
    R = U @ S @ Vt

    # Scale
    if with_scale and src_var > 1e-10:
        s = np.trace(np.diag(D) @ S) / src_var
    else:
        s = 1.0

    # Translation
    t = dst_mean - s * R @ src_mean

    return R, t, s


def align_trajectories(
    traj_est: Trajectory,
    traj_ref: Trajectory,
    with_scale: bool = False
) -> Tuple[Trajectory, np.ndarray]:
    """
    Align estimated trajectory to reference trajectory using Umeyama's method.

    Args:
        traj_est: Estimated trajectory
        traj_ref: Reference (ground truth) trajectory
        with_scale: If True, also correct for scale (useful for monocular VO)

    Returns:
        aligned_traj: Aligned estimated trajectory
        T_align: 4x4 alignment transformation
    """
    R, t, s = umeyama_alignment(traj_est.positions, traj_ref.positions, with_scale)

    # Build SE(3) alignment matrix
    T_align = np.eye(4)
    T_align[:3, :3] = s * R
    T_align[:3, 3] = t

    # Apply alignment to positions
    aligned_positions = s * (traj_est.positions @ R.T) + t

    # Apply alignment to orientations
    aligned_orientations = np.zeros_like(traj_est.orientations)
    R_align_q = rotation_matrix_to_quaternion(R)
    for i in range(len(traj_est)):
        q_est = traj_est.orientations[i]
        # q_aligned = R_align * q_est (quaternion multiplication)
        aligned_orientations[i] = quaternion_multiply(R_align_q, q_est)

    aligned_traj = Trajectory(
        timestamps=traj_est.timestamps.copy(),
        positions=aligned_positions,
        orientations=aligned_orientations
    )

    return aligned_traj, T_align


def quaternion_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """
    Multiply two quaternions: q1 * q2.

    Quaternion format: [qx, qy, qz, qw]
    """
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2

    return np.array([
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
        w1*w2 - x1*x2 - y1*y2 - z1*z2
    ])


# =============================================================================
# Trajectory Association
# =============================================================================

def associate_trajectories(
    traj_est: Trajectory,
    traj_ref: Trajectory,
    max_diff: float = 0.1,
    timestamp_unit: str = "seconds"
) -> Tuple[Trajectory, Trajectory]:
    """
    Associate trajectories by matching timestamps.

    For each reference timestamp, finds the closest estimated timestamp
    within max_diff threshold.

    Args:
        traj_est: Estimated trajectory
        traj_ref: Reference trajectory
        max_diff: Maximum timestamp difference for association
        timestamp_unit: "seconds" or "nanoseconds"

    Returns:
        Tuple of (matched_est, matched_ref) with same length
    """
    if timestamp_unit == "nanoseconds":
        max_diff_ns = max_diff
    else:
        max_diff_ns = max_diff * 1e9

    matched_est_idx = []
    matched_ref_idx = []

    for i, t_ref in enumerate(traj_ref.timestamps):
        # Find closest timestamp in estimated trajectory
        diffs = np.abs(traj_est.timestamps - t_ref)
        min_idx = np.argmin(diffs)

        if diffs[min_idx] <= max_diff_ns:
            matched_est_idx.append(min_idx)
            matched_ref_idx.append(i)

    if len(matched_est_idx) == 0:
        raise ValueError("No matching timestamps found within threshold")

    matched_est = Trajectory(
        timestamps=traj_est.timestamps[matched_est_idx],
        positions=traj_est.positions[matched_est_idx],
        orientations=traj_est.orientations[matched_est_idx]
    )

    matched_ref = Trajectory(
        timestamps=traj_ref.timestamps[matched_ref_idx],
        positions=traj_ref.positions[matched_ref_idx],
        orientations=traj_ref.orientations[matched_ref_idx]
    )

    return matched_est, matched_ref


# =============================================================================
# ATE (Absolute Trajectory Error)
# =============================================================================

def compute_ate(
    traj_est: Trajectory,
    traj_ref: Trajectory,
    pose_relation: PoseRelation = PoseRelation.TRANSLATION_PART,
    align: bool = True,
    with_scale: bool = False
) -> Tuple[np.ndarray, TrajectoryErrorStatistics, Optional[np.ndarray]]:
    """
    Compute Absolute Trajectory Error (ATE).

    ATE measures global trajectory consistency by comparing estimated poses
    directly against reference poses at each timestamp.

    The error at each timestep is:
        E_i = T_ref_i^{-1} @ T_est_i

    If alignment is enabled, T_est is first transformed by the optimal
    rigid-body alignment computed via Umeyama's method.

    Reference:
        Sturm et al., "A Benchmark for the Evaluation of RGB-D SLAM Systems",
        IROS 2012.

    Args:
        traj_est: Estimated trajectory
        traj_ref: Reference (ground truth) trajectory
        pose_relation: Which error component to extract
        align: Whether to align trajectories before computing error
        with_scale: Whether to correct for scale during alignment

    Returns:
        errors: Array of per-pose errors (N,)
        stats: Error statistics (RMSE, mean, std, etc.)
        T_align: Alignment transformation if align=True, else None
    """
    assert len(traj_est) == len(traj_ref), "Trajectories must have same length"

    T_align = None
    if align:
        traj_est, T_align = align_trajectories(traj_est, traj_ref, with_scale)

    # Get SE(3) poses
    poses_est = traj_est.get_poses_se3()
    poses_ref = traj_ref.get_poses_se3()

    # Compute per-pose errors
    n = len(traj_est)
    errors = np.zeros(n)

    for i in range(n):
        # E_i = T_ref_i^{-1} @ T_est_i
        E = relative_se3(poses_ref[i], poses_est[i])
        errors[i] = extract_pose_error(E, pose_relation)

    stats = compute_statistics(errors)

    return errors, stats, T_align


def extract_pose_error(E: np.ndarray, pose_relation: PoseRelation) -> float:
    """
    Extract scalar error from SE(3) error transformation.

    Args:
        E: 4x4 SE(3) error transformation
        pose_relation: Which component to extract

    Returns:
        Scalar error value
    """
    if pose_relation == PoseRelation.TRANSLATION_PART:
        # Euclidean norm of translation
        return np.linalg.norm(E[:3, 3])

    elif pose_relation == PoseRelation.ROTATION_ANGLE_RAD:
        # Rotation angle in radians
        return so3_rotation_angle(E[:3, :3])

    elif pose_relation == PoseRelation.ROTATION_ANGLE_DEG:
        # Rotation angle in degrees
        return np.degrees(so3_rotation_angle(E[:3, :3]))

    elif pose_relation == PoseRelation.FULL_TRANSFORMATION:
        # Frobenius norm of E - I
        return np.linalg.norm(E - np.eye(4), 'fro')

    else:
        raise ValueError(f"Unknown pose relation: {pose_relation}")


# =============================================================================
# RPE (Relative Pose Error)
# =============================================================================

def compute_rpe(
    traj_est: Trajectory,
    traj_ref: Trajectory,
    pose_relation: PoseRelation = PoseRelation.TRANSLATION_PART,
    delta: float = 1.0,
    delta_unit: DeltaUnit = DeltaUnit.FRAMES,
    all_pairs: bool = False
) -> Tuple[np.ndarray, TrajectoryErrorStatistics]:
    """
    Compute Relative Pose Error (RPE).

    RPE measures local consistency (drift) by comparing relative transformations
    between pose pairs separated by a fixed interval delta.

    The error for pose pair (i, j) where j = i + delta:
        E_ij = (T_ref_i^{-1} @ T_ref_j)^{-1} @ (T_est_i^{-1} @ T_est_j)
             = delta_ref_ij^{-1} @ delta_est_ij

    This measures the difference in relative motion between reference and estimate.

    Reference:
        Sturm et al., "A Benchmark for the Evaluation of RGB-D SLAM Systems",
        IROS 2012.

    Args:
        traj_est: Estimated trajectory
        traj_ref: Reference (ground truth) trajectory
        pose_relation: Which error component to extract
        delta: Interval between pose pairs
        delta_unit: Unit for delta (FRAMES, METERS, or SECONDS)
        all_pairs: If True, compute error for all valid pairs; else consecutive only

    Returns:
        errors: Array of per-pair errors
        stats: Error statistics (RMSE, mean, std, etc.)
    """
    assert len(traj_est) == len(traj_ref), "Trajectories must have same length"

    # Get SE(3) poses
    poses_est = traj_est.get_poses_se3()
    poses_ref = traj_ref.get_poses_se3()

    # Determine pairs based on delta_unit
    pairs = _get_delta_pairs(traj_ref, delta, delta_unit, all_pairs)

    if len(pairs) == 0:
        raise ValueError(f"No valid pose pairs found for delta={delta} {delta_unit.value}")

    # Compute per-pair errors
    errors = np.zeros(len(pairs))

    for k, (i, j) in enumerate(pairs):
        # delta_ref = T_ref_i^{-1} @ T_ref_j
        delta_ref = relative_se3(poses_ref[i], poses_ref[j])

        # delta_est = T_est_i^{-1} @ T_est_j
        delta_est = relative_se3(poses_est[i], poses_est[j])

        # E = delta_ref^{-1} @ delta_est
        E = relative_se3(delta_ref, delta_est)

        errors[k] = extract_pose_error(E, pose_relation)

    stats = compute_statistics(errors)

    return errors, stats


def _get_delta_pairs(
    traj: Trajectory,
    delta: float,
    delta_unit: DeltaUnit,
    all_pairs: bool
) -> List[Tuple[int, int]]:
    """Get index pairs based on delta specification."""
    n = len(traj)
    pairs = []

    if delta_unit == DeltaUnit.FRAMES:
        delta_int = int(delta)
        if all_pairs:
            for i in range(n):
                for j in range(i + delta_int, n, delta_int):
                    pairs.append((i, j))
        else:
            for i in range(n - delta_int):
                pairs.append((i, i + delta_int))

    elif delta_unit == DeltaUnit.METERS:
        distances = traj.get_distances()
        if all_pairs:
            for i in range(n):
                for j in range(i + 1, n):
                    if abs(distances[j] - distances[i] - delta) < delta * 0.1:
                        pairs.append((i, j))
        else:
            j = 0
            for i in range(n):
                while j < n and distances[j] - distances[i] < delta:
                    j += 1
                if j < n:
                    pairs.append((i, j))

    elif delta_unit == DeltaUnit.SECONDS:
        delta_ns = delta * 1e9
        if all_pairs:
            for i in range(n):
                for j in range(i + 1, n):
                    if abs(traj.timestamps[j] - traj.timestamps[i] - delta_ns) < delta_ns * 0.1:
                        pairs.append((i, j))
        else:
            j = 0
            for i in range(n):
                while j < n and traj.timestamps[j] - traj.timestamps[i] < delta_ns:
                    j += 1
                if j < n:
                    pairs.append((i, j))

    return pairs


# =============================================================================
# Statistics
# =============================================================================

def compute_statistics(errors: np.ndarray) -> TrajectoryErrorStatistics:
    """Compute standard statistics for error array."""
    if len(errors) == 0:
        return TrajectoryErrorStatistics(
            rmse=np.nan, mean=np.nan, std=np.nan, median=np.nan,
            min=np.nan, max=np.nan, sse=np.nan, num_samples=0
        )

    sse = np.sum(errors ** 2)
    return TrajectoryErrorStatistics(
        rmse=np.sqrt(sse / len(errors)),
        mean=np.mean(errors),
        std=np.std(errors),
        median=np.median(errors),
        min=np.min(errors),
        max=np.max(errors),
        sse=sse,
        num_samples=len(errors)
    )


# =============================================================================
# KITTI-style Metrics
# =============================================================================

def compute_kitti_metrics(
    traj_est: Trajectory,
    traj_ref: Trajectory,
    lengths: List[int] = [100, 200, 300, 400, 500, 600, 700, 800]
) -> Dict[str, float]:
    """
    Compute KITTI-style odometry metrics.

    Translation error: % drift per distance traveled
    Rotation error: deg/100m

    Computes errors over subsequences of specified lengths (in meters).

    Reference:
        Geiger et al., "Are we ready for Autonomous Driving? The KITTI
        Vision Benchmark Suite", CVPR 2012.

    Args:
        traj_est: Estimated trajectory
        traj_ref: Reference trajectory
        lengths: List of subsequence lengths in meters

    Returns:
        Dictionary with 't_rel' (%), 'r_rel' (deg/100m), and per-length errors
    """
    assert len(traj_est) == len(traj_ref), "Trajectories must have same length"

    poses_est = traj_est.get_poses_se3()
    poses_ref = traj_ref.get_poses_se3()
    distances = traj_ref.get_distances()

    trans_errors = []
    rot_errors = []

    for length in lengths:
        for i in range(len(traj_ref)):
            # Find endpoint j such that distance[j] - distance[i] >= length
            j = i + 1
            while j < len(traj_ref) and distances[j] - distances[i] < length:
                j += 1

            if j >= len(traj_ref):
                continue

            # Compute relative poses
            delta_ref = relative_se3(poses_ref[i], poses_ref[j])
            delta_est = relative_se3(poses_est[i], poses_est[j])

            # Error
            E = relative_se3(delta_ref, delta_est)

            # Translation error (%)
            ref_dist = distances[j] - distances[i]
            trans_err = np.linalg.norm(E[:3, 3]) / ref_dist * 100.0
            trans_errors.append(trans_err)

            # Rotation error (deg/m -> deg/100m)
            rot_err = np.degrees(so3_rotation_angle(E[:3, :3])) / ref_dist * 100.0
            rot_errors.append(rot_err)

    if len(trans_errors) == 0:
        return {'t_rel': np.nan, 'r_rel': np.nan}

    return {
        't_rel': np.mean(trans_errors),  # %
        'r_rel': np.mean(rot_errors),    # deg/100m
        't_rel_std': np.std(trans_errors),
        'r_rel_std': np.std(rot_errors),
        'num_segments': len(trans_errors)
    }


# =============================================================================
# Plotting Utilities
# =============================================================================

def plot_trajectory_comparison(
    traj_est: Trajectory,
    traj_ref: Trajectory,
    title: str = "Trajectory Comparison",
    aligned: bool = False,
    show_plot: bool = True,
    save_path: Optional[str] = None
):
    """
    Plot estimated vs reference trajectories in 2D (top-down XY view).
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        warnings.warn("matplotlib not available, skipping plot")
        return

    fig, ax = plt.subplots(figsize=(10, 8))

    ax.plot(traj_ref.positions[:, 0], traj_ref.positions[:, 1],
            'b-', linewidth=2, label='Ground Truth')
    ax.plot(traj_est.positions[:, 0], traj_est.positions[:, 1],
            'r--', linewidth=2, label='Estimated' + (' (aligned)' if aligned else ''))

    # Mark start/end
    ax.scatter(traj_ref.positions[0, 0], traj_ref.positions[0, 1],
               c='green', s=100, marker='o', zorder=5, label='Start')
    ax.scatter(traj_ref.positions[-1, 0], traj_ref.positions[-1, 1],
               c='red', s=100, marker='x', zorder=5, label='End')

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title(title)
    ax.legend()
    ax.axis('equal')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    if show_plot:
        plt.show()
    else:
        plt.close()


def plot_ate_over_time(
    errors: np.ndarray,
    timestamps: np.ndarray,
    title: str = "ATE over Time",
    show_plot: bool = True,
    save_path: Optional[str] = None
):
    """Plot ATE error over time."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        warnings.warn("matplotlib not available, skipping plot")
        return

    # Convert timestamps to seconds from start
    t_sec = (timestamps - timestamps[0]) / 1e9

    fig, ax = plt.subplots(figsize=(12, 4))

    ax.plot(t_sec, errors, 'b-', linewidth=1)
    ax.axhline(np.mean(errors), color='r', linestyle='--', label=f'Mean: {np.mean(errors):.3f}')
    ax.fill_between(t_sec, 0, errors, alpha=0.3)

    ax.set_xlabel('Time (s)')
    ax.set_ylabel('ATE (m)')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    if show_plot:
        plt.show()
    else:
        plt.close()


# =============================================================================
# High-Level API
# =============================================================================

class TrajectoryEvaluator:
    """
    High-level API for trajectory evaluation.

    Usage:
        evaluator = TrajectoryEvaluator(traj_est, traj_ref)
        ate_result = evaluator.compute_ate()
        rpe_result = evaluator.compute_rpe(delta=1.0, delta_unit=DeltaUnit.SECONDS)
        evaluator.print_results()
    """

    def __init__(
        self,
        traj_est: Trajectory,
        traj_ref: Trajectory,
        max_time_diff: float = 0.1,
        timestamp_unit: str = "seconds"
    ):
        """
        Initialize evaluator.

        Args:
            traj_est: Estimated trajectory
            traj_ref: Reference (ground truth) trajectory
            max_time_diff: Maximum time difference for timestamp association
            timestamp_unit: "seconds" or "nanoseconds"
        """
        # Associate trajectories by timestamp
        self.traj_est, self.traj_ref = associate_trajectories(
            traj_est, traj_ref, max_time_diff, timestamp_unit
        )

        self.ate_errors: Optional[np.ndarray] = None
        self.ate_stats: Optional[TrajectoryErrorStatistics] = None
        self.ate_align: Optional[np.ndarray] = None

        self.rpe_errors: Optional[np.ndarray] = None
        self.rpe_stats: Optional[TrajectoryErrorStatistics] = None

        self.kitti_metrics: Optional[Dict[str, float]] = None

    def compute_ate(
        self,
        pose_relation: PoseRelation = PoseRelation.TRANSLATION_PART,
        align: bool = True,
        with_scale: bool = False
    ) -> TrajectoryErrorStatistics:
        """Compute ATE and store results."""
        self.ate_errors, self.ate_stats, self.ate_align = compute_ate(
            self.traj_est, self.traj_ref, pose_relation, align, with_scale
        )
        return self.ate_stats

    def compute_rpe(
        self,
        pose_relation: PoseRelation = PoseRelation.TRANSLATION_PART,
        delta: float = 1.0,
        delta_unit: DeltaUnit = DeltaUnit.FRAMES,
        all_pairs: bool = False
    ) -> TrajectoryErrorStatistics:
        """Compute RPE and store results."""
        self.rpe_errors, self.rpe_stats = compute_rpe(
            self.traj_est, self.traj_ref, pose_relation, delta, delta_unit, all_pairs
        )
        return self.rpe_stats

    def compute_kitti(
        self,
        lengths: List[int] = [100, 200, 300, 400, 500, 600, 700, 800]
    ) -> Dict[str, float]:
        """Compute KITTI-style metrics."""
        self.kitti_metrics = compute_kitti_metrics(self.traj_est, self.traj_ref, lengths)
        return self.kitti_metrics

    def print_results(self, title: str = "Trajectory Evaluation Results"):
        """Print all computed results."""
        print("\n" + "=" * 70)
        print(title)
        print("=" * 70)
        print(f"Trajectory length: {len(self.traj_ref)} poses")

        if self.ate_stats is not None:
            print("\n--- ATE (Absolute Trajectory Error) ---")
            print(self.ate_stats)

        if self.rpe_stats is not None:
            print("\n--- RPE (Relative Pose Error) ---")
            print(self.rpe_stats)

        if self.kitti_metrics is not None:
            print("\n--- KITTI Metrics ---")
            print(f"Translation Error: {self.kitti_metrics['t_rel']:.4f} %")
            print(f"Rotation Error:    {self.kitti_metrics['r_rel']:.4f} deg/100m")
            print(f"Segments:          {self.kitti_metrics['num_segments']}")

        print("=" * 70)

    def get_results_dict(self) -> Dict:
        """Return all results as a dictionary."""
        results = {
            'num_poses': len(self.traj_ref),
        }

        if self.ate_stats is not None:
            results['ate'] = {
                'rmse': self.ate_stats.rmse,
                'mean': self.ate_stats.mean,
                'std': self.ate_stats.std,
                'median': self.ate_stats.median,
                'min': self.ate_stats.min,
                'max': self.ate_stats.max,
            }

        if self.rpe_stats is not None:
            results['rpe'] = {
                'rmse': self.rpe_stats.rmse,
                'mean': self.rpe_stats.mean,
                'std': self.rpe_stats.std,
                'median': self.rpe_stats.median,
                'min': self.rpe_stats.min,
                'max': self.rpe_stats.max,
            }

        if self.kitti_metrics is not None:
            results['kitti'] = self.kitti_metrics

        return results


# =============================================================================
# Main (Example Usage)
# =============================================================================

if __name__ == "__main__":
    # Example: Create synthetic trajectories for testing
    np.random.seed(42)

    n_poses = 100
    timestamps = np.arange(n_poses) * 1e8  # 100ms intervals in nanoseconds

    # Ground truth: circular trajectory
    t = np.linspace(0, 2 * np.pi, n_poses)
    radius = 10.0
    gt_positions = np.column_stack([
        radius * np.cos(t),
        radius * np.sin(t),
        np.zeros(n_poses)
    ])
    gt_orientations = np.tile([0, 0, 0, 1], (n_poses, 1))  # Identity rotation

    # Estimated: noisy version with drift
    drift = np.cumsum(np.random.randn(n_poses, 3) * 0.01, axis=0)
    est_positions = gt_positions + drift + np.random.randn(n_poses, 3) * 0.05
    est_orientations = gt_orientations + np.random.randn(n_poses, 4) * 0.01

    # Create trajectories
    traj_gt = Trajectory(timestamps, gt_positions, gt_orientations)
    traj_est = Trajectory(timestamps, est_positions, est_orientations)

    # Evaluate
    evaluator = TrajectoryEvaluator(traj_est, traj_gt, timestamp_unit="nanoseconds")

    evaluator.compute_ate(align=True)
    evaluator.compute_rpe(delta=10, delta_unit=DeltaUnit.FRAMES)

    evaluator.print_results("Synthetic Trajectory Test")

    # Plot
    plot_trajectory_comparison(traj_est, traj_gt, "Test Trajectories", show_plot=True)
