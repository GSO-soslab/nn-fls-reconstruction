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
# Coordinate Frame Conversions
# =============================================================================

def ned_to_enu_position(pos_ned: np.ndarray) -> np.ndarray:
    """
    Convert position from NED to ENU frame.

    NED: X=North, Y=East, Z=Down
    ENU: X=East, Y=North, Z=Up

    Args:
        pos_ned: Position(s) in NED frame, shape (3,) or (N, 3)

    Returns:
        Position(s) in ENU frame
    """
    if pos_ned.ndim == 1:
        return np.array([pos_ned[1], pos_ned[0], -pos_ned[2]])
    else:
        return np.column_stack([pos_ned[:, 1], pos_ned[:, 0], -pos_ned[:, 2]])


def enu_to_ned_position(pos_enu: np.ndarray) -> np.ndarray:
    """
    Convert position from ENU to NED frame.

    ENU: X=East, Y=North, Z=Up
    NED: X=North, Y=East, Z=Down

    Args:
        pos_enu: Position(s) in ENU frame, shape (3,) or (N, 3)

    Returns:
        Position(s) in NED frame
    """
    if pos_enu.ndim == 1:
        return np.array([pos_enu[1], pos_enu[0], -pos_enu[2]])
    else:
        return np.column_stack([pos_enu[:, 1], pos_enu[:, 0], -pos_enu[:, 2]])


def ned_to_enu_quaternion(q_ned: np.ndarray) -> np.ndarray:
    """
    Convert quaternion from NED to ENU frame.

    The rotation matrix R_ned_to_enu swaps X<->Y and negates Z:
        [0  1  0]
        [1  0  0]
        [0  0 -1]

    Args:
        q_ned: Quaternion(s) in NED frame [qx, qy, qz, qw], shape (4,) or (N, 4)

    Returns:
        Quaternion(s) in ENU frame
    """
    # Rotation from NED to ENU as quaternion
    # R = [[0,1,0],[1,0,0],[0,0,-1]] -> q = [0, 0, sqrt(2)/2, sqrt(2)/2] (90° around Z then flip)
    # Actually: swap X<->Y is 90° rotation around Z, then negate Z is 180° around X or Y
    # Simpler: just swap qx<->qy and negate qz
    if q_ned.ndim == 1:
        return np.array([q_ned[1], q_ned[0], -q_ned[2], q_ned[3]])
    else:
        return np.column_stack([q_ned[:, 1], q_ned[:, 0], -q_ned[:, 2], q_ned[:, 3]])


def enu_to_ned_quaternion(q_enu: np.ndarray) -> np.ndarray:
    """
    Convert quaternion from ENU to NED frame.

    Args:
        q_enu: Quaternion(s) in ENU frame [qx, qy, qz, qw], shape (4,) or (N, 4)

    Returns:
        Quaternion(s) in NED frame
    """
    # Same operation - it's symmetric
    if q_enu.ndim == 1:
        return np.array([q_enu[1], q_enu[0], -q_enu[2], q_enu[3]])
    else:
        return np.column_stack([q_enu[:, 1], q_enu[:, 0], -q_enu[:, 2], q_enu[:, 3]])


def convert_trajectory_ned_to_enu(traj: Trajectory) -> Trajectory:
    """
    Convert a trajectory from NED to ENU frame.

    Args:
        traj: Trajectory in NED frame

    Returns:
        Trajectory in ENU frame
    """
    return Trajectory(
        timestamps=traj.timestamps.copy(),
        positions=ned_to_enu_position(traj.positions),
        orientations=ned_to_enu_quaternion(traj.orientations)
    )


def convert_trajectory_enu_to_ned(traj: Trajectory) -> Trajectory:
    """
    Convert a trajectory from ENU to NED frame.

    Args:
        traj: Trajectory in ENU frame

    Returns:
        Trajectory in NED frame
    """
    return Trajectory(
        timestamps=traj.timestamps.copy(),
        positions=enu_to_ned_position(traj.positions),
        orientations=enu_to_ned_quaternion(traj.orientations)
    )


def ned_to_flu_position(pos_ned: np.ndarray) -> np.ndarray:
    """
    Convert position from NED to FLU (Front-Left-Up) frame.

    NED: X=North, Y=East, Z=Down
    FLU: X=Forward(North), Y=Left(-East), Z=Up(-Down)

    Transform: x_flu = x_ned, y_flu = -y_ned, z_flu = -z_ned

    Args:
        pos_ned: Position(s) in NED frame, shape (3,) or (N, 3)

    Returns:
        Position(s) in FLU frame
    """
    if pos_ned.ndim == 1:
        return np.array([pos_ned[0], -pos_ned[1], -pos_ned[2]])
    else:
        return np.column_stack([pos_ned[:, 0], -pos_ned[:, 1], -pos_ned[:, 2]])


def ned_to_flu_quaternion(q_ned: np.ndarray) -> np.ndarray:
    """
    Convert quaternion from NED to FLU frame.

    The rotation matrix R_ned_to_flu is:
        [1  0  0]
        [0 -1  0]
        [0  0 -1]

    This is a 180° rotation around the X-axis.

    Args:
        q_ned: Quaternion(s) in NED frame [qx, qy, qz, qw], shape (4,) or (N, 4)

    Returns:
        Quaternion(s) in FLU frame
    """
    # R_ned_to_flu is 180° rotation around X-axis
    # As quaternion: q_rot = [1, 0, 0, 0] (180° around X)
    # q_flu = q_rot * q_ned * q_rot^{-1}
    # For 180° around X: qx stays, qy negates, qz negates, qw stays
    if q_ned.ndim == 1:
        return np.array([q_ned[0], -q_ned[1], -q_ned[2], q_ned[3]])
    else:
        return np.column_stack([q_ned[:, 0], -q_ned[:, 1], -q_ned[:, 2], q_ned[:, 3]])


def convert_trajectory_ned_to_flu(traj: Trajectory) -> Trajectory:
    """
    Convert a trajectory from NED to FLU (Front-Left-Up) frame.

    Args:
        traj: Trajectory in NED frame

    Returns:
        Trajectory in FLU frame
    """
    return Trajectory(
        timestamps=traj.timestamps.copy(),
        positions=ned_to_flu_position(traj.positions),
        orientations=ned_to_flu_quaternion(traj.orientations)
    )


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
    save_path: Optional[str] = None,
    metrics: Optional[Dict] = None
):
    """
    Plot estimated vs reference trajectories in 2D (top-down XY view).

    Args:
        traj_est: Estimated trajectory
        traj_ref: Reference (ground truth) trajectory
        title: Plot title
        aligned: Whether trajectories were aligned
        show_plot: Whether to display the plot
        save_path: Path to save the plot image
        metrics: Optional dict with evaluation metrics to display on plot.
                 Expected keys: 'ate_translation', 'ate_rotation_deg', 'rpe_translation',
                 'duration_s', 'gt_distance_m', 'kitti'
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        warnings.warn("matplotlib not available, skipping plot")
        return

    fig, ax = plt.subplots(figsize=(12, 8))

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
    ax.legend(loc='upper left')
    ax.axis('equal')
    ax.grid(True, alpha=0.3)

    # Add metrics text box if provided
    if metrics is not None:
        metrics_text = _format_metrics_text(metrics, aligned)
        # Position text box in upper right
        props = dict(boxstyle='round,pad=0.5', facecolor='wheat', alpha=0.9)
        ax.text(0.98, 0.98, metrics_text, transform=ax.transAxes, fontsize=9,
                verticalalignment='top', horizontalalignment='right',
                bbox=props, fontfamily='monospace')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    if show_plot:
        plt.show()
    else:
        plt.close()


def _format_metrics_text(metrics: Dict, aligned: bool) -> str:
    """Format metrics dictionary into a display string for the plot."""
    lines = []

    # Header info
    if 'duration_s' in metrics:
        lines.append(f"Duration: {metrics['duration_s']:.1f} s")
    if 'gt_distance_m' in metrics:
        lines.append(f"Distance: {metrics['gt_distance_m']:.1f} m")
    if 'num_poses' in metrics:
        lines.append(f"Poses: {metrics['num_poses']}")

    lines.append("")  # blank line

    # ATE Translation
    if 'ate_translation' in metrics:
        ate_t = metrics['ate_translation']
        lines.append(f"ATE Translation {'(aligned)' if aligned else ''}:")
        lines.append(f"  RMSE: {ate_t['rmse']:.4f} m")
        lines.append(f"  Mean: {ate_t['mean']:.4f} m")
        lines.append(f"  Max:  {ate_t['max']:.4f} m")

    # ATE Rotation
    if 'ate_rotation_deg' in metrics:
        ate_r = metrics['ate_rotation_deg']
        lines.append(f"ATE Rotation:")
        lines.append(f"  RMSE: {ate_r['rmse']:.2f} deg")
        lines.append(f"  Mean: {ate_r['mean']:.2f} deg")

    lines.append("")  # blank line

    # RPE
    if 'rpe_translation' in metrics:
        rpe = metrics['rpe_translation']
        delta_str = f"{rpe.get('delta', '?')} {rpe.get('delta_unit', '?')}"
        lines.append(f"RPE ({delta_str}):")
        lines.append(f"  RMSE: {rpe['rmse']:.4f} m")
        lines.append(f"  Mean: {rpe['mean']:.4f} m")
        # Drift rate
        if rpe.get('delta_unit') == 'seconds' and rpe.get('delta', 0) > 0:
            drift_rate = rpe['rmse'] / rpe['delta']
            lines.append(f"  Drift: {drift_rate:.4f} m/s")
        elif rpe.get('delta_unit') == 'meters' and rpe.get('delta', 0) > 0:
            drift_pct = (rpe['rmse'] / rpe['delta']) * 100
            lines.append(f"  Drift: {drift_pct:.2f} %")

    # KITTI metrics
    if 'kitti' in metrics and metrics['kitti'] is not None:
        kitti = metrics['kitti']
        if 't_rel' in kitti and not np.isnan(kitti['t_rel']):
            lines.append("")
            lines.append("KITTI:")
            lines.append(f"  Trans: {kitti['t_rel']:.2f} %")
            lines.append(f"  Rot: {kitti['r_rel']:.2f} deg/100m")

    return "\n".join(lines)


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

def load_trajectories_from_npz(npz_path: str) -> Tuple[Trajectory, Trajectory]:
    """
    Load GT and TF trajectories from NPZ file saved by reconstruct script.

    Args:
        npz_path: Path to NPZ file

    Returns:
        (traj_tf, traj_gt): TF (estimated) and GT (reference) trajectories
    """
    data = np.load(npz_path)
    timestamps = data['timestamps']
    gt_positions = data['gt_positions']
    gt_orientations = data['gt_orientations']
    tf_positions = data['tf_positions']
    tf_orientations = data['tf_orientations']

    traj_gt = Trajectory(timestamps, gt_positions, gt_orientations)
    traj_tf = Trajectory(timestamps, tf_positions, tf_orientations)

    return traj_tf, traj_gt


def main():
    """CLI for trajectory metrics evaluation."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Evaluate trajectory metrics from saved NPZ file',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python trajectory_metrics.py traj.npz --plot
  python trajectory_metrics.py traj.npz --rpe-delta 1.0 --rpe-delta-unit seconds
  python trajectory_metrics.py traj.npz --no-align --save-plot traj_comparison.png
        """
    )
    parser.add_argument('npz_file', type=str, help='Path to trajectory NPZ file')
    parser.add_argument('--plot', action='store_true', help='Show trajectory comparison plot')
    parser.add_argument('--save-plot', type=str, default=None, help='Save plot to file')
    parser.add_argument('--no-align', action='store_true', help='Disable trajectory alignment')
    parser.add_argument('--rpe-delta', type=float, default=1.0, help='RPE delta interval')
    parser.add_argument('--rpe-delta-unit', type=str, default='seconds',
                       choices=['seconds', 'meters', 'frames'], help='RPE delta unit')

    args = parser.parse_args()

    # Load trajectories
    print(f"Loading trajectories from {args.npz_file}")
    traj_tf, traj_gt = load_trajectories_from_npz(args.npz_file)
    print(f"Loaded {len(traj_gt)} poses")

    # Create evaluator
    evaluator = TrajectoryEvaluator(traj_tf, traj_gt, max_time_diff=1.0,
                                     timestamp_unit="nanoseconds")

    # Compute metrics
    align = not args.no_align
    ate_trans = evaluator.compute_ate(PoseRelation.TRANSLATION_PART, align=align)
    ate_rot = evaluator.compute_ate(PoseRelation.ROTATION_ANGLE_DEG, align=align)

    delta_unit_map = {
        'seconds': DeltaUnit.SECONDS,
        'meters': DeltaUnit.METERS,
        'frames': DeltaUnit.FRAMES
    }
    delta_unit = delta_unit_map[args.rpe_delta_unit]

    try:
        rpe_trans = evaluator.compute_rpe(PoseRelation.TRANSLATION_PART,
                                          delta=args.rpe_delta, delta_unit=delta_unit)
    except ValueError as e:
        print(f"RPE computation failed: {e}")
        rpe_trans = None

    # Print results
    evaluator.print_results("Trajectory Evaluation")

    # Build metrics dict for plot
    gt_dist = traj_gt.get_distances()[-1]
    duration = (traj_gt.timestamps[-1] - traj_gt.timestamps[0]) / 1e9
    metrics = {
        'num_poses': len(traj_gt),
        'duration_s': duration,
        'gt_distance_m': gt_dist,
        'ate_translation': {
            'rmse': ate_trans.rmse,
            'mean': ate_trans.mean,
            'max': ate_trans.max,
        },
        'ate_rotation_deg': {
            'rmse': ate_rot.rmse,
            'mean': ate_rot.mean,
        }
    }
    if rpe_trans:
        metrics['rpe_translation'] = {
            'rmse': rpe_trans.rmse,
            'mean': rpe_trans.mean,
            'delta': args.rpe_delta,
            'delta_unit': args.rpe_delta_unit,
        }

    # Plot
    if args.plot or args.save_plot:
        plot_trajectory_comparison(
            evaluator.traj_est,
            evaluator.traj_ref,
            title="GT vs TF Trajectory Comparison",
            aligned=align,
            show_plot=args.plot,
            save_path=args.save_plot,
            metrics=metrics
        )


if __name__ == "__main__":
    main()
