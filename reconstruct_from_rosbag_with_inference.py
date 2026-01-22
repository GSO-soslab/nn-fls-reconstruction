#!/usr/bin/env python3
"""
Reconstruct full FLS point cloud from ROS 2 bag files using neural network inference.
Extracts FLS images, runs inference to predict phi angles, uses TF for pose,
and builds 3D point cloud.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import argparse
from pathlib import Path
import torch

try:
    from mcap_ros2.reader import read_ros2_messages
except ImportError:
    print("ERROR: mcap libraries not found. Install with: pip install mcap mcap-ros2-support")
    exit(1)

# Import your model
from full_three_stage_model import FullThreeStageModelCNN


class FLSPointCloudReconstructor:
    def __init__(self, model_path, max_range=40.0, min_range=0.5, num_bins=668, num_beams=4, device='cpu'):
        self.max_range = max_range
        self.min_range = min_range
        self.num_bins = num_bins
        self.num_beams = num_beams
        self.device = device

        # Load trained model
        print(f"Loading model from {model_path}...")
        self.model = FullThreeStageModelCNN()
        checkpoint = torch.load(model_path, map_location=device)

        # Handle different checkpoint formats
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            print("WARNING: Loaded model with strict=False (some weights may not match)")
        else:
            # Checkpoint is the state_dict itself
            self.model.load_state_dict(checkpoint, strict=False)
            print("WARNING: Loaded model with strict=False (some weights may not match)")

        self.model.to(device)
        self.model.eval()
        print("Model loaded successfully")

        # Storage for point clouds
        self.points = []  # FLS points (x, y, z)
        self.intensities = []
        self.mbes_points = []  # MBES points (x, y, z)
        self.tf_tree = {}  # Store TF transforms
        self.gt_odom = []  # Store ground truth odometry messages

    def build_tf_tree(self, tf_messages):
        """
        Build TF tree from collected messages.
        tf_messages: list of TFMessage objects with timestamps

        TF convention: parent->child means the transform that takes a point
        from the child frame and expresses it in the parent frame.
        T_parent_child @ point_in_child = point_in_parent
        """
        for _, tf_msg in tf_messages:
            for transform in tf_msg.transforms:
                # Use transform's header timestamp, not bag log_time
                header_stamp = transform.header.stamp
                timestamp_ns = int(header_stamp.sec * 1e9 + header_stamp.nanosec)
                parent = transform.header.frame_id
                child = transform.child_frame_id
                key = f"{parent}->{child}"

                if key not in self.tf_tree:
                    self.tf_tree[key] = []

                self.tf_tree[key].append({
                    'timestamp': timestamp_ns,
                    'translation': np.array([
                        transform.transform.translation.x,
                        transform.transform.translation.y,
                        transform.transform.translation.z
                    ]),
                    'rotation': np.array([
                        transform.transform.rotation.x,
                        transform.transform.rotation.y,
                        transform.transform.rotation.z,
                        transform.transform.rotation.w
                    ])
                })

        # Sort transforms by timestamp for efficient lookup
        for key in self.tf_tree:
            self.tf_tree[key].sort(key=lambda x: x['timestamp'])

        print(f"Built TF tree with {len(self.tf_tree)} transform chains")
        for key in self.tf_tree:
            print(f"  {key}: {len(self.tf_tree[key])} transforms")

    def build_gt_odom_list(self, odom_messages):
        """
        Build ground truth odometry list from collected messages.
        odom_messages: list of (timestamp_ns, Odometry) tuples
        """
        for timestamp_ns, odom_msg in odom_messages:
            pose = odom_msg.pose.pose
            self.gt_odom.append({
                'timestamp': timestamp_ns,
                'position': np.array([
                    pose.position.x,
                    pose.position.y,
                    pose.position.z
                ]),
                'orientation': np.array([
                    pose.orientation.x,
                    pose.orientation.y,
                    pose.orientation.z,
                    pose.orientation.w
                ])
            })

        # Sort by timestamp
        self.gt_odom.sort(key=lambda x: x['timestamp'])
        print(f"Built ground truth odom list with {len(self.gt_odom)} messages")

    def lookup_gt_odom(self, timestamp):
        """
        Look up ground truth odometry at a given timestamp with interpolation.
        Returns (position, rotation_matrix) for base_link in world frame.
        """
        if len(self.gt_odom) == 0:
            return None, None

        # Find odom before and after timestamp
        before = None
        after = None

        for i, odom in enumerate(self.gt_odom):
            if odom['timestamp'] <= timestamp:
                before = odom
            if odom['timestamp'] >= timestamp:
                after = odom
                break

        if before is None and after is None:
            return None, None

        if before is None:
            pos = after['position']
            rot = self.quaternion_to_rotation_matrix(*after['orientation'])
            return pos, rot

        if after is None:
            pos = before['position']
            rot = self.quaternion_to_rotation_matrix(*before['orientation'])
            return pos, rot

        if before['timestamp'] == after['timestamp']:
            pos = before['position']
            rot = self.quaternion_to_rotation_matrix(*before['orientation'])
            return pos, rot

        # Interpolate
        dt = after['timestamp'] - before['timestamp']
        alpha = (timestamp - before['timestamp']) / dt

        pos = (1 - alpha) * before['position'] + alpha * after['position']
        q_interp = self.slerp_quaternion(before['orientation'], after['orientation'], alpha)
        rot = self.quaternion_to_rotation_matrix(*q_interp)

        return pos, rot

    def get_transform_with_gt_odom(self, from_frame, timestamp):
        """
        Get transform from sensor frame to world using ground truth odometry
        for base_link->world, and TF for static sensor->base_link chain.

        Returns (translation, rotation_matrix) that takes points from from_frame to world.
        """
        # Get base_link pose in world from ground truth odometry
        base_pos, base_rot = self.lookup_gt_odom(timestamp)
        if base_pos is None:
            print("WARNING: Could not find ground truth odom at timestamp")
            return np.array([0.0, 0.0, 0.0]), np.eye(3)

        # Get static transform chain: sensor -> nose_tip -> base_link
        if 'fls' in from_frame:
            sensor_child = 'mvp2_test_robot/fls_link_sf'
        elif 'mbes' in from_frame:
            sensor_child = 'mvp2_test_robot/mbes_link_sf'
        else:
            return base_pos, base_rot

        # nose_tip->sensor (static)
        t1, R1 = self.lookup_single_transform('mvp2_test_robot/nose_tip_link',
                                               sensor_child, timestamp)
        if t1 is None:
            t1, R1 = np.array([0.0, 0.0, 0.0]), np.eye(3)

        # base_link->nose_tip (static)
        t2, R2 = self.lookup_single_transform('mvp2_test_robot/base_link',
                                               'mvp2_test_robot/nose_tip_link', timestamp)
        if t2 is None:
            t2, R2 = np.array([0.0, 0.0, 0.0]), np.eye(3)

        # Chain: sensor -> nose_tip -> base_link
        R_base_sensor = R2 @ R1
        t_base_sensor = R2 @ t1 + t2

        # Apply base_link -> world from GT odom
        R_combined = base_rot @ R_base_sensor
        t_combined = base_rot @ t_base_sensor + base_pos

        return t_combined, R_combined

    def quaternion_to_rotation_matrix(self, qx, qy, qz, qw):
        """Convert quaternion to 3x3 rotation matrix"""
        R = np.array([
            [1 - 2*(qy**2 + qz**2), 2*(qx*qy - qw*qz), 2*(qx*qz + qw*qy)],
            [2*(qx*qy + qw*qz), 1 - 2*(qx**2 + qz**2), 2*(qy*qz - qw*qx)],
            [2*(qx*qz - qw*qy), 2*(qy*qz + qw*qx), 1 - 2*(qx**2 + qy**2)]
        ])
        return R

    def slerp_quaternion(self, q0, q1, t):
        """
        Spherical linear interpolation between quaternions.
        q0, q1: quaternions as [x, y, z, w]
        t: interpolation factor (0 = q0, 1 = q1)
        Returns interpolated quaternion [x, y, z, w]
        """
        # Normalize quaternions
        q0 = np.array(q0)
        q1 = np.array(q1)
        q0 = q0 / np.linalg.norm(q0)
        q1 = q1 / np.linalg.norm(q1)

        # Compute dot product
        dot = np.dot(q0, q1)

        # If dot is negative, negate one quaternion to take shorter path
        if dot < 0:
            q1 = -q1
            dot = -dot

        # If quaternions are very close, use linear interpolation
        if dot > 0.9995:
            result = q0 + t * (q1 - q0)
            return result / np.linalg.norm(result)

        # SLERP
        theta_0 = np.arccos(dot)
        theta = theta_0 * t

        q2 = q1 - q0 * dot
        q2 = q2 / np.linalg.norm(q2)

        result = q0 * np.cos(theta) + q2 * np.sin(theta)
        return result

    def lookup_single_transform(self, parent_frame, child_frame, timestamp):
        """
        Look up a single transform from the TF tree with interpolation.
        Returns (translation, rotation_matrix) for parent->child transform.
        This transform takes points from child frame to parent frame:
            point_in_parent = R @ point_in_child + t

        Uses linear interpolation for translation and SLERP for rotation
        to get accurate transforms at the exact sensor timestamp.
        """
        key = f"{parent_frame}->{child_frame}"

        if key not in self.tf_tree or len(self.tf_tree[key]) == 0:
            return None, None

        transforms = self.tf_tree[key]

        # Find transforms before and after timestamp for interpolation
        before = None
        after = None

        for i, tf in enumerate(transforms):
            if tf['timestamp'] <= timestamp:
                before = tf
            if tf['timestamp'] >= timestamp:
                after = tf
                break

        # Handle edge cases
        if before is None and after is None:
            return None, None

        if before is None:
            # Timestamp is before all transforms, use first one
            translation = after['translation']
            qx, qy, qz, qw = after['rotation']
            rotation = self.quaternion_to_rotation_matrix(qx, qy, qz, qw)
            return translation, rotation

        if after is None:
            # Timestamp is after all transforms, use last one
            translation = before['translation']
            qx, qy, qz, qw = before['rotation']
            rotation = self.quaternion_to_rotation_matrix(qx, qy, qz, qw)
            return translation, rotation

        # Check if we have the exact timestamp
        if before['timestamp'] == after['timestamp']:
            translation = before['translation']
            qx, qy, qz, qw = before['rotation']
            rotation = self.quaternion_to_rotation_matrix(qx, qy, qz, qw)
            return translation, rotation

        # Interpolate between before and after
        dt = after['timestamp'] - before['timestamp']
        alpha = (timestamp - before['timestamp']) / dt

        # Linear interpolation for translation
        translation = (1 - alpha) * before['translation'] + alpha * after['translation']

        # SLERP for rotation
        q_interp = self.slerp_quaternion(before['rotation'], after['rotation'], alpha)
        rotation = self.quaternion_to_rotation_matrix(*q_interp)

        return translation, rotation

    def get_transform_at_time(self, from_frame, to_frame, timestamp):
        """
        Get transform that takes a point from from_frame to to_frame.
        point_in_to_frame = R @ point_in_from_frame + t

        Chains through intermediate frames if needed.
        Returns (translation, rotation_matrix).
        """
        # Try direct lookup: to_frame->from_frame gives us what we need
        # Because TF stores parent->child, and we want from->to
        t, R = self.lookup_single_transform(to_frame, from_frame, timestamp)
        if t is not None:
            return t, R

        # print("Frame ID lookup failed:", from_frame, "to", to_frame, "at time", timestamp)

        # Try reverse lookup and invert
        # t, R = self.lookup_single_transform(from_frame, to_frame, timestamp)
        # if t is not None:
        #     R_inv = R.T
        #     t_inv = -R_inv @ t
        #     return t_inv, R_inv

        # Chain transforms for sensor -> world/odom
        # TF tree has: world->odom, odom->base_link, base_link->nose_tip_link,
        #              nose_tip_link->fls_link_sf, nose_tip_link->mbes_link_sf
        #
        # Each T_parent_child takes points from child to parent.

        if ('fls' in from_frame or 'mbes' in from_frame) and ('odom' in to_frame or 'world' in to_frame):
            # Full TF chain: sensor -> nose_tip -> base -> odom -> world
            if 'fls' in from_frame:
                sensor_child = 'mvp2_test_robot/fls_link_sf'
            else:
                sensor_child = 'mvp2_test_robot/mbes_link_sf'

            # Get nose_tip->sensor
            t1, R1 = self.lookup_single_transform('mvp2_test_robot/nose_tip_link',
                                                   sensor_child, timestamp)
            if t1 is None:
                t1, R1 = np.array([0.0, 0.0, 0.0]), np.eye(3)

            # Get base_link->nose_tip
            t2, R2 = self.lookup_single_transform('mvp2_test_robot/base_link',
                                                   'mvp2_test_robot/nose_tip_link', timestamp)
            if t2 is None:
                t2, R2 = np.array([0.0, 0.0, 0.0]), np.eye(3)

            # Get odom->base_link
            t3, R3 = self.lookup_single_transform('mvp2_test_robot/odom',
                                                   'mvp2_test_robot/base_link', timestamp)
            if t3 is None:
                print("WARNING: odom->base transform not found")
                return np.array([0.0, 0.0, 0.0]), np.eye(3)

            # Chain transforms: sensor -> nose_tip -> base -> odom
            R_combined = R3 @ R2 @ R1
            t_combined = R3 @ R2 @ t1 + R3 @ t2 + t3

            # Add world->odom if targeting world frame
            if 'world' in to_frame:
                t4, R4 = self.lookup_single_transform('mvp2_test_robot/world',
                                                       'mvp2_test_robot/odom', timestamp)
                if t4 is not None:
                    R_combined = R4 @ R_combined
                    t_combined = R4 @ t_combined + t4

            return t_combined, R_combined

        # Return identity transform if not found
        print(f"WARNING: Could not find transform {from_frame} -> {to_frame}")

        return np.array([0.0, 0.0, 0.0]), np.eye(3)

    def extract_azimuth_from_transform(self, rotation_matrix):
        """
        Extract azimuth angle (yaw) from rotation matrix.
        Azimuth is the rotation around the Z-axis.
        """
        # Extract yaw from rotation matrix: atan2(R[1,0], R[0,0])
        azimuth = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
        return azimuth

    def extract_pitch_from_transform(self, rotation_matrix):
        """
        Extract pitch angle from rotation matrix.
        Pitch is the rotation around the Y-axis.
        """
        # Extract pitch: asin(-R[2,0])
        pitch = np.arcsin(-np.clip(rotation_matrix[2, 0], -1.0, 1.0))
        return pitch

    def preprocess_image_for_inference(self, image_msg):
        """
        Convert ROS Image message to model input format.
        Model expects [batch, 668] intensities from a single range column.
        """
        # Decode image data
        height = image_msg.height
        width = image_msg.width
        encoding = image_msg.encoding
        data = np.frombuffer(image_msg.data, dtype=np.uint8)

        if encoding == 'mono8':
            # Grayscale image - use raw pixel values (0-255) as float32
            img = data.reshape((height, width)).astype(np.float32)
        elif encoding == 'rgb8':
            # RGB image - convert to grayscale, keep raw pixel values
            img = data.reshape((height, width, 3))
            img = np.mean(img, axis=2).astype(np.float32)
        else:
            raise ValueError(f"Unsupported encoding: {encoding}")

        # Extract center column (or mean across columns) as 668 intensities
        # The polar image has 668 range bins vertically
        if width > 1:
            # Take center column
            center_col = width // 2
            intensities = img[:, center_col]
        else:
            intensities = img.flatten()

        # Ensure we have exactly 668 values
        if len(intensities) != self.num_bins:
            raise ValueError(f"Expected {self.num_bins} range bins, got {len(intensities)}")

        # Convert to torch tensor [1, 668]
        tensor = torch.from_numpy(intensities).unsqueeze(0)

        return tensor

    def run_inference(self, image_tensor):
        """
        Run neural network inference on preprocessed image.
        Returns predicted phi angles for each range bin and beam.
        """
        with torch.no_grad():
            image_tensor = image_tensor.to(self.device)

            # Model returns: final_predictions [B, 2672], neg20_logits, valid_vs_neg10_logits, angle_preds
            final_predictions, _, _, _ = self.model(image_tensor)

            # final_predictions is [batch, 2672] where 2672 = 668 bins * 4 beams
            # Already contains phi angles (or -20.0, -10.0 for invalid)
            phi_angles = final_predictions.cpu().numpy().reshape(self.num_bins, self.num_beams)

        return phi_angles

    def reconstruct_points(self, phi_angles, position, rotation):
        """
        Convert phi angles to 3D points using position and rotation from TF.

        Args:
            phi_angles: [num_bins, num_beams] array of elevation angles
            position: [x, y, z] translation from sensor to world frame
            rotation: 3x3 rotation matrix from sensor to world frame
        """
        for bin_idx in range(self.num_bins):
            # Calculate range for this bin (reverse indexed)
            reverse_idx = self.num_bins - 1 - bin_idx
            range_val = self.min_range + (reverse_idx / self.num_bins) * (self.max_range - self.min_range)

            # Get phi angles for this bin (4 beams)
            phi_beams = phi_angles[bin_idx]

            # Use all valid beams
            for beam_idx in range(self.num_beams):
                phi_rad = phi_beams[beam_idx]

                # Filter out invalid predictions: -20.0 = no detection, -10.0 = uncertain
                if np.isnan(phi_rad) or phi_rad <= -10.0:
                    continue

                # Point in sensor frame (based on R1: Z is forward/beam, Y is vertical)
                # Phi is elevation angle from beam axis
                x_local = 0.0  # No lateral spread
                y_local = range_val * np.sin(phi_rad)  # Vertical in sensor frame
                z_local = range_val * np.cos(phi_rad)  # Forward along beam axis

                # Apply full TF transform (robot pose only)
                point_local = np.array([x_local, y_local, z_local])
                point_world = rotation @ point_local + position

                self.points.append(point_world)
                self.intensities.append(1.0)

    def process_laserscan(self, msg, position, rotation):
        """
        Process LaserScan message (MBES) and convert to 3D points.

        Args:
            msg: LaserScan ROS message
            position: [x, y, z] translation from sensor to world frame
            rotation: 3x3 rotation matrix from sensor to world frame
        """
        angle = msg.angle_min
        for r in msg.ranges:
            if msg.range_min < r < msg.range_max:
                # LaserScan points in sensor's XY plane
                x_sonar = r * np.cos(angle)
                y_sonar = r * np.sin(angle)
                z_sonar = 0.0

                # Transform to world frame using full TF transform
                point_sonar = np.array([x_sonar, y_sonar, z_sonar])
                point_world = rotation @ point_sonar + position

                self.mbes_points.append(point_world)

            angle += msg.angle_increment

    def process_bag(self, bag_path, fls_topic, mbes_topic, use_tf,
                    sonar_frame, mbes_frame, world_frame,
                    use_gt_odom=False, gt_odom_topic=None):
        """
        Process ROS 2 bag file (MCAP format) and extract FLS and MBES point clouds.

        Args:
            bag_path: Path to .mcap bag file
            fls_topic: Topic name for FLS image data
            mbes_topic: Topic name for MBES LaserScan data
            use_tf: Whether to use TF transforms
            sonar_frame: Frame ID of the FLS sonar
            mbes_frame: Frame ID of the MBES sensor
            world_frame: World frame ID (usually 'odom' or 'map')
            use_gt_odom: Whether to use ground truth odometry instead of TF for robot pose
            gt_odom_topic: Topic name for ground truth odometry (required if use_gt_odom=True)
        """
        print(f"Processing MCAP bag: {bag_path}")
        bag_path = Path(bag_path)

        tf_messages = []
        gt_odom_messages = []
        print("Sonar frame:", sonar_frame, "MBES frame:", mbes_frame, "World frame:", world_frame)
        if use_gt_odom:
            if gt_odom_topic is None:
                raise ValueError("gt_odom_topic must be specified when use_gt_odom=True")
            print(f"Using ground truth odometry from: {gt_odom_topic}")

        # First pass: collect TF data and/or ground truth odometry
        if use_tf or use_gt_odom:
            print("Collecting TF/odometry data...")
            with open(bag_path, 'rb') as f:
                for mcap_msg in read_ros2_messages(f):
                    if mcap_msg.channel.topic in ['/tf', '/tf_static']:
                        tf_messages.append((mcap_msg.log_time, mcap_msg.ros_msg))
                    elif use_gt_odom and mcap_msg.channel.topic == gt_odom_topic:
                        header_stamp = mcap_msg.ros_msg.header.stamp
                        timestamp_ns = int(header_stamp.sec * 1e9 + header_stamp.nanosec)
                        gt_odom_messages.append((timestamp_ns, mcap_msg.ros_msg))

            print(f"Collected {len(tf_messages)} TF messages")
            if len(tf_messages) > 0:
                self.build_tf_tree(tf_messages)
            else:
                print("WARNING: No TF messages found! Will use identity transforms for static frames.")

            if use_gt_odom:
                print(f"Collected {len(gt_odom_messages)} ground truth odometry messages")
                if len(gt_odom_messages) > 0:
                    self.build_gt_odom_list(gt_odom_messages)
                else:
                    print("WARNING: No ground truth odometry messages found!")

        # Second pass: process FLS and MBES messages
        print(f"Processing FLS messages from topic: {fls_topic}")
        print(f"Processing MBES messages from topic: {mbes_topic}")
        fls_count = 0
        mbes_count = 0

        with open(bag_path, 'rb') as f:
            for mcap_msg in read_ros2_messages(f):
                # Process FLS messages
                if mcap_msg.channel.topic == fls_topic:
                    # Preprocess image
                    image_tensor = self.preprocess_image_for_inference(mcap_msg.ros_msg)

                    # Run inference to get phi angles
                    phi_angles = self.run_inference(image_tensor)

                    # Get timestamp from message header
                    header_stamp = mcap_msg.ros_msg.header.stamp
                    timestamp_ns = int(header_stamp.sec * 1e9 + header_stamp.nanosec)

                    if use_gt_odom:
                        # Use ground truth odometry for base_link pose + TF for static sensor transforms
                        position, rotation = self.get_transform_with_gt_odom(sonar_frame, timestamp_ns)
                    elif use_tf:
                        # Use full TF chain
                        position, rotation = self.get_transform_at_time(
                            sonar_frame, world_frame, timestamp_ns
                        )
                    else:
                        # Assume static pose at origin
                        position = np.array([0.0, 0.0, 0.0])
                        rotation = np.eye(3)

                    # Reconstruct 3D points
                    self.reconstruct_points(phi_angles, position, rotation)
                    fls_count += 1

                    if fls_count % 10 == 0:
                        print(f"Processed {fls_count} FLS messages, {len(self.points)} points")

                # Process MBES messages
                elif mcap_msg.channel.topic == mbes_topic:
                    # Get timestamp from message header
                    header_stamp = mcap_msg.ros_msg.header.stamp
                    timestamp_ns = int(header_stamp.sec * 1e9 + header_stamp.nanosec)

                    if use_gt_odom:
                        # Use ground truth odometry for base_link pose + TF for static sensor transforms
                        position, rotation = self.get_transform_with_gt_odom(mbes_frame, timestamp_ns)
                    elif use_tf:
                        # Use full TF chain
                        position, rotation = self.get_transform_at_time(
                            mbes_frame, world_frame, timestamp_ns
                        )
                    else:
                        position = np.array([0.0, 0.0, 0.0])
                        rotation = np.eye(3)

                    self.process_laserscan(mcap_msg.ros_msg, position, rotation)
                    mbes_count += 1

                    if mbes_count % 100 == 0:
                        print(f"Processed {mbes_count} MBES messages, {len(self.mbes_points)} points")

        print(f"\nTotal FLS messages processed: {fls_count}")
        print(f"Total FLS points extracted: {len(self.points)}")
        print(f"Total MBES messages processed: {mbes_count}")
        print(f"Total MBES points extracted: {len(self.mbes_points)}")

    def get_point_cloud(self):
        """Return FLS point cloud as numpy array (N, 3)"""
        return np.array(self.points)

    def get_mbes_point_cloud(self):
        """Return MBES point cloud as numpy array (N, 3)"""
        return np.array(self.mbes_points)

    def save_ply(self, output_path):
        """Save point cloud to PLY file"""
        points = np.array(self.points)

        with open(output_path, 'w') as f:
            # Write header
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"element vertex {len(points)}\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            f.write("end_header\n")

            # Write points
            for point in points:
                f.write(f"{point[0]} {point[1]} {point[2]}\n")

        print(f"Saved point cloud to {output_path}")

    def save_mbes_ply(self, output_path):
        """Save MBES point cloud to PLY file"""
        if len(self.mbes_points) == 0:
            print("No MBES points to save!")
            return

        points = np.array(self.mbes_points)

        with open(output_path, 'w') as f:
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"element vertex {len(points)}\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            f.write("end_header\n")

            for point in points:
                f.write(f"{point[0]} {point[1]} {point[2]}\n")

        print(f"Saved MBES point cloud to {output_path}")

    def compute_roughness(self, points, radius=1.0, max_nn=50):
        """
        Compute roughness (surface variation) for each point using Open3D.
        Roughness = lambda_min / sum(lambdas)
        """
        if len(points) < max_nn:
            return np.zeros(len(points))

        try:
            import open3d as o3d
        except ImportError:
            print("WARNING: open3d not found, cannot compute roughness. Skipping filter.")
            return np.zeros(len(points)) + 999.0  # Return high roughness to keep all points

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        
        # Estimate covariances
        pcd.estimate_covariances(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn)
        )
        
        covariances = np.asarray(pcd.covariances)
        
        # Compute eigenvalues (vectorized)
        eigenvalues = np.linalg.eigvalsh(covariances)
        
        # eigenvalues are sorted in ascending order by eigvalsh
        sum_eigen = np.sum(eigenvalues, axis=1)
        # Avoid division by zero
        sum_eigen[sum_eigen == 0] = 1.0
        
        roughness = eigenvalues[:, 0] / sum_eigen

        return roughness

    def analyze_roughness_distribution(self, other_points=None, roughness_radius=1.0,
                                       plot=True, percentiles=[50, 75, 90, 95, 99]):
        """
        Analyze the roughness distribution of the point cloud to help select an appropriate
        threshold for filtering flat surfaces.

        This method computes roughness for all points and provides statistical analysis
        including histogram, percentiles, and cumulative distribution to inform threshold
        selection.

        Args:
            other_points: Optional external point cloud. If None, uses self.mbes_points.
            roughness_radius: Radius for roughness computation in meters (default 1.0m)
            plot: Whether to plot roughness distribution (default True)
            percentiles: List of percentiles to report (default [50, 75, 90, 95, 99])

        Returns:
            dict with roughness statistics and distribution data
        """
        mbes_pts = np.array(other_points) if other_points is not None else np.array(self.mbes_points)

        if len(mbes_pts) < 10:
            print("Not enough points for roughness analysis")
            return None

        print(f"\nComputing roughness for {len(mbes_pts)} points...")
        roughness = self.compute_roughness(mbes_pts, radius=roughness_radius)

        # Compute statistics
        stats = {
            'total_points': len(mbes_pts),
            'mean': np.mean(roughness),
            'std': np.std(roughness),
            'median': np.median(roughness),
            'min': np.min(roughness),
            'max': np.max(roughness),
            'roughness_values': roughness
        }

        # Compute percentiles
        stats['percentiles'] = {}
        for p in percentiles:
            stats['percentiles'][p] = np.percentile(roughness, p)

        # Print statistics
        print("\n" + "="*70)
        print("Roughness Distribution Analysis")
        print("="*70)
        print(f"Total points: {stats['total_points']}")
        print(f"Roughness radius: {roughness_radius:.2f} m")
        print(f"\nRoughness Statistics:")
        print(f"  Mean:   {stats['mean']:.6f}")
        print(f"  Std:    {stats['std']:.6f}")
        print(f"  Median: {stats['median']:.6f}")
        print(f"  Min:    {stats['min']:.6f}")
        print(f"  Max:    {stats['max']:.6f}")

        print(f"\nPercentiles:")
        for p in percentiles:
            pct_val = stats['percentiles'][p]
            points_below = np.sum(roughness < pct_val)
            pct_below = points_below / len(roughness) * 100
            print(f"  {p}th percentile: {pct_val:.6f} ({points_below} points, {pct_below:.1f}%)")

        # Suggested thresholds based on distribution
        print(f"\nSuggested Thresholds (based on percentiles):")
        print(f"  Conservative (filter ~50%): {stats['percentiles'][50]:.6f}")
        print(f"  Moderate (filter ~75%):     {stats['percentiles'][75]:.6f}")
        print(f"  Aggressive (filter ~90%):   {stats['percentiles'][90]:.6f}")

        # Count points in typical ranges
        print(f"\nRoughness Range Analysis:")
        ranges = [(0, 0.01), (0.01, 0.03), (0.03, 0.05), (0.05, 0.10), (0.10, 0.33)]
        for r_min, r_max in ranges:
            count = np.sum((roughness >= r_min) & (roughness < r_max))
            pct = count / len(roughness) * 100
            print(f"  {r_min:.2f} - {r_max:.2f}: {count:6d} points ({pct:5.1f}%)")

        print("="*70)

        if plot:
            self._plot_roughness_distribution(roughness, stats, percentiles)

        return stats

    def _plot_roughness_distribution(self, roughness, stats, percentiles):
        """Plot roughness distribution histogram and cumulative distribution."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Histogram
        axes[0].hist(roughness, bins=100, edgecolor='black', alpha=0.7)
        axes[0].axvline(stats['mean'], color='red', linestyle='--', linewidth=2, label=f"Mean: {stats['mean']:.4f}")
        axes[0].axvline(stats['median'], color='green', linestyle='--', linewidth=2, label=f"Median: {stats['median']:.4f}")

        # Add percentile lines
        colors = ['orange', 'purple', 'brown']
        for i, p in enumerate([75, 90, 95]):
            if p in stats['percentiles']:
                axes[0].axvline(stats['percentiles'][p], color=colors[i % len(colors)],
                              linestyle=':', linewidth=1.5, label=f"{p}th: {stats['percentiles'][p]:.4f}")

        axes[0].set_xlabel('Roughness (λ_min / Σλ)')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title('Roughness Distribution')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Cumulative distribution
        sorted_roughness = np.sort(roughness)
        cumulative = np.arange(1, len(sorted_roughness) + 1) / len(sorted_roughness) * 100

        axes[1].plot(sorted_roughness, cumulative, linewidth=2)
        axes[1].axhline(50, color='red', linestyle='--', linewidth=1, alpha=0.5)
        axes[1].axhline(75, color='orange', linestyle='--', linewidth=1, alpha=0.5)
        axes[1].axhline(90, color='purple', linestyle='--', linewidth=1, alpha=0.5)
        axes[1].axhline(95, color='brown', linestyle='--', linewidth=1, alpha=0.5)

        axes[1].set_xlabel('Roughness (λ_min / Σλ)')
        axes[1].set_ylabel('Cumulative Percentage (%)')
        axes[1].set_title('Cumulative Distribution Function')
        axes[1].grid(True, alpha=0.3)
        axes[1].set_xlim(left=0)

        plt.tight_layout()
        plt.show()

    def compute_map_to_map_error_roman_singh(self, bin_size=0.5, other_points=None):
        """
        Compute map-to-map error between FLS and MBES point clouds using the
        Roman & Singh consistency-based method.

        Reference:
            C. Roman and H. Singh, "Consistency Based Error Evaluation for Deep Sea
            Bathymetric Mapping with Robotic Vehicles," 2006 IEEE/RSJ International
            Conference on Intelligent Robots and Systems.

        This method directly compares overlapping regions from different source maps
        by binning the space and computing mean elevation differences per bin,
        avoiding the bin-size bias issues of PCA-based planar fitting.

        Args:
            bin_size: Size of spatial bins in meters (default 0.5m)
            other_points: Optional external point cloud. If None, uses self.mbes_points.

        Returns:
            dict with error metrics and spatial error grid, or None if insufficient data
        """
        fls_pts = np.array(self.points)
        mbes_pts = np.array(other_points) if other_points is not None else np.array(self.mbes_points)

        if len(fls_pts) < 10 or len(mbes_pts) < 10:
            print("Not enough points for map-to-map comparison")
            return None

        # Find overlapping XY region
        fls_min = np.min(fls_pts[:, :2], axis=0)
        fls_max = np.max(fls_pts[:, :2], axis=0)
        mbes_min = np.min(mbes_pts[:, :2], axis=0)
        mbes_max = np.max(mbes_pts[:, :2], axis=0)

        # Overlap bounds
        xy_min = np.maximum(fls_min, mbes_min)
        xy_max = np.minimum(fls_max, mbes_max)

        if np.any(xy_min >= xy_max):
            print("No overlapping region between FLS and MBES point clouds")
            return None

        # Create bin grid
        nx = int(np.ceil((xy_max[0] - xy_min[0]) / bin_size))
        ny = int(np.ceil((xy_max[1] - xy_min[1]) / bin_size))

        if nx < 1 or ny < 1:
            print("Overlap region too small for binning")
            return None

        # Initialize error grid
        error_grid = np.full((ny, nx), np.nan)
        bin_counts = np.zeros((ny, nx), dtype=int)

        # Bin FLS points
        fls_binned = {}
        for pt in fls_pts:
            bx = int((pt[0] - xy_min[0]) / bin_size)
            by = int((pt[1] - xy_min[1]) / bin_size)
            if 0 <= bx < nx and 0 <= by < ny:
                key = (bx, by)
                if key not in fls_binned:
                    fls_binned[key] = []
                fls_binned[key].append(pt[2])  # Store Z values

        # Bin MBES points
        mbes_binned = {}
        for pt in mbes_pts:
            bx = int((pt[0] - xy_min[0]) / bin_size)
            by = int((pt[1] - xy_min[1]) / bin_size)
            if 0 <= bx < nx and 0 <= by < ny:
                key = (bx, by)
                if key not in mbes_binned:
                    mbes_binned[key] = []
                mbes_binned[key].append(pt[2])  # Store Z values

        # Compute map-to-map error for each bin with points from both maps
        errors = []
        for key in fls_binned:
            if key in mbes_binned:
                fls_z = np.array(fls_binned[key])
                mbes_z = np.array(mbes_binned[key])

                # Mean Z difference (signed error)
                fls_mean_z = np.mean(fls_z)
                mbes_mean_z = np.mean(mbes_z)
                z_error = np.abs(fls_mean_z - mbes_mean_z)

                bx, by = key
                error_grid[by, bx] = z_error
                bin_counts[by, bx] = len(fls_z) + len(mbes_z)
                errors.append(z_error)

        if len(errors) == 0:
            print("No overlapping bins found")
            return None

        errors = np.array(errors)

        metrics = {
            'method': 'Roman & Singh (2006)',
            'mean_error': np.mean(errors),
            'std_error': np.std(errors),
            'median_error': np.median(errors),
            'max_error': np.max(errors),
            'min_error': np.min(errors),
            'rmse': np.sqrt(np.mean(errors**2)),
            'num_bins': len(errors),
            'total_bins': nx * ny,
            'coverage': len(errors) / (nx * ny),
            'bin_size': bin_size,
            'error_grid': error_grid,
            'bin_counts': bin_counts,
            'xy_min': xy_min,
            'xy_max': xy_max
        }

        return metrics

    def compute_map_to_map_error_with_roughness_filter(self, bin_size=0.5, other_points=None,
                                                         min_roughness=0.05, roughness_radius=1.0):
        """
        Compute map-to-map error with eigenvalue-based roughness filtering to exclude
        flat/planar surfaces and focus evaluation on geometrically complex regions.

        This method extends the Roman & Singh binning approach by first filtering out
        points with low surface roughness (flat regions like seafloor, tank bottom, walls)
        that may artificially reduce error metrics without testing reconstruction quality
        on complex geometry.

        Surface roughness is computed using eigenvalue decomposition of local point
        neighborhoods, following the approach in:
            - "Overall Filtering Algorithm for Multiscale Noise Removal From Point Cloud Data,"
              IEEE Access, 2021.
            - "Eigen-Factors: Plane Estimation for Multi-Frame and Time-Continuous Point
              Cloud Alignment," IROS 2019.

        Roughness metric: λ_min / (λ_0 + λ_1 + λ_2)
            - Low values (→0): Planar/flat surfaces
            - High values (→0.33): Complex/rough surfaces

        Args:
            bin_size: Size of spatial bins in meters (default 0.5m)
            other_points: Optional external point cloud. If None, uses self.mbes_points.
            min_roughness: Minimum roughness threshold to retain a point (default 0.05).
                          Points with roughness < threshold are filtered out.
            roughness_radius: Radius for local neighborhood roughness computation (default 1.0m)

        Returns:
            dict with error metrics and spatial error grid, or None if insufficient data
        """
        fls_pts = np.array(self.points)
        mbes_pts = np.array(other_points) if other_points is not None else np.array(self.mbes_points)

        if len(fls_pts) < 10 or len(mbes_pts) < 10:
            print("Not enough points for map-to-map comparison")
            return None

        # Apply eigenvalue-based roughness filtering to MBES (ground truth)
        print(f"Filtering flat surfaces (roughness < {min_roughness})...")
        roughness = self.compute_roughness(mbes_pts, radius=roughness_radius)
        mask = roughness >= min_roughness

        original_count = len(mbes_pts)
        mbes_pts = mbes_pts[mask]
        filtered_count = original_count - len(mbes_pts)
        print(f"Filtered out {filtered_count} flat surface points ({filtered_count/original_count*100:.1f}%)")
        print(f"Retained {len(mbes_pts)} complex surface points for error evaluation")

        if len(mbes_pts) < 10:
            print("Too few complex points remaining after roughness filtering!")
            return None

        # Find overlapping XY region (after filtering)
        fls_min = np.min(fls_pts[:, :2], axis=0)
        fls_max = np.max(fls_pts[:, :2], axis=0)
        mbes_min = np.min(mbes_pts[:, :2], axis=0)
        mbes_max = np.max(mbes_pts[:, :2], axis=0)

        # Overlap bounds
        xy_min = np.maximum(fls_min, mbes_min)
        xy_max = np.minimum(fls_max, mbes_max)

        if np.any(xy_min >= xy_max):
            print("No overlapping region between FLS and filtered MBES point clouds")
            return None

        # Create bin grid
        nx = int(np.ceil((xy_max[0] - xy_min[0]) / bin_size))
        ny = int(np.ceil((xy_max[1] - xy_min[1]) / bin_size))

        if nx < 1 or ny < 1:
            print("Overlap region too small for binning")
            return None

        # Initialize error grid
        error_grid = np.full((ny, nx), np.nan)
        bin_counts = np.zeros((ny, nx), dtype=int)

        # Bin FLS points
        fls_binned = {}
        for pt in fls_pts:
            bx = int((pt[0] - xy_min[0]) / bin_size)
            by = int((pt[1] - xy_min[1]) / bin_size)
            if 0 <= bx < nx and 0 <= by < ny:
                key = (bx, by)
                if key not in fls_binned:
                    fls_binned[key] = []
                fls_binned[key].append(pt[2])  # Store Z values

        # Bin filtered MBES points
        mbes_binned = {}
        for pt in mbes_pts:
            bx = int((pt[0] - xy_min[0]) / bin_size)
            by = int((pt[1] - xy_min[1]) / bin_size)
            if 0 <= bx < nx and 0 <= by < ny:
                key = (bx, by)
                if key not in mbes_binned:
                    mbes_binned[key] = []
                mbes_binned[key].append(pt[2])  # Store Z values

        # Compute map-to-map error for each bin with points from both maps
        errors = []
        for key in fls_binned:
            if key in mbes_binned:
                fls_z = np.array(fls_binned[key])
                mbes_z = np.array(mbes_binned[key])

                # Mean Z difference
                fls_mean_z = np.mean(fls_z)
                mbes_mean_z = np.mean(mbes_z)
                z_error = np.abs(fls_mean_z - mbes_mean_z)

                bx, by = key
                error_grid[by, bx] = z_error
                bin_counts[by, bx] = len(fls_z) + len(mbes_z)
                errors.append(z_error)

        if len(errors) == 0:
            print("No overlapping bins found after filtering")
            return None

        errors = np.array(errors)

        metrics = {
            'method': 'Roman & Singh + Eigenvalue Roughness Filter',
            'mean_error': np.mean(errors),
            'std_error': np.std(errors),
            'median_error': np.median(errors),
            'max_error': np.max(errors),
            'min_error': np.min(errors),
            'rmse': np.sqrt(np.mean(errors**2)),
            'num_bins': len(errors),
            'total_bins': nx * ny,
            'coverage': len(errors) / (nx * ny),
            'bin_size': bin_size,
            'min_roughness': min_roughness,
            'roughness_radius': roughness_radius,
            'points_filtered': filtered_count,
            'points_retained': len(mbes_pts),
            'filter_percentage': filtered_count / original_count * 100,
            'error_grid': error_grid,
            'bin_counts': bin_counts,
            'xy_min': xy_min,
            'xy_max': xy_max
        }

        return metrics

    def print_map_to_map_error(self, method='roman_singh', bin_size=0.5, other_points=None,
                                min_roughness=0.05, roughness_radius=1.0, plot=False):
        """
        Print map-to-map error results and optionally plot error distribution.

        Args:
            method: Error computation method, one of:
                   - 'roman_singh': Pure Roman & Singh baseline (default)
                   - 'roughness_filter': Roman & Singh + eigenvalue roughness filtering
                   - 'both': Compute and compare both methods
            bin_size: Size of spatial bins in meters (default 0.5m)
            other_points: Optional external point cloud. If None, uses self.mbes_points.
            min_roughness: Minimum roughness threshold for 'roughness_filter' method (default 0.05)
            roughness_radius: Radius for roughness computation (default 1.0m)
            plot: Whether to plot error distribution (default False)

        Returns:
            metrics dict(s) - single dict for one method, or tuple of (roman_singh, roughness_filter) for 'both'
        """
        if method == 'roman_singh':
            metrics = self.compute_map_to_map_error_roman_singh(bin_size, other_points)
            if metrics is None:
                return None

            print("\n" + "="*70)
            print(f"Map-to-Map Error: {metrics['method']}")
            print("="*70)
            print(f"\nBin size: {metrics['bin_size']:.2f} m")
            print(f"Overlapping bins: {metrics['num_bins']} / {metrics['total_bins']} ({metrics['coverage']*100:.1f}% coverage)")
            print(f"\nZ-Error Statistics (meters):")
            print(f"  Mean:   {metrics['mean_error']:.4f}")
            print(f"  Std:    {metrics['std_error']:.4f}")
            print(f"  Median: {metrics['median_error']:.4f}")
            print(f"  Min:    {metrics['min_error']:.4f}")
            print(f"  Max:    {metrics['max_error']:.4f}")
            print(f"  RMSE:   {metrics['rmse']:.4f}")
            print("="*70)

            if plot:
                self._plot_error_grid(metrics)

            return metrics

        elif method == 'roughness_filter':
            metrics = self.compute_map_to_map_error_with_roughness_filter(
                bin_size, other_points, min_roughness, roughness_radius
            )
            if metrics is None:
                return None

            print("\n" + "="*70)
            print(f"Map-to-Map Error: {metrics['method']}")
            print("="*70)
            print(f"\nRoughness Filtering:")
            print(f"  Threshold: {metrics['min_roughness']:.3f}")
            print(f"  Radius: {metrics['roughness_radius']:.2f} m")
            print(f"  Filtered: {metrics['points_filtered']} points ({metrics['filter_percentage']:.1f}%)")
            print(f"  Retained: {metrics['points_retained']} complex surface points")
            print(f"\nBin size: {metrics['bin_size']:.2f} m")
            print(f"Overlapping bins: {metrics['num_bins']} / {metrics['total_bins']} ({metrics['coverage']*100:.1f}% coverage)")
            print(f"\nZ-Error Statistics (meters):")
            print(f"  Mean:   {metrics['mean_error']:.4f}")
            print(f"  Std:    {metrics['std_error']:.4f}")
            print(f"  Median: {metrics['median_error']:.4f}")
            print(f"  Min:    {metrics['min_error']:.4f}")
            print(f"  Max:    {metrics['max_error']:.4f}")
            print(f"  RMSE:   {metrics['rmse']:.4f}")
            print("="*70)

            if plot:
                self._plot_error_grid(metrics)

            return metrics

        elif method == 'both':
            # Compute both methods
            metrics_rs = self.compute_map_to_map_error_roman_singh(bin_size, other_points)
            metrics_rf = self.compute_map_to_map_error_with_roughness_filter(
                bin_size, other_points, min_roughness, roughness_radius
            )

            if metrics_rs is None and metrics_rf is None:
                print("Both methods failed to compute metrics")
                return None

            # Print side-by-side comparison
            print("\n" + "="*70)
            print("Map-to-Map Error Comparison")
            print("="*70)

            if metrics_rs is not None:
                print(f"\n{metrics_rs['method']}:")
                print(f"  Bins:   {metrics_rs['num_bins']} / {metrics_rs['total_bins']} ({metrics_rs['coverage']*100:.1f}% coverage)")
                print(f"  Mean:   {metrics_rs['mean_error']:.4f} m")
                print(f"  Median: {metrics_rs['median_error']:.4f} m")
                print(f"  RMSE:   {metrics_rs['rmse']:.4f} m")
            else:
                print(f"\n{metrics_rs['method']}: FAILED")

            if metrics_rf is not None:
                print(f"\n{metrics_rf['method']}:")
                print(f"  Filtered: {metrics_rf['points_filtered']} points ({metrics_rf['filter_percentage']:.1f}%)")
                print(f"  Bins:   {metrics_rf['num_bins']} / {metrics_rf['total_bins']} ({metrics_rf['coverage']*100:.1f}% coverage)")
                print(f"  Mean:   {metrics_rf['mean_error']:.4f} m")
                print(f"  Median: {metrics_rf['median_error']:.4f} m")
                print(f"  RMSE:   {metrics_rf['rmse']:.4f} m")
            else:
                print(f"\n{metrics_rf['method']}: FAILED")

            # Print difference if both succeeded
            if metrics_rs is not None and metrics_rf is not None:
                print(f"\nDifference (Roughness - Baseline):")
                print(f"  ΔRMSE:   {metrics_rf['rmse'] - metrics_rs['rmse']:+.4f} m ({(metrics_rf['rmse']/metrics_rs['rmse']-1)*100:+.1f}%)")
                print(f"  ΔMean:   {metrics_rf['mean_error'] - metrics_rs['mean_error']:+.4f} m")
                print(f"  ΔMedian: {metrics_rf['median_error'] - metrics_rs['median_error']:+.4f} m")

            print("="*70)

            if plot and metrics_rf is not None:
                self._plot_error_grid(metrics_rf)

            return (metrics_rs, metrics_rf)

        else:
            raise ValueError(f"Unknown method '{method}'. Use 'roman_singh', 'roughness_filter', or 'both'")

    def _plot_error_grid(self, metrics):
        """Plot the spatial error distribution."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Error heatmap
        im = axes[0].imshow(metrics['error_grid'], origin='lower', cmap='hot',
                           extent=[metrics['xy_min'][0], metrics['xy_max'][0],
                                  metrics['xy_min'][1], metrics['xy_max'][1]])
        axes[0].set_xlabel('X (m)')
        axes[0].set_ylabel('Y (m)')
        axes[0].set_title('Map-to-Map Z-Error (m)')
        plt.colorbar(im, ax=axes[0], label='Error (m)')

        # Error histogram
        valid_errors = metrics['error_grid'][~np.isnan(metrics['error_grid'])]
        axes[1].hist(valid_errors, bins=30, edgecolor='black', alpha=0.7)
        axes[1].axvline(metrics['mean_error'], color='r', linestyle='--', label=f"Mean: {metrics['mean_error']:.3f}m")
        axes[1].axvline(metrics['median_error'], color='g', linestyle='--', label=f"Median: {metrics['median_error']:.3f}m")
        axes[1].set_xlabel('Z-Error (m)')
        axes[1].set_ylabel('Bin Count')
        axes[1].set_title('Error Distribution')
        axes[1].legend()

        plt.tight_layout()
        plt.show()

    def visualize_3d(self, subsample=1):
        """Visualize point cloud in 3D"""
        points = np.array(self.points)

        if len(points) == 0:
            print("No points to visualize!")
            return

        # Subsample for visualization
        points = points[::subsample]

        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')

        ax.scatter(points[:, 0], points[:, 1], points[:, 2],
                  c=points[:, 2], cmap='viridis', s=0.1, alpha=0.6)

        ax.set_xlabel('X (meters)')
        ax.set_ylabel('Y (meters)')
        ax.set_zlabel('Z (meters)')
        ax.set_title(f'FLS Point Cloud ({len(points)} points)')
        ax.axis('equal')

        plt.show()

    def visualize_2d_projections(self, subsample=1):
        """Visualize XY, XZ, YZ projections"""
        points = np.array(self.points)

        if len(points) == 0:
            print("No points to visualize!")
            return

        points = points[::subsample]

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        # XY projection (top view)
        axes[0].scatter(points[:, 0], points[:, 1], c='blue', s=0.1, alpha=0.5)
        axes[0].set_xlabel('X (meters)')
        axes[0].set_ylabel('Y (meters)')
        axes[0].set_title('Top View (XY)')
        axes[0].axis('equal')
        axes[0].grid(True, alpha=0.3)

        # XZ projection (side view)
        axes[1].scatter(points[:, 0], points[:, 2], c='green', s=0.1, alpha=0.5)
        axes[1].set_xlabel('X (meters)')
        axes[1].set_ylabel('Z (meters)')
        axes[1].set_title('Side View (XZ)')
        axes[1].axis('equal')
        axes[1].grid(True, alpha=0.3)

        # YZ projection (front view)
        axes[2].scatter(points[:, 1], points[:, 2], c='red', s=0.1, alpha=0.5)
        axes[2].set_xlabel('Y (meters)')
        axes[2].set_ylabel('Z (meters)')
        axes[2].set_title('Front View (YZ)')
        axes[2].axis('equal')
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()


def main():
    parser = argparse.ArgumentParser(description='Reconstruct FLS and MBES point clouds from ROS 2 bag with NN inference')
    parser.add_argument('bag_path', type=str, help='Path to ROS 2 .mcap bag file')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--topic', type=str, default='/mvp2_test_robot/fls/data/image',
                       help='FLS image topic name')
    parser.add_argument('--mbes-topic', type=str, default='/mvp2_test_robot/mbes/data',
                       help='MBES LaserScan topic name')
    parser.add_argument('--no-tf', action='store_true',
                       help='Disable TF usage, assume identity transforms')
    parser.add_argument('--use-gt-odom', action='store_true',
                       help='Use ground truth odometry topic for robot pose instead of TF')
    parser.add_argument('--gt-odom-topic', type=str,
                       help='Ground truth odometry topic (required with --use-gt-odom)')
    parser.add_argument('--sonar-frame', type=str, default='mvp2_test_robot/fls_link_ros',
                       help='FLS sonar frame ID')
    parser.add_argument('--mbes-frame', type=str, default='mvp2_test_robot/mbes_link_sf',
                       help='MBES sensor frame ID')
    parser.add_argument('--world-frame', type=str, default='mvp2_test_robot/world',
                       help='World frame ID (fixed reference frame)')
    parser.add_argument('--output', type=str, help='Output PLY file path for FLS')
    parser.add_argument('--mbes-output', type=str, help='Output PLY file path for MBES')
    parser.add_argument('--visualize', choices=['3d', '2d', 'both'],
                       help='Visualization mode')
    parser.add_argument('--subsample', type=int, default=1,
                       help='Subsampling factor for visualization')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'],
                       help='Device to run inference on')
    parser.add_argument('--compare', action='store_true',
                       help='Run map-to-map error comparison between FLS and MBES')
    parser.add_argument('--error-method', type=str, default='roman_singh',
                       choices=['roman_singh', 'roughness_filter', 'both'],
                       help='Error evaluation method: roman_singh (baseline), roughness_filter (with filtering), or both (default: roman_singh)')
    parser.add_argument('--bin-size', type=float, default=0.5,
                       help='Bin size in meters for map-to-map error (default: 0.5)')
    parser.add_argument('--min-roughness', type=float, default=0.05,
                       help='Minimum roughness threshold for roughness_filter method (default: 0.05)')
    parser.add_argument('--roughness-radius', type=float, default=1.0,
                       help='Radius for roughness computation in meters (default: 1.0)')
    parser.add_argument('--plot-error', action='store_true',
                       help='Plot spatial error distribution')
    parser.add_argument('--analyze-roughness', action='store_true',
                       help='Analyze roughness distribution to help select threshold')


    args = parser.parse_args()

    # Create reconstructor
    reconstructor = FLSPointCloudReconstructor(
        model_path=args.model,
        device=args.device
    )

    # Process bag (this will detect sensor mount pitch on first message)
    reconstructor.process_bag(
        args.bag_path,
        fls_topic=args.topic,
        mbes_topic=args.mbes_topic,
        use_tf=not args.no_tf,
        sonar_frame=args.sonar_frame,
        mbes_frame=args.mbes_frame,
        world_frame=args.world_frame,
        use_gt_odom=args.use_gt_odom,
        gt_odom_topic=args.gt_odom_topic
    )

    # Save FLS to PLY if requested
    if args.output:
        reconstructor.save_ply(args.output)

    # Save MBES to PLY if requested
    if args.mbes_output:
        reconstructor.save_mbes_ply(args.mbes_output)

    # Analyze roughness distribution if requested
    if args.analyze_roughness:
        reconstructor.analyze_roughness_distribution(
            other_points=None,  # Use internal MBES points
            roughness_radius=args.roughness_radius,
            plot=True
        )

    # Run map-to-map error comparison if requested
    if args.compare:
        reconstructor.print_map_to_map_error(
            method=args.error_method,
            bin_size=args.bin_size,
            other_points=None,  # Use internal MBES points
            min_roughness=args.min_roughness,
            roughness_radius=args.roughness_radius,
            plot=args.plot_error
        )

    # Visualize if requested
    if args.visualize in ['3d', 'both']:
        reconstructor.visualize_3d(subsample=args.subsample)

    if args.visualize in ['2d', 'both']:
        reconstructor.visualize_2d_projections(subsample=args.subsample)


if __name__ == "__main__":
    main()
