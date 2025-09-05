#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import pandas as pd
import numpy as np
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header, Float32
import sensor_msgs_py.point_cloud2 as pc2
from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster
import argparse
import sys

class SonarPointCloudPublisher(Node):
    def __init__(self):
        super().__init__('sonar_pointcloud_publisher')
        
        # Declare parameters
        self.declare_parameter('original_csv', '/home/farhang/Downloads/fls_all_with_phi.csv')
        self.declare_parameter('predicted_csv', '/home/farhang/Downloads/fls_2d_terrain_prediction_output.csv')
        self.declare_parameter('frame_id', 'sonar_link')
        self.declare_parameter('publish_rate', 2.0)
        self.declare_parameter('azimuth', 0.0)
        self.declare_parameter('auto_play', True)
        self.declare_parameter('intensity_threshold', 0.1)
        self.declare_parameter('publish_both', True)
        
        # Get parameters
        self.original_csv = self.get_parameter('original_csv').get_parameter_value().string_value
        self.predicted_csv = self.get_parameter('predicted_csv').get_parameter_value().string_value
        self.frame_id = self.get_parameter('frame_id').get_parameter_value().string_value
        self.publish_rate = self.get_parameter('publish_rate').get_parameter_value().double_value
        self.azimuth = self.get_parameter('azimuth').get_parameter_value().double_value
        self.auto_play = self.get_parameter('auto_play').get_parameter_value().bool_value
        self.intensity_threshold = self.get_parameter('intensity_threshold').get_parameter_value().double_value
        self.publish_both = self.get_parameter('publish_both').get_parameter_value().bool_value
        
        # Publishers
        self.pc_original_pub = self.create_publisher(PointCloud2, 'sonar_pointcloud_original', 10)
        self.pc_predicted_pub = self.create_publisher(PointCloud2, 'sonar_pointcloud_predicted', 10)
        self.timestamp_pub = self.create_publisher(Float32, 'current_timestamp', 1)
        
        # TF broadcaster
        self.tf_broadcaster = TransformBroadcaster(self)
        
        # Load and preprocess data
        self.load_data()
        
        # State variables
        self.current_row = 0
        self.paused = False
        
        # Timer for automatic playback
        if self.auto_play:
            self.timer = self.create_timer(1.0/self.publish_rate, self.timer_callback)
        
        # Subscriber for manual control
        self.control_sub = self.create_subscription(
            Float32, 'pointcloud_control', self.control_callback, 10)
        
        self.get_logger().info(f"Sonar Point Cloud Publisher initialized:")
        self.get_logger().info(f"  Original CSV: {len(self.df_original)} rows")
        if self.publish_both:
            self.get_logger().info(f"  Predicted CSV: {len(self.df_predicted)} rows")
        self.get_logger().info(f"  Publishing at {self.publish_rate} Hz")
        self.get_logger().info(f"  Topics: 'sonar_pointcloud_original'" + 
                     (" and 'sonar_pointcloud_predicted'" if self.publish_both else ""))
        
        # Print control instructions
        self.get_logger().info("Control commands (publish to /pointcloud_control):")
        self.get_logger().info("  -1: pause/unpause")
        self.get_logger().info("  -2: reset to beginning")
        self.get_logger().info("  N: jump to row N")

    def load_data(self):
        """Load and preprocess both CSV files"""
        self.get_logger().info(f"Loading original sonar data from: {self.original_csv}")
        
        try:
            # Read original CSV
            self.df_original = pd.read_csv(self.original_csv)
            self.df_original = self.df_original.replace([-10, -20], np.nan)
            self.get_logger().info(f"Loaded {len(self.df_original)} original sonar measurements")
        except Exception as e:
            self.get_logger().error(f"Failed to load original CSV: {e}")
            sys.exit(1)
        
        # Read predicted CSV if requested
        if self.publish_both:
            try:
                self.get_logger().info(f"Loading predicted sonar data from: {self.predicted_csv}")
                self.df_predicted = pd.read_csv(self.predicted_csv)
                self.df_predicted = self.df_predicted.replace([-10, -20], np.nan)
                self.get_logger().info(f"Loaded {len(self.df_predicted)} predicted sonar measurements")
                
                # Check if they have same number of rows
                if len(self.df_original) != len(self.df_predicted):
                    self.get_logger().warn(f"Row count mismatch: Original={len(self.df_original)}, "
                                f"Predicted={len(self.df_predicted)}")
            except Exception as e:
                self.get_logger().error(f"Failed to load predicted CSV: {e}")
                self.get_logger().info("Continuing with original data only...")
                self.publish_both = False
                self.df_predicted = None
        else:
            self.df_predicted = None

    def initialize(self):
        """Initialize the node after parameters are set"""
        self.load_data()
        
        self.get_logger().info(f"Sonar Point Cloud Publisher initialized:")
        self.get_logger().info(f"  Original CSV: {len(self.df_original)} rows")
        if self.publish_both and self.df_predicted is not None:
            self.get_logger().info(f"  Predicted CSV: {len(self.df_predicted)} rows")
        self.get_logger().info(f"  Publishing at {self.publish_rate} Hz")
        self.get_logger().info(f"  Topics: 'sonar_pointcloud_original'" + 
                     (" and 'sonar_pointcloud_predicted'" if self.publish_both else ""))
        
        # Print control instructions
        self.get_logger().info("Control commands (publish to /pointcloud_control):")
        self.get_logger().info("  -1: pause/unpause")
        self.get_logger().info("  -2: reset to beginning")
        self.get_logger().info("  N: jump to row N")

    def csv_row_to_pointcloud(self, row_data, timestamp):
        """Convert a single CSV row to point cloud data"""
        row_idx = row_data.name
        
        # Extract the three data sections
        intensities = row_data.iloc[1:669].values  # Columns 1-668
        tangents = row_data.iloc[669:3341].values  # Columns 669-3340
        phis = row_data.iloc[3341:6013].values     # Columns 3341-6012
        
        # Reshape tangents and phis
        tangents_reshaped = tangents.reshape(4, 668)
        phis_reshaped = phis.reshape(4, 668)
        
        points = []
        
        # Process each measurement point
        for point_idx in range(668):
            intensity = intensities[point_idx]
            
            if not pd.isna(intensity) and intensity > self.intensity_threshold:
                # Process each of the 4 tangent/phi pairs
                for pair_idx in range(4):
                    tangent = tangents_reshaped[pair_idx, point_idx]
                    phi = phis_reshaped[pair_idx, point_idx]
                    
                    if not pd.isna(tangent) and not pd.isna(phi):
                        # Convert to Cartesian coordinates
                        range_val = intensity
                        
                        phi_rad = phi
                        tangent_rad = tangent
                        az_rad = self.azimuth
                        
                        # Spherical to Cartesian conversion
                        # x = range_val * np.cos(phi_rad) * np.cos(tangent_rad) * np.cos(az_rad)
                        # y = range_val * np.cos(phi_rad) * np.cos(tangent_rad) * np.sin(az_rad)
                        # z = range_val * np.sin(phi_rad)
                        
                        x = range_val * np.cos(phi_rad)
                        y = range_val * np.sin(az_rad)
                        z = range_val * np.sin(phi_rad)

                        # Normal from tangent
                        n_x = -tangent
                        n_y = 0.0
                        n_z = 1.0
                        norm = np.sqrt(n_x**2 + n_y**2 + n_z**2)
                        n_x /= norm
                        n_y /= norm
                        n_z /= norm

                        # if len(points) < 5:  # Print first few points for debugging
                        #     self.get_logger().info(f"Point {len(points)}: phi={phi:.3f}, tangent={tangent:.3f}, "
                        #                         f"x={x:.3f}, y={y:.3f}, z={z:.3f}")
                        # Add point with additional metadata
                        points.append([x, y, z, intensity, n_x, n_y, n_z, point_idx, pair_idx])

        return np.array(points)

    # def create_pointcloud2_msg(self, points, timestamp, data_type="original"):
    #     """Create a PointCloud2 message from points array"""
    #     header = Header()
    #     header.stamp = self.get_clock().now().to_msg()
    #     header.frame_id = self.frame_id
        
    #     # Just use XYZ points - simplest approach
    #     xyz_points = points[:, :3].astype(np.float32)
        
    #     # Create PointCloud2 message with just XYZ
    #     pc2_msg = pc2.create_cloud_xyz32(header, xyz_points)
        
    #     return pc2_msg

    # def create_pointcloud2_msg(self, points, timestamp, data_type="original"):
    #     """Create a PointCloud2 message from points array"""
    #     header = Header()
    #     header.stamp = self.get_clock().now().to_msg()
    #     header.frame_id = self.frame_id
        
    #     if data_type == "predicted":
    #         header.frame_id = "sonar_link_predicted"
    #     else:
    #         header.frame_id = "sonar_link_original"
            
    #     # Create structured array with XYZ + intensity
    #     dtype = [
    #         ('x', np.float32),
    #         ('y', np.float32), 
    #         ('z', np.float32),
    #         ('intensity', np.float32)
    #     ]
        

    #     # Convert to structured array
    #     structured_points = np.empty(len(points), dtype=dtype)
    #     structured_points['x'] = points[:, 0].astype(np.float32)
    #     structured_points['y'] = points[:, 1].astype(np.float32)
    #     structured_points['z'] = points[:, 2].astype(np.float32)
    #     structured_points['intensity'] = points[:, 3].astype(np.float32)  # intensity is column 3
        
    #     # Define fields for XYZI
    #     fields = [
    #         PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
    #         PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
    #         PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
    #         PointField(name='intensity', offset=12, datatype=PointField.FLOAT32, count=1)
    #     ]
        
        
    #     # Create PointCloud2 message
    #     pc2_msg = pc2.create_cloud(header, fields, structured_points)
        
    #     return pc2_msg
    
    def create_pointcloud2_msg(self, points, timestamp, data_type="original"):
        """Create a PointCloud2 message including normals"""
        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = self.frame_id

        if data_type == "predicted":
            header.frame_id = "sonar_link_predicted"
        else:
            header.frame_id = "sonar_link_original"

        # Create structured array with XYZ + intensity
        dtype = [
            ('x', np.float32), 
            ('y', np.float32), 
            ('z', np.float32),
            ('intensity', np.float32), 
            ('nx', np.float32), 
            ('ny', np.float32),
            ('nz', np.float32),
            ('point_idx', np.uint32), 
            ('pair_idx', np.uint32)
        ]

        # Define fields: XYZ + intensity + normals
        fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name='intensity', offset=12, datatype=PointField.FLOAT32, count=1),
            PointField(name='nx', offset=16, datatype=PointField.FLOAT32, count=1),
            PointField(name='ny', offset=20, datatype=PointField.FLOAT32, count=1),
            PointField(name='nz', offset=24, datatype=PointField.FLOAT32, count=1),
            PointField(name='point_idx', offset=28, datatype=PointField.UINT32, count=1),
            PointField(name='pair_idx', offset=32, datatype=PointField.UINT32, count=1),
        ]


        # Create structured array for point cloud
        structured_points = np.zeros(len(points), dtype=dtype)
        structured_points['x'] = points[:,0]
        structured_points['y'] = points[:,1]
        structured_points['z'] = points[:,2]
        structured_points['intensity'] = points[:,3]
        structured_points['nx'] = points[:,4]
        structured_points['ny'] = points[:,5]
        structured_points['nz'] = points[:,6]

        # Compute point_step and row_step
        point_step = 36  #7 floats × 4 bytes + 2 uint32 × 4 bytes
        row_step = point_step * len(points)

        # Flatten structured array to list of tuples
        flat_points = [tuple(p) for p in structured_points]

        pc2_msg = pc2.create_cloud(header, fields, flat_points)
        pc2_msg.height = 1
        pc2_msg.width = len(points)
        pc2_msg.is_dense = True
        pc2_msg.is_bigendian = False
        pc2_msg.point_step = point_step
        pc2_msg.row_step = row_step

        return pc2_msg

    def publish_tf_frame(self, timestamp):
        """Publish TF frame for the sonar"""
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = "base_link" 
        t.child_frame_id = self.frame_id
        
        # Set sonar position and orientation
        t.transform.translation.x = 0.0
        t.transform.translation.y = 0.0
        t.transform.translation.z = 0.0
        
        t.transform.rotation.x = 0.0
        t.transform.rotation.y = 0.0
        t.transform.rotation.z = 0.0
        t.transform.rotation.w = 1.0
        
        self.tf_broadcaster.sendTransform(t)

    def timer_callback(self):
        """Timer callback for automatic playback"""
        if not self.paused and self.current_row < len(self.df_original):
            self.publish_current_row()
            self.current_row += 1
        elif self.current_row >= len(self.df_original):
            self.get_logger().info("Reached end of data, restarting...")
            self.current_row = 0

    def control_callback(self, msg):
        """Control callback for manual navigation"""
        command = msg.data
        
        if command == -1:  # Pause/unpause
            self.paused = not self.paused
            self.get_logger().info(f"Playback {'paused' if self.paused else 'resumed'}")
        elif command == -2:  # Reset to beginning
            self.current_row = 0
            self.get_logger().info("Reset to beginning")
        elif 0 <= command < len(self.df_original):  # Jump to specific row
            self.current_row = int(command)
            self.get_logger().info(f"Jumped to row {self.current_row}")
            self.publish_current_row()

    def publish_current_row(self):
        """Publish point clouds for current row (both original and predicted)"""
        if self.current_row >= len(self.df_original):
            return
        
        # Get original data
        row_original = self.df_original.iloc[self.current_row]
        timestamp = row_original.iloc[0]  # First column is timestamp
        
        # Convert original to point cloud
        points_original = self.csv_row_to_pointcloud(row_original, timestamp)
        
        if len(points_original) > 0:
            # Create and publish original PointCloud2 message
            pc2_msg_original = self.create_pointcloud2_msg(points_original, timestamp, "original")
            self.pc_original_pub.publish(pc2_msg_original)
            
            log_msg = f"Published row {self.current_row}/{len(self.df_original)}, " \
                     f"timestamp: {timestamp:.3f}, original points: {len(points_original)}"
        else:
            self.get_logger().warn(f"No valid points in original row {self.current_row}")
            log_msg = f"Row {self.current_row}: No valid original points"
        
        # Handle predicted data if available
        if self.publish_both and self.df_predicted is not None and self.current_row < len(self.df_predicted):
            row_predicted = self.df_predicted.iloc[self.current_row]
            points_predicted = self.csv_row_to_pointcloud(row_predicted, timestamp)
            
            if len(points_predicted) > 0:
                # Create and publish predicted PointCloud2 message
                pc2_msg_predicted = self.create_pointcloud2_msg(points_predicted, timestamp, "predicted")
                self.pc_predicted_pub.publish(pc2_msg_predicted)
                log_msg += f", predicted points: {len(points_predicted)}"
            else:
                log_msg += ", predicted points: 0"
        
        # Publish timestamp and TF
        timestamp_msg = Float32()
        timestamp_msg.data = timestamp
        self.timestamp_pub.publish(timestamp_msg)
        self.publish_tf_frame(timestamp)
        
        # Throttled logging (every 5 seconds)
        if self.current_row % int(self.publish_rate * 5) == 0:
            self.get_logger().info(log_msg)

def main():
    parser = argparse.ArgumentParser(description='Sonar Point Cloud Publisher for ROS2')
    parser.add_argument('--original-csv', type=str, 
                       default='/home/farhang/Downloads/fls_all_with_phi.csv',
                       help='Path to original CSV file')
    parser.add_argument('--predicted-csv', type=str,
                       default='/home/farhang/Downloads/fls_2d_terrain_prediction_output.csv',
                       help='Path to predicted CSV file')
    parser.add_argument('--rate', type=float, default=2.0,
                       help='Publishing rate in Hz')
    parser.add_argument('--frame-id', type=str, default='sonar_link',
                       help='Frame ID for point clouds')
    parser.add_argument('--azimuth', type=float, default=0.0,
                       help='Sonar azimuth angle in degrees')
    parser.add_argument('--threshold', type=float, default=0.1,
                       help='Minimum intensity threshold')
    parser.add_argument('--original-only', action='store_true',
                       help='Only publish original data (skip predicted)')
    parser.add_argument('--manual', action='store_true',
                       help='Manual mode (no auto-play)')
    
    args = parser.parse_args()
    
    rclpy.init()
    
    # Create node with parameters from command line
    node = SonarPointCloudPublisher()
    
    # Update parameters from command line arguments
    node.original_csv = args.original_csv
    node.predicted_csv = args.predicted_csv
    node.publish_rate = args.rate
    node.frame_id = args.frame_id
    node.azimuth = args.azimuth
    node.intensity_threshold = args.threshold
    node.publish_both = not args.original_only
    node.auto_play = not args.manual
    
    # Initialize the node with updated parameters
    node.initialize()
    
    print("\n=== Sonar Point Cloud Publisher ===")
    print(f"Original CSV: {args.original_csv}")
    if not args.original_only:
        print(f"Predicted CSV: {args.predicted_csv}")
    print(f"Publishing rate: {args.rate} Hz")
    print(f"Frame ID: {args.frame_id}")
    print("\nTopics:")
    print("  /sonar_pointcloud_original")
    if not args.original_only:
        print("  /sonar_pointcloud_predicted")
    print("  /current_timestamp")
    print("\nControl (publish Float32 to /pointcloud_control):")
    print("  -1: pause/unpause")
    print("  -2: reset to beginning") 
    print("  N: jump to row N")
    print("\nPress Ctrl+C to stop\n")
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()