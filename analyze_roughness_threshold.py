#!/usr/bin/env python3
"""
Script to analyze roughness distribution and help select appropriate threshold.

Usage:
    python analyze_roughness_threshold.py <bag_file> --model <model_path>

This will:
1. Load MBES point cloud from the bag
2. Compute roughness for all points
3. Show distribution statistics and suggested thresholds
4. Plot histogram and cumulative distribution
5. Help you choose an informed threshold value

Then you can run the full comparison with your chosen threshold:
    python reconstruct_from_rosbag_with_inference.py <bag_file> --model <model_path> \\
        --compare --error-method both --min-roughness <YOUR_THRESHOLD> --plot-error
"""

import sys
import argparse
from reconstruct_from_rosbag_with_inference import FLSPointCloudReconstructor


def main():
    parser = argparse.ArgumentParser(
        description='Analyze roughness distribution to select filtering threshold',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Analyze roughness distribution with default radius (1.0m)
    python analyze_roughness_threshold.py data.mcap --model model.pth

    # Use different roughness computation radius
    python analyze_roughness_threshold.py data.mcap --model model.pth --roughness-radius 0.5

    # After analysis, run comparison with chosen threshold
    python reconstruct_from_rosbag_with_inference.py data.mcap --model model.pth \\
        --compare --error-method both --min-roughness 0.03 --plot-error
        """
    )

    parser.add_argument('bag_path', type=str, help='Path to ROS 2 .mcap bag file')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--topic', type=str, default='/mvp2_test_robot/fls/data/image',
                       help='FLS image topic name')
    parser.add_argument('--mbes-topic', type=str, default='/mvp2_test_robot/mbes/data',
                       help='MBES LaserScan topic name')
    parser.add_argument('--no-tf', action='store_true',
                       help='Disable TF usage, assume identity transforms')
    parser.add_argument('--sonar-frame', type=str, default='mvp2_test_robot/fls_link_ros',
                       help='FLS sonar frame ID')
    parser.add_argument('--mbes-frame', type=str, default='mvp2_test_robot/mbes_link_sf',
                       help='MBES sensor frame ID')
    parser.add_argument('--world-frame', type=str, default='mvp2_test_robot/world',
                       help='World frame ID (fixed reference frame)')
    parser.add_argument('--roughness-radius', type=float, default=1.0,
                       help='Radius for roughness computation in meters (default: 1.0)')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'],
                       help='Device to run inference on')

    args = parser.parse_args()

    print("="*70)
    print("Roughness Distribution Analysis Tool")
    print("="*70)
    print(f"\nThis will help you select an appropriate roughness threshold")
    print(f"for filtering flat surfaces from error evaluation.\n")

    # Create reconstructor
    reconstructor = FLSPointCloudReconstructor(
        model_path=args.model,
        device=args.device
    )

    # Process bag to get MBES points
    print(f"Processing bag: {args.bag_path}")
    reconstructor.process_bag(
        args.bag_path,
        fls_topic=args.topic,
        mbes_topic=args.mbes_topic,
        use_tf=not args.no_tf,
        sonar_frame=args.sonar_frame,
        mbes_frame=args.mbes_frame,
        world_frame=args.world_frame
    )

    # Analyze roughness distribution
    stats = reconstructor.analyze_roughness_distribution(
        other_points=None,  # Use internal MBES points
        roughness_radius=args.roughness_radius,
        plot=True
    )

    if stats is not None:
        print("\n" + "="*70)
        print("Recommended Next Steps:")
        print("="*70)
        print("\n1. Review the plots showing roughness distribution")
        print("2. Choose a threshold based on:")
        print("   - What % of points you want to filter (see percentiles)")
        print("   - Whether you want conservative (50%), moderate (75%), or aggressive (90%) filtering")
        print("\n3. Run full error comparison with your chosen threshold:")
        print(f"\n   python reconstruct_from_rosbag_with_inference.py {args.bag_path} \\")
        print(f"       --model {args.model} \\")
        print(f"       --compare --error-method both \\")
        print(f"       --min-roughness <YOUR_THRESHOLD> \\")
        print(f"       --plot-error")
        print("\n4. For your paper, justify the threshold by:")
        print("   - Reporting the distribution statistics shown above")
        print("   - Explaining that X% of points (flat surfaces) were filtered")
        print("   - Citing the eigenvalue decomposition papers for the metric formula")
        print("="*70)


if __name__ == "__main__":
    main()
