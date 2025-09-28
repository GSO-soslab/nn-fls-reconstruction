import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np


def extract_specific_points_with_details(row_data, range_resolution, intensity_threshold, azimuth,
                                       point_indices=None, beam_indices=None, has_indices=False):
    """
    Extract specific points with detailed information for manual verification

    Args:
        row_data: CSV row data
        range_resolution: Range resolution parameter
        intensity_threshold: Intensity threshold
        azimuth: Azimuth angle
        point_indices: List of specific point indices to extract (0-667), if None extracts all
        beam_indices: List of specific beam indices (0-3) to extract for each point, if None extracts all
        has_indices: Whether the row has index column

    Returns:
        List of dictionaries with detailed point information
    """

    # Split CSV row into values
    if isinstance(row_data, str):
        values = row_data.split(',')
    else:
        values = row_data

    # Convert to float, handling empty/invalid values
    def safe_float(val):
        try:
            if isinstance(val, str):
                val = val.strip()
                if val == '' or val == 'nan':
                    return float('nan')
            return float(val)
        except (ValueError, TypeError):
            return float('nan')

    numeric_values = [safe_float(v) for v in values]

    # Extract intensities and phis based on column positions
    if has_indices:
        intensities = numeric_values[2:670]   # columns 2-669
        phis = numeric_values[3342:6014]      # columns 3342-6013
    else:
        intensities = numeric_values[1:669]   # columns 1-668
        phis = numeric_values[3341:6013]      # columns 3341-6012

    # If no specific points requested, use all points
    if point_indices is None:
        point_indices = list(range(668))

    # If no specific beams requested, use all beams
    if beam_indices is None:
        beam_indices = list(range(4))

    point_details = []

    for point_idx in point_indices:
        if point_idx >= 668:
            continue

        reverse_idx = 667 - point_idx

        # Check if we have valid intensity data
        if point_idx < len(intensities):
            intensity = intensities[point_idx]
        else:
            continue

        # Range calculation using the full range span
        max_range = 40.0
        min_range = 0.5
        range_val = min_range + (reverse_idx / 668) * (max_range - min_range)

        phi_start_idx = point_idx * 4

        for beam_idx in beam_indices:
            phi_idx = phi_start_idx + beam_idx

            # Check if phi index is within bounds
            if phi_idx < len(phis):
                phi_rad = phis[phi_idx]
            else:
                continue

            # Calculate 3D coordinates regardless of phi validity for verification
            x = range_val * np.cos(azimuth) * np.cos(phi_rad) if not pd.isna(phi_rad) else float('nan')
            y = range_val * np.sin(azimuth) * np.cos(phi_rad) if not pd.isna(phi_rad) else float('nan')
            z = range_val * np.sin(phi_rad) if not pd.isna(phi_rad) else float('nan')

            point_info = {
                'point_idx': point_idx,
                'reverse_idx': reverse_idx,
                'beam_idx': beam_idx,
                'phi_idx': phi_idx,
                'intensity': intensity,
                'phi_rad': phi_rad,
                'phi_deg': np.degrees(phi_rad) if not pd.isna(phi_rad) else float('nan'),
                'range_val': range_val,
                'azimuth_rad': azimuth,
                'azimuth_deg': np.degrees(azimuth),
                'x': x,
                'y': y,
                'z': z,
                'valid_phi': not pd.isna(phi_rad) and phi_rad not in [-10.0, -20.0],
                'above_intensity_threshold': not pd.isna(intensity) and intensity > intensity_threshold
            }

            point_details.append(point_info)

    return point_details


def analyze_specific_row(test_csv_path, original_csv_path, row_idx,
                        point_indices=None, beam_indices=None,
                        range_resolution=0.05988024, intensity_threshold=0.1, azimuth=0.0):
    """
    Analyze specific points from a specific row for manual verification

    Args:
        test_csv_path: Path to test CSV
        original_csv_path: Path to original CSV
        row_idx: Row index to analyze
        point_indices: List of point indices to extract (e.g., [100, 200, 300])
        beam_indices: List of beam indices to extract (e.g., [0, 1, 2, 3])
    """

    # Read both files as pure text lines
    with open(test_csv_path, 'r') as f:
        test_lines = [line.rstrip('\n\r') for line in f.readlines()]

    with open(original_csv_path, 'r') as f:
        original_lines = [line.rstrip('\n\r') for line in f.readlines()]

    # Find the test row corresponding to our desired original row index
    test_row_data = None
    for line in test_lines:
        if line.strip():
            parts = line.split(',', 1)
            if len(parts) >= 2:
                try:
                    if int(parts[0]) == row_idx:
                        test_row_data = parts[1]
                        break
                except ValueError:
                    continue

    if test_row_data is None:
        print(f"Row index {row_idx} not found in test data")
        return

    # Get original row data
    if row_idx >= len(original_lines):
        print(f"Row index {row_idx} out of range for original data")
        return

    original_row_data = original_lines[row_idx]

    print(f"\n=== ANALYZING ROW {row_idx} ===")
    print(f"Point indices: {point_indices if point_indices else 'All points (0-667)'}")
    print(f"Beam indices: {beam_indices if beam_indices else 'All beams (0-3)'}")
    print(f"Azimuth: {azimuth} rad ({np.degrees(azimuth):.2f} deg)")
    print(f"Intensity threshold: {intensity_threshold}")

    # Extract details from both original and test data
    original_details = extract_specific_points_with_details(
        original_row_data, range_resolution, intensity_threshold, azimuth,
        point_indices, beam_indices, has_indices=False
    )

    test_details = extract_specific_points_with_details(
        test_row_data, range_resolution, intensity_threshold, azimuth,
        point_indices, beam_indices, has_indices=False
    )

    print(f"\nFound {len(original_details)} points in original data")
    print(f"Found {len(test_details)} points in test data")

    # Display detailed information for manual verification
    print("\n" + "="*120)
    print(f"{'Src':<4} {'PtIdx':<5} {'BmIdx':<5} {'PhiIdx':<6} {'Intensity':<10} {'Phi(rad)':<10} {'Phi(deg)':<10} {'Range':<8} {'X':<10} {'Y':<10} {'Z':<10} {'Valid':<5}")
    print("="*120)

    # Display original points
    for detail in original_details[:20]:  # Limit to first 20 for readability
        print(f"{'Orig':<4} {detail['point_idx']:<5} {detail['beam_idx']:<5} {detail['phi_idx']:<6} "
              f"{detail['intensity']:<10.4f} {detail['phi_rad']:<10.4f} {detail['phi_deg']:<10.2f} "
              f"{detail['range_val']:<8.3f} {detail['x']:<10.4f} {detail['y']:<10.4f} {detail['z']:<10.4f} "
              f"{detail['valid_phi']:<5}")

    print("-"*120)

    # Display test points
    for detail in test_details[:20]:  # Limit to first 20 for readability
        print(f"{'Test':<4} {detail['point_idx']:<5} {detail['beam_idx']:<5} {detail['phi_idx']:<6} "
              f"{detail['intensity']:<10.4f} {detail['phi_rad']:<10.4f} {detail['phi_deg']:<10.2f} "
              f"{detail['range_val']:<8.3f} {detail['x']:<10.4f} {detail['y']:<10.4f} {detail['z']:<10.4f} "
              f"{detail['valid_phi']:<5}")

    # Manual calculation example for first valid point
    if original_details:
        print(f"\n=== MANUAL CALCULATION EXAMPLE ===")
        detail = original_details[0]
        print(f"For point_idx={detail['point_idx']}, beam_idx={detail['beam_idx']}:")
        print(f"  reverse_idx = 667 - {detail['point_idx']} = {detail['reverse_idx']}")
        print(f"  range_val = 0.5 + ({detail['reverse_idx']}/668) * (40.0-0.5) = {detail['range_val']:.6f}")
        print(f"  phi_idx = {detail['point_idx']} * 4 + {detail['beam_idx']} = {detail['phi_idx']}")
        print(f"  phi_rad = {detail['phi_rad']:.6f} ({detail['phi_deg']:.2f} degrees)")
        print(f"  azimuth = {azimuth:.6f} ({np.degrees(azimuth):.2f} degrees)")
        print(f"  x = {detail['range_val']:.6f} * cos({azimuth:.6f}) * cos({detail['phi_rad']:.6f}) = {detail['x']:.6f}")
        print(f"  y = {detail['range_val']:.6f} * sin({azimuth:.6f}) * cos({detail['phi_rad']:.6f}) = {detail['y']:.6f}")
        print(f"  z = {detail['range_val']:.6f} * sin({detail['phi_rad']:.6f}) = {detail['z']:.6f}")

    return original_details, test_details


def plot_specific_points(original_details, test_details, row_idx, output_dir="./specific_points"):
    """Plot the specific points for visualization"""
    os.makedirs(output_dir, exist_ok=True)

    plt.figure(figsize=(15, 10))

    # Plot original points
    orig_x = [d['x'] for d in original_details if d['valid_phi']]
    orig_z = [d['z'] for d in original_details if d['valid_phi']]

    # Plot test points
    test_x = [d['x'] for d in test_details if d['valid_phi']]
    test_z = [d['z'] for d in test_details if d['valid_phi']]

    if orig_x:
        plt.scatter(orig_x, orig_z, c='blue', s=50, alpha=0.7, label=f'Original Row {row_idx}', marker='o')

    if test_x:
        plt.scatter(test_x, test_z, c='red', s=30, alpha=0.7, label=f'Test Row {row_idx}', marker='x')

    plt.xlabel('X (meters)')
    plt.ylabel('Z (meters)')
    plt.title(f'Specific Points Analysis - Row {row_idx}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis('equal')

    plt.axhline(y=0, color='gray', linestyle='--', alpha=0.3)
    plt.axvline(x=0, color='gray', linestyle='--', alpha=0.3)

    plt.savefig(f"{output_dir}/specific_points_row_{row_idx}.png", dpi=300, bbox_inches='tight')
    plt.show()


def main():
    # Example usage: analyze specific points from row 25
    test_csv_path = "./data_splits/test_predictions_with_indices.csv"
    original_csv_path = "/Users/farhang/Downloads/fls_all_with_phi.csv"

    # Analyze specific points and beams from row 25
    row_idx = 25
    point_indices = [100, 200, 300, 400, 500]  # Analyze these specific point indices
    beam_indices = [0, 1, 2, 3]  # Analyze all beam indices for each point

    original_details, test_details = analyze_specific_row(
        test_csv_path=test_csv_path,
        original_csv_path=original_csv_path,
        row_idx=row_idx,
        point_indices=point_indices,
        beam_indices=beam_indices,
        azimuth=0.0
    )

    # Plot the specific points
    if original_details or test_details:
        plot_specific_points(original_details, test_details, row_idx)


if __name__ == "__main__":
    main()