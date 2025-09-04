import pandas as pd
import numpy as np

def csv_to_pointcloud(csv_file, output_file, azimuth=0):
    """
    Convert CSV data to point cloud format for CloudCompare.
    Converts -10 and -20 values to NaN before processing.
    """

    # Read CSV
    df = pd.read_csv(csv_file)

    # Replace -10 and -20 with NaN in the entire dataframe
    df = df.replace([-10, -20], np.nan)

    points = []

    for row_idx, row in df.iterrows():
        timestamp = row.iloc[0]  # First column is timestamp

        # Extract the three data sections
        intensities = row.iloc[1:669].values  # Columns 1-668 (668 values)
        tangents = row.iloc[669:3341].values  # Columns 669-3340 (2672 values = 668×4)
        phis = row.iloc[3341:6013].values     # Columns 3341-6012 (2672 values = 668×4)

        # Reshape tangents and phis back to [4, 668] structure
        tangents_reshaped = tangents.reshape(4, 668)  # 4 sets of 668 tangents
        phis_reshaped = phis.reshape(4, 668)          # 4 sets of 668 phis

        # Process each of the 668 measurement points
        for point_idx in range(668):
            intensity = intensities[point_idx]

            if not pd.isna(intensity) and intensity > 0:  # Skip NaN and invalid measurements

                # Process each of the 4 tangent/phi pairs for this measurement point
                for pair_idx in range(4):
                    tangent = tangents_reshaped[pair_idx, point_idx]
                    phi = phis_reshaped[pair_idx, point_idx]

                    if not pd.isna(tangent) and not pd.isna(phi):
                        # Convert spherical coordinates to Cartesian
                        range_val = intensity

                        # Convert angles to radians
                        phi_rad = np.radians(phi)
                        tangent_rad = np.radians(tangent)
                        az_rad = np.radians(azimuth)

                        # Spherical to Cartesian conversion
                        x = range_val * np.cos(phi_rad) * np.cos(tangent_rad) * np.cos(az_rad)
                        y = range_val * np.cos(phi_rad) * np.cos(tangent_rad) * np.sin(az_rad)
                        z = range_val * np.sin(phi_rad)

                        points.append([x, y, z, intensity, row_idx, point_idx, pair_idx, timestamp])

    # Convert to numpy array
    points = np.array(points)

    # Save as different formats
    if output_file.endswith('.xyz'):
        np.savetxt(output_file, points[:, :3], fmt='%.6f', delimiter=' ')
    elif output_file.endswith('.ply'):
        save_ply(output_file, points)
    else:
        np.savetxt(output_file, points, fmt='%.6f', delimiter=' ',
                   header='X Y Z Intensity RowIndex PointIndex PairIndex Timestamp', comments='')

def save_ply(filename, points):
    """Save points in PLY format with all metadata"""
    with open(filename, 'w') as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property float intensity\n")
        f.write("property int row_index\n")
        f.write("property int point_index\n")
        f.write("property int pair_index\n")
        f.write("property float timestamp\n")
        f.write("end_header\n")

        for point in points:
            f.write(f"{point[0]:.6f} {point[1]:.6f} {point[2]:.6f} {point[3]:.6f} {int(point[4])} {int(point[5])} {int(point[6])} {point[7]:.6f}\n")

def create_comparison_pointclouds(original_csv, predicted_csv, output_dir="./"):
    """
    Create point clouds from both original and predicted data for comparison.
    """
    import os

    os.makedirs(output_dir, exist_ok=True)

    original_output = os.path.join(output_dir, "original_sonar_data.ply")
    csv_to_pointcloud(original_csv, original_output, azimuth=0)

    predicted_output = os.path.join(output_dir, "predicted_sonar_data.ply")
    csv_to_pointcloud(predicted_csv, predicted_output, azimuth=0)

# Usage
if __name__ == "__main__":
    original_csv = "/Users/farhang/Downloads/fls_all_with_phi.csv"
    predicted_csv = "/Users/farhang/Downloads/fls_2d_terrain_prediction_output.csv"

    create_comparison_pointclouds(original_csv, predicted_csv, output_dir="./pointclouds/")
