import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist
from tqdm import tqdm

# ── CSV layout constants ──────────────────────────────────────────────────────
N_BINS      = 468  # number of range bins
N_BEAMS     = 4    # reflections (tangents/phis) per bin
# Col 0                              : timestamp       (original CSV)
# Col 0 = row_index, Col 1           : timestamp       (test CSV with indices)
# Cols 1           .. N_BINS         : intensities     (N_BINS values)
# Cols N_BINS+1    .. (1+N_BEAMS)*N_BINS   : tangents (N_BEAMS per bin)
# Cols (1+N_BEAMS)*N_BINS+1 .. (1+2*N_BEAMS)*N_BINS   : phis (N_BEAMS per bin)
_INT_START  = 1
_INT_END    = 1 + N_BINS                           # 469
_PHI_START  = 1 + (1 + N_BEAMS) * N_BINS          # 2341
_PHI_END    = 1 + (1 + 2 * N_BEAMS) * N_BINS      # 4213
# ─────────────────────────────────────────────────────────────────────────────

# Set global font sizes for all plots
plt.rcParams.update({
    'font.size': 25,           # Base font size
    'axes.labelsize': 20,      # X and Y labels
    'axes.titlesize': 22,      # Plot titles
    'xtick.labelsize': 20,     # X-axis tick labels
    'ytick.labelsize': 20,     # Y-axis tick labels
    'legend.fontsize': 20,     # Legend text
    'figure.titlesize': 30     # Figure title
})

# only specified non nan values to extract
# def extract_pcl_points_from_row(row_data, range_resolution, intensity_threshold, azimuth, has_indices=False):
#     """Extract x,z points using correct range calculation - pure text processing"""

#     # Split CSV row into values
#     if isinstance(row_data, str):
#         values = row_data.split(',')
#     else:
#         # Handle case where it's already a list
#         print("-----It is a list -----")
#         values = row_data

#     # Convert to float, handling empty/invalid values
#     def safe_float(val):
#         try:
#             if isinstance(val, str):
#                 val = val.strip()
#                 if val == '' or val == 'nan':
#                     return float('nan')
#             return float(val)
#         except (ValueError, TypeError):
#             return float('nan')

#     numeric_values = [safe_float(v) for v in values]
#     # print(len(numeric_values))
#     # print("First 10 cols:", numeric_values[:10])
#     # print("Around 3335–3345:", numeric_values[3335:3345])
#     # print("Around 6005–6015:", numeric_values[6005:6015])

#     # Extract intensities and phis based on column positions
#     if has_indices:
#         # Skip first column (index) in test data
#         intensities = numeric_values[2:670]  # columns 2-669
#         # print(intensities[:5])
#         phis = numeric_values[3342:6014]     # columns 3342-6013
#         print(phis[439:445])
#         # Save a row to another csv to compare only for reading csv verification
#         print("HAs Indcice")
#     else:
#         intensities = numeric_values[1:669]  # columns 1-668
#         # print(intensities[:5])
#         phis = numeric_values[3341:6013]     # columns 3341-6012
#         # print(phis[320:340])
#         print(phis[439:445])
#         print("No Indcice")



#     x_points = []
#     z_points = []

#     for point_idx in range(668):
#         reverse_idx = 667 - point_idx

#         # # Check if we have valid intensity data
#         # if point_idx < len(intensities):
#         #     intensity = intensities[point_idx]
#         # else:
#         #     continue

#         # if not pd.isna(intensity) and intensity > intensity_threshold:

#         # Range calculation using the full range span
#         max_range = 40.0
#         min_range = 0.5
#         range_val = min_range + (reverse_idx / 668) * (max_range - min_range)

#         phi_start_idx = point_idx * 4
#         # print(point_idx)
#         for beam_idx in range(4):
#             phi_idx = phi_start_idx + beam_idx

#             # Check if phi index is within bounds
#             if phi_idx < len(phis):
#                 # print(beam_idx)
#                 phi_rad = phis[phi_idx]
#                 # print("THISSSS")
#             else:
#                 # print("THAATTT")
#                 continue

#             if not pd.isna(phi_rad) and phi_rad not in [-10.0, -20.0]:
#                 # Calculate 3D coordinates
#                 x = range_val * np.cos(azimuth) * np.cos(phi_rad)
#                 y = range_val * np.sin(azimuth) * np.cos(phi_rad)
#                 z = range_val * np.sin(phi_rad)

#                 x_points.append(x)
#                 z_points.append(z)
#     # print(len(x_points))
#     return x_points, z_points

def extract_pcl_points_from_row(row_data, range_resolution, intensity_threshold, azimuth, has_indices=False):
    """Extract x,z points using correct range calculation - pure text processing"""

    # Split CSV row into values
    if isinstance(row_data, str):
        values = row_data.split(',')
    else:
        # Handle case where it's already a list
        print("-----It is a list -----")
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
    # test CSV has an extra row_index prepended, so shift by 1
    offset = 1 if has_indices else 0
    intensities = numeric_values[offset + _INT_START : offset + _INT_END]
    phis        = numeric_values[offset + _PHI_START : offset + _PHI_END]

    x_points = []
    z_points = []

    for point_idx in range(1, N_BINS):
        reverse_idx = N_BINS - point_idx

        # Range calculation using the full range span
        max_range = 40.0
        min_range = 0.0
        range_val = (max_range * reverse_idx) / N_BINS
        # # Take only the FIRST valid phi from the 4 beams
        # phi_start_idx = point_idx * 4
        # phi_rad = -20.0

        # for beam_idx in range(4):
        #     phi_idx = phi_start_idx + beam_idx
        #     if phi_idx < len(phis):
        #         temp_phi = phis[phi_idx]
        #         if not pd.isna(temp_phi) and abs(temp_phi + 10.0) > 0.01 and abs(temp_phi + 20.0) > 0.01:
        #             phi_rad = temp_phi
        #             break

        # # Calculate 3D coordinates using the first valid phi only
        # if abs(phi_rad + 10.0) > 0.01 and abs(phi_rad + 20.0) > 0.01:
        #     x = range_val * np.cos(azimuth) * np.cos(phi_rad)
        #     y = range_val * np.sin(azimuth) * np.cos(phi_rad)
        #     z = range_val * np.sin(phi_rad)

        #     x_points.append(x)
        #     z_points.append(z)
        # else:
        #     # Append NaN to maintain index alignment for distance calculations
        #     x_points.append(float('nan'))
        #     z_points.append(float('nan'))


            ## Temp
            # x = phi_rad
            # z= phi_rad

        # # Take ALL the valid phis from the 4 beams
        phi_start_idx = point_idx * N_BEAMS

        for beam_idx in range(N_BEAMS):
            phi_idx = phi_start_idx + beam_idx

            if phi_idx < len(phis):
                phi_rad = phis[phi_idx]
                if pd.isna(phi_rad):
                    phi_rad = -20.0
            else:
                phi_rad = -20.0

            # Calculate 3D coordinates for all points, use NaN for invalid phi values
            if phi_rad != -10.0 and phi_rad != -20.0:
                x = range_val * np.cos(azimuth) * np.cos(phi_rad)
                y = range_val * np.sin(azimuth) * np.cos(phi_rad)
                z = range_val * np.sin(phi_rad)

                x_points.append(x)
                z_points.append(z)
            else:
                # Append NaN to maintain index alignment for distance calculations
                x_points.append(float('nan'))
                z_points.append(float('nan'))

    # print(f"Total points generated: {len(x_points)} (expected: {668*4})")
    return x_points, z_points

def compute_chamfer_distance(pred_x, pred_z, gt_x, gt_z):
    """
    Compute Chamfer distance using point-to-point correspondence (not nearest neighbor).
    Both arrays should have same length (2672 values).
    Only compute distances where both pred and GT are valid (not -10 or -20).

    Args:
        pred_x, pred_z: Arrays of predicted x, z coordinates (length 2672)
        gt_x, gt_z: Arrays of ground truth x, z coordinates (length 2672)

    Returns:
        chamfer_dist: Mean Euclidean distance at corresponding indices
    """
    pred_x = np.array(pred_x, dtype=float)
    pred_z = np.array(pred_z, dtype=float)
    gt_x = np.array(gt_x, dtype=float)
    gt_z = np.array(gt_z, dtype=float)

    # Find valid points where BOTH pred and GT are valid (not -10 or -20)
    pred_valid = (np.abs(pred_x + 10.0) > 0.01) & (np.abs(pred_x + 20.0) > 0.01) & \
                 (np.abs(pred_z + 10.0) > 0.01) & (np.abs(pred_z + 20.0) > 0.01)

    gt_valid = (np.abs(gt_x + 10.0) > 0.01) & (np.abs(gt_x + 20.0) > 0.01) & \
               (np.abs(gt_z + 10.0) > 0.01) & (np.abs(gt_z + 20.0) > 0.01)

    # Only compute distance where both are valid
    both_valid = pred_valid & gt_valid

    if not np.any(both_valid):
        return float('nan')

    # Compute Euclidean distances at corresponding indices (point-to-point)
    distances = np.sqrt((pred_x[both_valid] - gt_x[both_valid])**2 +
                       (pred_z[both_valid] - gt_z[both_valid])**2)

    # Chamfer: mean of point-to-point distances
    chamfer_dist = np.mean(distances)

    return chamfer_dist

def compute_hausdorff_distance(pred_x, pred_z, gt_x, gt_z):
    """
    Compute Hausdorff distance using point-to-point correspondence (not nearest neighbor).
    Both arrays should have same length (2672 values).
    Only compute distances where both pred and GT are valid (not -10 or -20).

    Args:
        pred_x, pred_z: Arrays of predicted x, z coordinates (length 2672)
        gt_x, gt_z: Arrays of ground truth x, z coordinates (length 2672)

    Returns:
        hausdorff_dist: Maximum Euclidean distance at corresponding indices
    """
    pred_x = np.array(pred_x, dtype=float)
    pred_z = np.array(pred_z, dtype=float)
    gt_x = np.array(gt_x, dtype=float)
    gt_z = np.array(gt_z, dtype=float)

    # Find valid points where BOTH pred and GT are valid (not -10 or -20)
    pred_valid = (np.abs(pred_x + 10.0) > 0.01) & (np.abs(pred_x + 20.0) > 0.01) & \
                 (np.abs(pred_z + 10.0) > 0.01) & (np.abs(pred_z + 20.0) > 0.01)

    gt_valid = (np.abs(gt_x + 10.0) > 0.01) & (np.abs(gt_x + 20.0) > 0.01) & \
               (np.abs(gt_z + 10.0) > 0.01) & (np.abs(gt_z + 20.0) > 0.01)

    # Only compute distance where both are valid
    both_valid = pred_valid & gt_valid

    if not np.any(both_valid):
        return float('nan')

    # Compute Euclidean distances at corresponding indices (point-to-point)
    distances = np.sqrt((pred_x[both_valid] - gt_x[both_valid])**2 +
                       (pred_z[both_valid] - gt_z[both_valid])**2)

    # Hausdorff: maximum of point-to-point distances
    hausdorff_dist = np.max(distances)

    return hausdorff_dist

def compute_nn_chamfer_distance(pred_x, pred_z, gt_x, gt_z):
    """
    Compute Chamfer distance using nearest neighbor search (standard definition).

    Args:
        pred_x, pred_z: Arrays of predicted x, z coordinates
        gt_x, gt_z: Arrays of ground truth x, z coordinates

    Returns:
        nn_chamfer_dist: Mean of bidirectional nearest neighbor distances
    """
    pred_x = np.array(pred_x, dtype=float)
    pred_z = np.array(pred_z, dtype=float)
    gt_x = np.array(gt_x, dtype=float)
    gt_z = np.array(gt_z, dtype=float)

    # Filter valid points
    pred_valid = (np.abs(pred_x + 10.0) > 0.01) & (np.abs(pred_x + 20.0) > 0.01) & \
                 (np.abs(pred_z + 10.0) > 0.01) & (np.abs(pred_z + 20.0) > 0.01)

    gt_valid = (np.abs(gt_x + 10.0) > 0.01) & (np.abs(gt_x + 20.0) > 0.01) & \
               (np.abs(gt_z + 10.0) > 0.01) & (np.abs(gt_z + 20.0) > 0.01)

    pred_points = np.column_stack([pred_x[pred_valid], pred_z[pred_valid]])
    gt_points = np.column_stack([gt_x[gt_valid], gt_z[gt_valid]])

    if len(pred_points) == 0 or len(gt_points) == 0:
        return float('nan')

    # Compute pairwise distances
    dist_matrix = cdist(pred_points, gt_points, metric='euclidean')

    # Forward: pred -> GT (for each pred point, find nearest GT point)
    min_dist_pred_to_gt = np.min(dist_matrix, axis=1)
    forward_chamfer = np.mean(min_dist_pred_to_gt)

    # Backward: GT -> pred (for each GT point, find nearest pred point)
    min_dist_gt_to_pred = np.min(dist_matrix, axis=0)
    backward_chamfer = np.mean(min_dist_gt_to_pred)

    # Chamfer distance is the average of both directions
    nn_chamfer_dist = (forward_chamfer + backward_chamfer) / 2.0

    return nn_chamfer_dist

def compute_nn_hausdorff_distance(pred_x, pred_z, gt_x, gt_z):
    """
    Compute Hausdorff distance using nearest neighbor search (standard definition).

    Args:
        pred_x, pred_z: Arrays of predicted x, z coordinates
        gt_x, gt_z: Arrays of ground truth x, z coordinates

    Returns:
        nn_hausdorff_dist: Maximum of bidirectional nearest neighbor distances
    """
    pred_x = np.array(pred_x, dtype=float)
    pred_z = np.array(pred_z, dtype=float)
    gt_x = np.array(gt_x, dtype=float)
    gt_z = np.array(gt_z, dtype=float)

    # Filter valid points
    pred_valid = (np.abs(pred_x + 10.0) > 0.01) & (np.abs(pred_x + 20.0) > 0.01) & \
                 (np.abs(pred_z + 10.0) > 0.01) & (np.abs(pred_z + 20.0) > 0.01)

    gt_valid = (np.abs(gt_x + 10.0) > 0.01) & (np.abs(gt_x + 20.0) > 0.01) & \
               (np.abs(gt_z + 10.0) > 0.01) & (np.abs(gt_z + 20.0) > 0.01)

    pred_points = np.column_stack([pred_x[pred_valid], pred_z[pred_valid]])
    gt_points = np.column_stack([gt_x[gt_valid], gt_z[gt_valid]])

    if len(pred_points) == 0 or len(gt_points) == 0:
        return float('nan')

    # Compute pairwise distances
    dist_matrix = cdist(pred_points, gt_points, metric='euclidean')

    # Forward: max over (min distance from each pred point to GT)
    min_dist_pred_to_gt = np.min(dist_matrix, axis=1)
    forward_hausdorff = np.max(min_dist_pred_to_gt)

    # Backward: max over (min distance from each GT point to pred)
    min_dist_gt_to_pred = np.min(dist_matrix, axis=0)
    backward_hausdorff = np.max(min_dist_gt_to_pred)

    # Hausdorff distance is the maximum of both directions
    nn_hausdorff_dist = max(forward_hausdorff, backward_hausdorff)

    return nn_hausdorff_dist

def calculate_curvature(x_points, z_points, smoothing=10):
    """
    Calculate curvature as the rate of change of tangent angles.

    Args:
        x_points: Array of x coordinates
        z_points: Array of z coordinates
        smoothing: Window size for rolling average smoothing

    Returns:
        mean_curvature: Mean absolute curvature value
    """
    # Remove NaN values
    valid_mask = ~(np.isnan(x_points) | np.isnan(z_points))
    x = np.array(x_points)[valid_mask]
    z = np.array(z_points)[valid_mask]

    if len(x) < smoothing:
        return float('nan')

    # Smooth the points first
    df = pd.DataFrame({'x': x, 'z': z})
    df['x_smooth'] = df['x'].rolling(smoothing, center=True).mean()
    df['z_smooth'] = df['z'].rolling(smoothing, center=True).mean()
    df.dropna(inplace=True)

    if len(df) < 2:
        return float('nan')

    x = df['x_smooth'].values
    z = df['z_smooth'].values

    # Calculate tangent angles at each point
    dx = np.gradient(x)
    dz = np.gradient(z)
    tangent_angles = np.arctan2(dz, dx)

    # Rate of change of angle (curvature indicator)
    curvature = np.abs(np.gradient(tangent_angles))

    # Return mean curvature
    return np.mean(curvature)

# # Old one with single x vs z plots
# # def save_test_indices_vs_original_pcl_plots(test_csv_path, original_csv_path, output_dir="./test_indices_vs_original",
# #                                           range_resolution=0.05988024, intensity_threshold=0.1, azimuth=0.0,max_rows=None):
# def save_test_indices_vs_original_pcl_plots(test_csv_path, original_csv_path, output_dir="./test_indices_vs_original",
#                                         range_resolution=0.05988024, intensity_threshold=0.1, azimuth=0.0, specific_row_idx=None):
#     """
#     Plot only the rows that are in test split vs their corresponding original rows
#     """
#     os.makedirs(output_dir, exist_ok=True)

#     # Read both files as pure text lines
#     with open(test_csv_path, 'r') as f:
#         test_lines = [line.rstrip('\n\r') for line in f.readlines()]

#     with open(original_csv_path, 'r') as f:
#         original_lines = [line.rstrip('\n\r') for line in f.readlines()]

#     # Extract test indices from first column of each test line
#     test_indices = []
#     test_data_lines = []

#     for line in test_lines:
#         if line.strip():  # Skip empty lines
#             parts = line.split(',', 1)  # Split only on first comma
#             if len(parts) >= 2:
#                 try:
#                     test_indices.append(int(parts[0]))  # First part is the index
#                     test_data_lines.append(parts[1])   # Rest is the actual data
#                     # test_data_lines.append(line)
#                 except ValueError:
#                     print(f"Warning: Could not parse index from line: {line[:50]}...")
#                     continue

#     # # Process only up to max_rows if specified
#     # rows_to_process = test_indices[:max_rows] if max_rows else test_indices

#     if specific_row_idx is not None:
#         # Find the test data entry that corresponds to this original row index
#         if specific_row_idx in test_indices:
#             test_idx = test_indices.index(specific_row_idx)
#             rows_to_process = [(test_idx, specific_row_idx)]
#         else:
#             print(f"Row index {specific_row_idx} not found in test data")
#             return
#     else:
#         rows_to_process = enumerate(test_indices)

#     # Plot only these specific indices
#     # for i, original_row_idx in enumerate(test_indices):
#     # for i, original_row_idx in enumerate(rows_to_process):
#     for i, original_row_idx in rows_to_process:

#         try:
#             # Get original row (convert to 0-based indexing)
#             if original_row_idx < len(original_lines):
#                 original_row_data = original_lines[original_row_idx]

#             else:
#                 print(f"Warning: Index {original_row_idx} out of range for original data")
#                 continue

#             # Get corresponding test row data
#             test_row_data = test_data_lines[i]

#             # Extract points using pure text processing
#             orig_x, orig_z = extract_pcl_points_from_row(original_row_data, range_resolution, intensity_threshold, azimuth, has_indices=False)
#             test_x, test_z = extract_pcl_points_from_row(test_row_data, range_resolution, intensity_threshold, azimuth, has_indices=True)

#             print("Orig data length",len(orig_x), len(orig_z))
#             print("Test data length",len(test_x), len(test_z))

#             # # Plot comparison
#             if len(orig_x) > 0 or len(test_x) > 0:
#                 plt.figure(figsize=(15, 8))

#                 # Original data (blue)
#                 if len(orig_x) > 0:
#                     plt.scatter(orig_x, orig_z, c='blue', s=3, alpha=0.8, label=f'Original Row {original_row_idx}')

#                 # Test data (red)
#                 if len(test_x) > 0:
#                     plt.scatter(test_x, test_z, c='red', s=2, alpha=0.6, label=f'Test Split Row {original_row_idx}')

#                 plt.xlabel('X (meters)')
#                 plt.ylabel('Z (meters)')
#                 plt.title(f'Original vs Test Split - Row Index {original_row_idx}')
#                 plt.legend()
#                 plt.grid(True, alpha=0.3)
#                 plt.axis('equal')

#                 # Add lines from sensor to show measurement directions (optional)
#                 plt.axhline(y=0, color='gray', linestyle='--', alpha=0.3)
#                 plt.axvline(x=0, color='gray', linestyle='--', alpha=0.3)

#                 plt.savefig(f"{output_dir}/row_index_{int(original_row_idx):04d}.png", dpi=300, bbox_inches='tight')
#                 plt.close()

#                 if i < 5:  # Print first few for verification
#                     print(f"Plotted row index {original_row_idx}: Original={len(orig_x)}, Test={len(test_x)} points")

#         except Exception as e:
#             print(f"Error processing row {i} (original index {original_row_idx}): {e}")
#             continue

#     # print(f"Finished plotting {len(test_indices)} test rows")


# 3 subplots of x vs z and indices vs all x and z
def save_test_indices_vs_original_pcl_plots(test_csv_path, original_csv_path, output_dir="./test_indices_vs_original",
                                        range_resolution=0.05988024, intensity_threshold=0.1, azimuth=-0.000341, specific_row_idx=None):
    """
    Plot only the rows that are in test split vs their corresponding original rows
    """
    os.makedirs(output_dir, exist_ok=True)

    # Lists to accumulate metrics across all rows
    all_mean_distances = []
    all_chamfer_distances = []
    all_hausdorff_distances = []
    all_gt_curvatures = []
    all_test_curvatures = []
    all_curvature_errors = []
    all_correspondence_distances = []  # Store ALL individual point-to-point correspondence distances
    all_original_row_indices = []  # Track the original row index for each frame

    # Read both files as pure text lines
    with open(test_csv_path, 'r') as f:
        test_lines = [line.rstrip('\n\r') for line in f.readlines()]

    with open(original_csv_path, 'r') as f:
        original_lines = [line.rstrip('\n\r') for line in f.readlines()]

    # Extract test indices from first column of each test line
    test_indices = []
    test_data_lines = []

    for line in test_lines:
        if line.strip():  # Skip empty lines
            parts = line.split(',', 1)  # Split only on first comma
            if len(parts) >= 2:
                try:
                    test_indices.append(int(parts[0]))  # First part is the index
                    test_data_lines.append(parts[1])   # Rest is the actual data
                except ValueError:
                    print(f"Warning: Could not parse index from line: {line[:50]}...")
                    continue

    if specific_row_idx is not None:
        # Find the test data entry that corresponds to this original row index
        if specific_row_idx in test_indices:
            test_idx = test_indices.index(specific_row_idx)
            rows_to_process = [(test_idx, specific_row_idx)]
        else:
            print(f"Row index {specific_row_idx} not found in test data")
            return
    else:
        rows_to_process = enumerate(test_indices)

    # Plot only these specific indices
    for i, original_row_idx in tqdm(rows_to_process, desc="Processing frames"):
        try:
            # Get original row (convert to 0-based indexing)
            if original_row_idx < len(original_lines):
                original_row_data = original_lines[original_row_idx]
            else:
                print(f"Warning: Index {original_row_idx} out of range for original data")
                continue

            # Get corresponding test row data
            test_row_data = test_data_lines[i]

            # Extract points using pure text processing
            orig_x, orig_z = extract_pcl_points_from_row(original_row_data, range_resolution, intensity_threshold, azimuth, has_indices=False)
            test_x, test_z = extract_pcl_points_from_row(test_row_data, range_resolution, intensity_threshold, azimuth, has_indices=True)

            # Compute simple point-to-point mean distance for valid points
            test_x_arr = np.array(test_x, dtype=float)
            test_z_arr = np.array(test_z, dtype=float)
            orig_x_arr = np.array(orig_x, dtype=float)
            orig_z_arr = np.array(orig_z, dtype=float)

            # Find points valid in both
            both_valid = ~(np.isnan(test_x_arr) | np.isnan(test_z_arr) | np.isnan(orig_x_arr) | np.isnan(orig_z_arr))

            if np.any(both_valid):
                distances = np.sqrt((test_x_arr[both_valid] - orig_x_arr[both_valid])**2 +
                                   (test_z_arr[both_valid] - orig_z_arr[both_valid])**2)
                mean_distance = np.mean(distances)
                # Store all individual distances for histogram
                all_correspondence_distances.extend(distances.tolist())
            else:
                mean_distance = float('nan')

            # Calculate curvatures for both GT and test
            gt_curvature = calculate_curvature(orig_x, orig_z, smoothing=15)
            test_curvature = calculate_curvature(test_x, test_z, smoothing=15)

            # Calculate Chamfer and Hausdorff distances
            chamfer_dist = compute_nn_chamfer_distance(test_x, test_z, orig_x, orig_z)
            hausdorff_dist = compute_nn_hausdorff_distance(test_x, test_z, orig_x, orig_z)

            # Accumulate metrics (with original row index tracking)
            if not np.isnan(mean_distance):
                all_mean_distances.append(mean_distance)
                all_original_row_indices.append(original_row_idx)  # Track original index
            if not np.isnan(chamfer_dist):
                all_chamfer_distances.append(chamfer_dist)
            if not np.isnan(hausdorff_dist):
                all_hausdorff_distances.append(hausdorff_dist)
            if not np.isnan(gt_curvature):
                all_gt_curvatures.append(gt_curvature)
            if not np.isnan(test_curvature):
                all_test_curvatures.append(test_curvature)
            if not np.isnan(gt_curvature) and not np.isnan(test_curvature):
                all_curvature_errors.append(abs(gt_curvature - test_curvature))

            # Create figure
            if len(orig_x) > 0 or len(test_x) > 0:
                fig, (ax1) = plt.subplots(1, 1, figsize=(12, 12))

            # Filter NaN and plot - create clean point lists with index tracking
            if len(orig_x) > 0:
                orig_x_clean = []
                orig_z_clean = []
                orig_indices = []
                for idx, (x, z) in enumerate(zip(orig_x, orig_z)):
                    if not (np.isnan(x) or np.isnan(z)):
                        orig_x_clean.append(x)
                        orig_z_clean.append(z)
                        orig_indices.append(idx)
                if orig_x_clean:
                    ax1.scatter(orig_x_clean, orig_z_clean, c='blue', s=10, alpha=0.8, label=f'Original Row {original_row_idx}')

            if len(test_x) > 0:
                test_x_clean = []
                test_z_clean = []
                test_indices = []
                for idx, (x, z) in enumerate(zip(test_x, test_z)):
                    if not (np.isnan(x) or np.isnan(z)):
                        test_x_clean.append(x)
                        test_z_clean.append(z)
                        test_indices.append(idx)
                if test_x_clean:
                    ax1.scatter(test_x_clean, test_z_clean, c='red', s=10, alpha=0.6, label=f'Test Split Row {original_row_idx}')

            # Add correspondence arrows between matching points (based on original index)
            if len(test_x_clean) > 0 and len(orig_x_clean) > 0:
                # Create mapping from original index to clean list position
                orig_idx_map = {orig_idx: i for i, orig_idx in enumerate(orig_indices)}
                test_idx_map = {test_idx: i for i, test_idx in enumerate(test_indices)}

                # Find common indices that exist in both clean lists
                common_indices = set(orig_indices) & set(test_indices)
                common_indices = sorted(common_indices)

                # Show all correspondence arrows
                for orig_idx in common_indices:
                    orig_pos = orig_idx_map[orig_idx]
                    test_pos = test_idx_map[orig_idx]
                    ax1.arrow(orig_x_clean[orig_pos], orig_z_clean[orig_pos],
                             test_x_clean[test_pos] - orig_x_clean[orig_pos],
                             test_z_clean[test_pos] - orig_z_clean[orig_pos],
                             head_width=0.02, head_length=0.02,
                             fc='gray', ec='gray', alpha=0.5, linestyle='--', linewidth=0.5)

            # Update title with distance metric and curvatures
            title_text = f'X vs Z - Row Index {original_row_idx}\n'
            if not np.isnan(mean_distance):
                title_text += f'Mean Point-to-Point Distance: {mean_distance:.4f}m\n'
            else:
                title_text += 'Mean Point-to-Point Distance: N/A (no common valid points)\n'

            # Add curvature values
            if not np.isnan(gt_curvature):
                title_text += f'GT Curvature: {gt_curvature:.6f}  '
            else:
                title_text += 'GT Curvature: N/A  '

            if not np.isnan(test_curvature):
                title_text += f'Test Curvature: {test_curvature:.6f}'
            else:
                title_text += 'Test Curvature: N/A'

            ax1.set_xlabel('X (meters)', fontsize=26)
            ax1.set_ylabel('Z (meters)', fontsize=26)
            ax1.set_title(title_text, fontsize=30)
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            ax1.axis('equal')
            ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.3)
            ax1.axvline(x=0, color='gray', linestyle='--', alpha=0.3)

            # # Subplot 2: X points vs point index (0 to 668*4)
            # point_indices = np.arange(len(orig_x)) if len(orig_x) > 0 else np.arange(len(test_x))

            # if len(orig_x) > 0:
            #     ax2.plot(point_indices[:len(orig_x)], orig_x, 'b-', linewidth=1, alpha=0.8, label='Original X')
            # if len(test_x) > 0:
            #     ax2.plot(point_indices[:len(test_x)], test_x, 'r-', linewidth=1, alpha=0.8, label='Test X')

            # ax2.set_xlabel('Point Index (0 to 668×4)')
            # ax2.set_ylabel('X (meters)')
            # ax2.set_title(f'X Coordinates - Row Index {original_row_idx}')
            # ax2.legend()
            # ax2.grid(True, alpha=0.3)

            # Subplot 3: Z points vs point index (0 to 668*4)
            # uncomment to remove filtering -10 and -20
            # if len(orig_z) > 0:
            #     ax3.plot(point_indices[:len(orig_z)], orig_z, 'b-', linewidth=1, alpha=0.8, label='Original Phi')
            # if len(test_z) > 0:
            #     ax3.plot(point_indices[:len(test_z)], test_z, 'r-', linewidth=1, alpha=0.8, label='Test Phi')

            # if len(orig_z) > 0:
            #     # Filter out -10 and -20 values
            #     orig_mask = (np.array(orig_z) != -10) & (np.array(orig_z) != -20)
            #     if np.any(orig_mask):
            #         filtered_orig_z = np.array(orig_z)[orig_mask]
            #         filtered_orig_indices = np.arange(len(orig_z))[orig_mask]
            #         ax3.scatter(filtered_orig_indices, filtered_orig_z, c='blue', s=2, alpha=0.8, label='Original Phi')

            # if len(orig_z) > 0:
            #     orig_z = np.array(orig_z).reshape(668, 4)
            #     for point_idx in range(668):
            #         for beam_idx in range(4):
            #             phi_val = orig_z[point_idx, beam_idx]

            #             # Filter out -10 and -20 values
            #             # if phi_val not in [-10, -20]:
            #             ax3.scatter(point_idx, phi_val, c='blue', s=30, alpha=0.8,marker='x',
            #                             label='Original Phi' if (point_idx == 0 and beam_idx == 0) else "")

            # # if len(test_z) > 0:
            # #     # Filter out -10 and -20 values
            # #     test_mask = (np.array(test_z) != -10) & (np.array(test_z) != -20)
            # #     if np.any(test_mask):
            # #         filtered_test_z = np.array(test_z)[test_mask]
            # #         filtered_test_indices = np.arange(len(test_z))[test_mask]
            # #         ax3.scatter(filtered_test_indices, filtered_test_z, c='red', s=2, alpha=0.8, label='Test Phi')

            # if len(test_z) > 0:
            #     test_z = np.array(test_z).reshape(668, 4)  # reshape flat array to (668,4)
            #     for point_idx in range(668):
            #         for beam_idx in range(4):
            #             phi_val = test_z[point_idx, beam_idx]

            #             # Filter out -10 and -20 values
            #             # if phi_val not in [-10, -20]:
            #             ax3.scatter(point_idx, phi_val, s=5, alpha=0.8,marker='o', facecolors='none', edgecolors='red',
            #                             label='Test Phi' if (point_idx == 0 and beam_idx == 0) else "")

            # ax3.set_xlabel('Point Index')
            # ax3.set_ylabel('Z (meters)')
            # # ax3.set_ylabel('Phi (Rads)')
            # ax3.set_title(f'Z Coordinates - Row Index {original_row_idx}')
            # ax3.legend(loc = 'upper right')
            # ax3.grid(True, alpha=0.3)

            # plt.tight_layout()
            plt.savefig(f"{output_dir}/row_index_{int(original_row_idx):04d}.png", dpi=300, bbox_inches='tight')
            plt.show()
            plt.close()

            if i < 5:  # Print first few for verification
                print(f"Plotted row index {original_row_idx}: Original={len(orig_x)}, Test={len(test_x)} points")
                if not np.isnan(mean_distance):
                    print(f"  Mean Point-to-Point Distance: {mean_distance:.4f}m")
                else:
                    print(f"  Mean Point-to-Point Distance: N/A (no common valid points)")

        except Exception as e:
            print(f"Error processing row {i} (original index {original_row_idx}): {e}")
            continue

    # Create summary plots (always, even for single row)
    if len(all_correspondence_distances) > 0:
        create_distribution_plots(all_mean_distances, all_chamfer_distances, all_hausdorff_distances,
                                 all_gt_curvatures, all_test_curvatures, all_curvature_errors, output_dir)

        # Create curvature classification plot with original row indices
        # use_gt_curvature=True uses GT curvature for classification
        # use_gt_curvature=False uses predicted curvature for classification
        create_curvature_classification_plot(all_mean_distances, all_gt_curvatures, all_test_curvatures,
                                            all_original_row_indices, output_dir, use_gt_curvature=False)

        # Create error histograms by curvature class (shows performance per complexity)
        create_error_histograms_by_curvature(all_mean_distances, all_gt_curvatures, output_dir)

def create_distribution_plots(correspondence_distances, chamfer_distances, hausdorff_distances,
                              gt_curvatures, test_curvatures, curvature_errors, output_dir):
    """Create histogram and box plot for per-frame mean reconstruction errors."""

    if len(correspondence_distances) == 0:
        print("No data to plot")
        return

    # Create figure with 2 subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Histogram - now showing ALL individual correspondences
    ax1.hist(correspondence_distances, bins=50, alpha=0.7, color='steelblue', edgecolor='black')
    ax1.set_xlabel('Correspondence Distance (m)', fontsize=plt.rcParams['axes.labelsize'])
    ax1.set_ylabel('Frequency (Number of Point Correspondences)', fontsize=plt.rcParams['axes.labelsize'])
    ax1.set_title('All Point-to-Point Correspondence Distances (Histogram)',
                  fontsize=plt.rcParams['axes.titlesize'], fontweight='bold')
    ax1.grid(True, alpha=0.3)

    # Add statistics text
    mean_val = np.mean(correspondence_distances)
    std_val = np.std(correspondence_distances)
    median_val = np.median(correspondence_distances)
    ax1.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.4f}m')
    ax1.axvline(median_val, color='green', linestyle='--', linewidth=2, label=f'Median: {median_val:.4f}m')
    ax1.legend()

    # Box plot
    bp = ax2.boxplot([correspondence_distances], tick_labels=['Correspondence\nDistances'], patch_artist=True, showmeans=True)
    bp['boxes'][0].set_facecolor('lightblue')

    ax2.set_ylabel('Distance (m)', fontsize=plt.rcParams['axes.labelsize'])
    ax2.set_title('All Point-to-Point Correspondence Distances (Box Plot)',
                  fontsize=plt.rcParams['axes.titlesize'], fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')

    # Add statistics text box with sigma ranges (Gaussian distribution)
    stats_text = (
        f'N: {len(correspondence_distances)}\n'
        f'Mean: {mean_val:.4f}m\n'
        f'Std: {std_val:.4f}m\n'
        f'Median: {median_val:.4f}m\n'
        f'Min: {np.min(correspondence_distances):.4f}m\n'
        f'Max: {np.max(correspondence_distances):.4f}m\n'
        f'1σ: [{mean_val - std_val:.4f}, {mean_val + std_val:.4f}]m\n'
        f'2σ: [{mean_val - 2*std_val:.4f}, {mean_val + 2*std_val:.4f}]m\n'
        f'3σ: [{mean_val - 3*std_val:.4f}, {mean_val + 3*std_val:.4f}]m'
    )
    ax2.text(1.15, 0.5, stats_text, transform=ax2.transAxes, fontsize=14,  # Keep stats box readable
             verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
             family='monospace')

    plt.tight_layout()
    plt.savefig(f"{output_dir}/correspondence_distances_histogram.png", dpi=300, bbox_inches='tight')
    # plt.show(block=True)
    plt.close()

    print(f"\nCorrespondence distance histogram saved to {output_dir}/correspondence_distances_histogram.png")
    print(f"Total Correspondences: {len(correspondence_distances)}")
    print(f"Mean Distance: {mean_val:.4f}m ± {std_val:.4f}m")

def create_curvature_classification_plot(mean_distances, gt_curvatures, test_curvatures, original_row_indices, output_dir, use_gt_curvature=True):
    """
    Classify frames by curvature and plot reconstruction error per curvature class.
    Shows how model performance varies with surface curvature.

    Args:
        mean_distances: List of mean reconstruction errors per frame
        gt_curvatures: List of GT curvatures per frame
        test_curvatures: List of predicted curvatures per frame
        original_row_indices: List of original row indices from the full dataset
        output_dir: Output directory for plots
        use_gt_curvature: If True, classify by GT curvature; if False, classify by predicted curvature
    """
    # Choose which curvature to use for classification
    curvatures_for_classification = gt_curvatures if use_gt_curvature else test_curvatures
    curvature_type = "GT" if use_gt_curvature else "Predicted"

    if len(mean_distances) == 0 or len(curvatures_for_classification) == 0:
        print("No curvature data to plot")
        return

    # Pair up the data (frame-level metrics) WITH ORIGINAL ROW INDEX
    paired_data = [(orig_idx, curv, dist) for orig_idx, curv, dist in zip(original_row_indices, curvatures_for_classification, mean_distances)
                   if not np.isnan(curv) and not np.isnan(dist)]

    if len(paired_data) == 0:
        print("No valid paired curvature-distance data")
        return

    frame_indices, curvatures, distances = zip(*paired_data)
    frame_indices = np.array(frame_indices)  # Now these are ORIGINAL row indices
    curvatures = np.array(curvatures)
    distances = np.array(distances)

    # Classify curvatures into 3 classes based on tertiles (33rd and 66th percentiles)
    low_thresh = np.percentile(curvatures, 33)
    high_thresh = np.percentile(curvatures, 67)

    # Create class distance dictionary and track frame indices
    class_distances = {'Low': [], 'Medium': [], 'High': []}
    class_frames = {'Low': [], 'Medium': [], 'High': []}

    for frame_idx, curv, dist in zip(frame_indices, curvatures, distances):
        if curv <= low_thresh:
            label = 'Low'
        elif curv <= high_thresh:
            label = 'Medium'
        else:
            label = 'High'
        class_distances[label].append(dist)
        class_frames[label].append((frame_idx, curv, dist))

    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Subplot 1: Box plot of reconstruction error by curvature class
    class_names = ['Low', 'Medium', 'High']
    data_to_plot = [class_distances[name] for name in class_names]

    bp = ax1.boxplot(data_to_plot, labels=class_names, patch_artist=True, showmeans=True)
    colors = ['lightgreen', 'lightyellow', 'lightcoral']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)

    ax1.set_xlabel('Curvature Class', fontsize=plt.rcParams['axes.labelsize'])
    ax1.set_ylabel('Mean Reconstruction Error (m)', fontsize=plt.rcParams['axes.labelsize'])
    ax1.set_title(f'Reconstruction Error vs {curvature_type} Curvature Class',
                  fontsize=plt.rcParams['axes.titlesize'], fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')

    # Add statistics text
    stats_text = ""
    for name in class_names:
        data = class_distances[name]
        if len(data) > 0:
            mean_err = np.mean(data)
            std_err = np.std(data)
            stats_text += f"{name}: {mean_err:.4f}m ± {std_err:.4f}m (n={len(data)})\n"

    ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes, fontsize=14,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
             family='monospace')

    # Subplot 2: Scatter plot of error vs curvature
    ax2.scatter(curvatures, distances, alpha=0.5, s=20, c=distances, cmap='coolwarm')
    ax2.set_xlabel(f'{curvature_type} Curvature', fontsize=plt.rcParams['axes.labelsize'])
    ax2.set_ylabel('Mean Reconstruction Error (m)', fontsize=plt.rcParams['axes.labelsize'])
    ax2.set_title(f'Reconstruction Error vs {curvature_type} Curvature (Scatter)',
                  fontsize=plt.rcParams['axes.titlesize'], fontweight='bold')
    ax2.grid(True, alpha=0.3)

    # Add vertical lines for class boundaries
    ax2.axvline(low_thresh, color='green', linestyle='--', linewidth=2, alpha=0.7, label=f'Low/Med: {low_thresh:.6f}')
    ax2.axvline(high_thresh, color='red', linestyle='--', linewidth=2, alpha=0.7, label=f'Med/High: {high_thresh:.6f}')
    ax2.legend()

    plt.tight_layout()
    filename_suffix = "gt" if use_gt_curvature else "pred"
    plt.savefig(f"{output_dir}/curvature_classification_error_{filename_suffix}.png", dpi=300, bbox_inches='tight')
    # plt.show(block=True)
    plt.close()

    print(f"\nCurvature classification plot saved to {output_dir}/curvature_classification_error_{filename_suffix}.png")
    print(f"Classification based on: {curvature_type} Curvature")
    print(f"Low curvature threshold: {low_thresh:.6f}")
    print(f"High curvature threshold: {high_thresh:.6f}")

    # Save frame classification to CSV for inspection
    with open(f"{output_dir}/curvature_frame_classification_{filename_suffix}.csv", 'w') as f:
        f.write(f"Original_Row_Index,{curvature_type}_Curvature,Mean_Error,Class\n")
        for class_name in ['Low', 'Medium', 'High']:
            for frame_idx, curv, dist in class_frames[class_name]:
                f.write(f"{frame_idx},{curv:.8f},{dist:.6f},{class_name}\n")

    print(f"\nFrame classification saved to {output_dir}/curvature_frame_classification_{filename_suffix}.csv")
    print("Note: Frame indices are ORIGINAL row indices from the full dataset")

    # Print example frames from each class for visual inspection
    print("\n=== Example Frames by Curvature Class ===")
    for class_name in ['Low', 'Medium', 'High']:
        frames = class_frames[class_name]
        if len(frames) > 0:
            # Sort by curvature to get most representative examples
            frames_sorted = sorted(frames, key=lambda x: x[1])  # Sort by curvature

            # Pick 5 representative frames (min, 25%, median, 75%, max curvature)
            n = len(frames_sorted)
            indices_to_show = [0, n//4, n//2, 3*n//4, n-1] if n >= 5 else range(n)

            print(f"\n{class_name} Curvature Class ({len(frames)} frames):")
            for i in indices_to_show:
                frame_idx, curv, dist = frames_sorted[i]
                print(f"  Original Row {frame_idx}: curvature={curv:.8f}, error={dist:.4f}m")

            # Print the highest error frame in this class
            worst_frame = max(frames, key=lambda x: x[2])
            print(f"  --> Worst error in {class_name} class: Original Row {worst_frame[0]} (curv={worst_frame[1]:.8f}, error={worst_frame[2]:.4f}m)")


def create_error_histograms_by_curvature(mean_distances, gt_curvatures, output_dir):
    """
    Create histograms of reconstruction errors for each curvature class (Low, Medium, High).
    Shows how model performance (error distribution) varies with surface complexity.

    Args:
        mean_distances: List of mean reconstruction errors per frame
        gt_curvatures: List of GT curvatures per frame
        output_dir: Output directory for plots
    """
    mean_distances = np.array(mean_distances)
    curvatures = np.array(gt_curvatures)

    # Remove NaN values (paired)
    valid_mask = ~np.isnan(curvatures) & ~np.isnan(mean_distances)
    curvatures = curvatures[valid_mask]
    mean_distances = mean_distances[valid_mask]

    if len(curvatures) == 0:
        print("No valid data for error histograms by curvature")
        return

    # Calculate thresholds (tertiles)
    low_thresh = np.percentile(curvatures, 33)
    high_thresh = np.percentile(curvatures, 67)

    # Classify errors by curvature class
    low_errors = mean_distances[curvatures <= low_thresh]
    medium_errors = mean_distances[(curvatures > low_thresh) & (curvatures <= high_thresh)]
    high_errors = mean_distances[curvatures > high_thresh]

    # Create figure with 3 subplots (one histogram per class)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    class_data = [
        ('Low Curvature', low_errors, 'green'),
        ('Medium Curvature', medium_errors, 'orange'),
        ('High Curvature', high_errors, 'red')
    ]

    # Find global min/max for consistent x-axis across all histograms
    global_min = mean_distances.min()
    global_max = mean_distances.max()
    bins = np.linspace(global_min, global_max, 50)

    for ax, (class_name, class_errors, color) in zip(axes, class_data):
        if len(class_errors) > 0:
            ax.hist(class_errors, bins=bins, color=color, alpha=0.7, edgecolor='black', linewidth=0.5)

            # Add statistics
            mean_val = np.mean(class_errors)
            std_val = np.std(class_errors)
            median_val = np.median(class_errors)

            # Add mean line
            ax.axvline(mean_val, color='black', linestyle='--', linewidth=2)

            stats_text = f'N = {len(class_errors)}\nMean = {mean_val:.4f}m\nStd = {std_val:.4f}m\nMedian = {median_val:.4f}m'
            ax.text(0.95, 0.95, stats_text, transform=ax.transAxes, fontsize=14,
                   verticalalignment='top', horizontalalignment='right',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                   family='monospace')

        ax.set_xlabel('Reconstruction Error (m)', fontsize=plt.rcParams['axes.labelsize'])
        ax.set_ylabel('Frequency (Number of Frames)', fontsize=plt.rcParams['axes.labelsize'])
        ax.set_title(f'{class_name}', fontsize=plt.rcParams['axes.titlesize'],
                    fontweight='bold', color=color)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(global_min, global_max)

    plt.suptitle('Reconstruction Error Distribution by Curvature Class\n'
                 f'(Low: curv ≤ {low_thresh:.6f} | Medium: curv ≤ {high_thresh:.6f} | High: curv > {high_thresh:.6f})',
                 fontsize=plt.rcParams['figure.titlesize'], fontweight='bold')
    plt.tight_layout()

    output_path = f"{output_dir}/error_histograms_by_curvature.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    # plt.show(block=True)
    plt.close()

    print(f"\nError histograms by curvature saved to {output_path}")
    print(f"Frames per class: Low={len(low_errors)}, Medium={len(medium_errors)}, High={len(high_errors)}")


def main():

    # predictions_with_indices_path = "./data_splits/test_predictions_with_indices.csv"
    # predictions_with_indices_path = "./data_splits/train_predictions_with_indices.csv"
    # predictions_with_indices_path = "./data_splits/test_predictions_final.csv"
    # predictions_with_indices_path = "./data_splits/test_predictions_two_stage.csv"
    predictions_with_indices_path = "./data_splits/test_predictions_full_three_stage.csv"

    # Generate comparison plots for ALL test data (creates histogram over all correspondences)
    save_test_indices_vs_original_pcl_plots(
        test_csv_path=predictions_with_indices_path,
        original_csv_path="/Users/farhang/Downloads/fls_all_with_phis_long.csv",
        output_dir="./test_indices_vs_original",
        specific_row_idx=8194  # Process all rows

    )

    # # To process a specific row only:
    # save_test_indices_vs_original_pcl_plots(
    #     test_csv_path=predictions_with_indices_path,
    #     original_csv_path="/Users/farhang/Downloads/fls_all_with_phis_long.csv",
    #     output_dir="./test_indices_vs_original",
    #     specific_row_idx=15001
    # )

if __name__ == "__main__":
    main()