import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np

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
    # print(len(numeric_values))
    # print("First 10 cols:", numeric_values[:10])
    # print("Around 3335–3345:", numeric_values[3335:3345])
    # print("Around 6005–6015:", numeric_values[6005:6015])

    # Extract intensities and phis based on column positions
    if has_indices:
        # Skip first column (index) in test data
        intensities = numeric_values[2:670]  # columns 2-669
        # print(intensities[:5])
        phis = numeric_values[3342:6014]     # columns 3342-6013\
        # phis = numeric_values[3342:6014].reshape(668, 4)
        print(phis[437])
        # print("Has Indices")
    else:
        intensities = numeric_values[1:669]  # columns 1-668
        # print(intensities[:5])
        phis = numeric_values[3341:6013]     # columns 3341-6012
        # phis = numeric_values[3341:6013].reshape(668, 4)
        print(phis[437])
        # print("No Indices")

    x_points = []
    z_points = []

    for point_idx in range(668):
        reverse_idx = 667 - point_idx

        # Range calculation using the full range span
        max_range = 40.0
        min_range = 0.5
        range_val = min_range + (reverse_idx / 668) * (max_range - min_range)

        phi_start_idx = point_idx * 4

        for beam_idx in range(4):
            phi_idx = phi_start_idx + beam_idx

            # Get phi value, defaulting to -20 if out of bounds or NaN
            # if not pd.isna(phi_rad) and phi_rad not in [-10.0, -20.0]:
            if phi_idx < len(phis):
                phi_rad = phis[phi_idx]
                if pd.isna(phi_rad): #or phi_rad in [-10.0, -20.0] :
                    phi_rad = -20.0
                # elif phi_rad =
            # else:
            #     phi_rad = -20.0

            # # Calculate 3D coordinates for ALL points (no filtering)
            # x = range_val * np.cos(azimuth) * np.cos(phi_rad)
            # y = range_val * np.sin(azimuth) * np.cos(phi_rad)
            # z = range_val * np.sin(phi_rad)

            # # # Temp
            x = phi_rad
            z= phi_rad

            x_points.append(x)
            z_points.append(z)

    # print(f"Total points generated: {len(x_points)} (expected: {668*4})")
    return x_points, z_points

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
                                        range_resolution=0.05988024, intensity_threshold=0.1, azimuth=0.0, specific_row_idx=None):
    """
    Plot only the rows that are in test split vs their corresponding original rows
    """
    os.makedirs(output_dir, exist_ok=True)

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
    for i, original_row_idx in rows_to_process:
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

            print("Orig data length", len(orig_x), len(orig_z))
            print("Test data length", len(test_x), len(test_z))

            # Create figure with 3 subplots (1 row, 3 columns)
            if len(orig_x) > 0 or len(test_x) > 0:
                # fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))
                fig, (ax3) = plt.subplots(1, 1, figsize=(20, 6))

                # Subplot 1: Scatter plot (X vs Z)
                # if len(orig_x) > 0:
                #     ax1.scatter(orig_x, orig_z, c='blue', s=3, alpha=0.8, label=f'Original Row {original_row_idx}')
                # if len(test_x) > 0:
                #     ax1.scatter(test_x, test_z, c='red', s=2, alpha=0.6, label=f'Test Split Row {original_row_idx}')

            # if len(orig_x) > 0:
            #     # Filter out -10 and -20 values from both x and z
            #     orig_mask = (np.array(orig_x) != -10) & (np.array(orig_x) != -20) & \
            #                 (np.array(orig_z) != -10) & (np.array(orig_z) != -20)
            #     if np.any(orig_mask):
            #         filtered_orig_x = np.array(orig_x)[orig_mask]
            #         filtered_orig_z = np.array(orig_z)[orig_mask]
            #         ax1.scatter(filtered_orig_x, filtered_orig_z, c='blue', s=3, alpha=0.8, label=f'Original Row {original_row_idx}')

            # if len(test_x) > 0:
            #     # Filter out -10 and -20 values from both x and z
            #     test_mask = (np.array(test_x) != -10) & (np.array(test_x) != -20) & \
            #                 (np.array(test_z) != -10) & (np.array(test_z) != -20)
            #     if np.any(test_mask):
            #         filtered_test_x = np.array(test_x)[test_mask]
            #         filtered_test_z = np.array(test_z)[test_mask]
            #         ax1.scatter(filtered_test_x, filtered_test_z, c='red', s=2, alpha=0.6, label=f'Test Split Row {original_row_idx}')

            #     ax1.set_xlabel('X (meters)')
            #     ax1.set_ylabel('Z (meters)')
            #     ax1.set_title(f'X vs Z - Row Index {original_row_idx}')
            #     ax1.legend()
            #     ax1.grid(True, alpha=0.3)
            #     ax1.axis('equal')
            #     ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.3)
            #     ax1.axvline(x=0, color='gray', linestyle='--', alpha=0.3)

            #     # Subplot 2: X points vs point index (0 to 668*4)
            #     point_indices = np.arange(len(orig_x)) if len(orig_x) > 0 else np.arange(len(test_x))

            #     if len(orig_x) > 0:
            #         ax2.plot(point_indices[:len(orig_x)], orig_x, 'b-', linewidth=1, alpha=0.8, label='Original X')
            #     if len(test_x) > 0:
            #         ax2.plot(point_indices[:len(test_x)], test_x, 'r-', linewidth=1, alpha=0.8, label='Test X')

            #     ax2.set_xlabel('Point Index (0 to 668×4)')
            #     ax2.set_ylabel('X (meters)')
            #     ax2.set_title(f'X Coordinates - Row Index {original_row_idx}')
            #     ax2.legend()
            #     ax2.grid(True, alpha=0.3)

                # Subplot 3: Z points vs point index (0 to 668*4)
                #uncimment to remove filtering -10 and -20
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

                if len(orig_z) > 0:
                    orig_z = np.array(orig_z).reshape(668, 4)
                    for point_idx in range(668):
                        for beam_idx in range(4):
                            phi_val = orig_z[point_idx, beam_idx]

                            # Filter out -10 and -20 values
                            # if phi_val not in [-10, -20]:
                            ax3.scatter(point_idx, phi_val, c='blue', s=30, alpha=0.8,marker='x',
                                            label='Original Phi' if (point_idx == 0 and beam_idx == 0) else "")

                # if len(test_z) > 0:
                #     # Filter out -10 and -20 values
                #     test_mask = (np.array(test_z) != -10) & (np.array(test_z) != -20)
                #     if np.any(test_mask):
                #         filtered_test_z = np.array(test_z)[test_mask]
                #         filtered_test_indices = np.arange(len(test_z))[test_mask]
                #         ax3.scatter(filtered_test_indices, filtered_test_z, c='red', s=2, alpha=0.8, label='Test Phi')

                if len(test_z) > 0:
                    test_z = np.array(test_z).reshape(668, 4)  # reshape flat array to (668,4)
                    for point_idx in range(668):
                        for beam_idx in range(4):
                            phi_val = test_z[point_idx, beam_idx]

                            # Filter out -10 and -20 values
                            # if phi_val not in [-10, -20]:
                            ax3.scatter(point_idx, phi_val, s=5, alpha=0.8,marker='o', facecolors='none', edgecolors='red',
                                            label='Test Phi' if (point_idx == 0 and beam_idx == 0) else "")

                ax3.set_xlabel('Point Index')
                # ax3.set_ylabel('Z (meters)')
                ax3.set_ylabel('Phi (Rads)')
                ax3.set_title(f'Z Coordinates - Row Index {original_row_idx}')
                ax3.legend(loc = 'upper right')
                ax3.grid(True, alpha=0.3)

                plt.tight_layout()
                plt.savefig(f"{output_dir}/row_index_{int(original_row_idx):04d}.png", dpi=300, bbox_inches='tight')
                plt.show()
                plt.close()

                if i < 5:  # Print first few for verification
                    print(f"Plotted row index {original_row_idx}: Original={len(orig_x)}, Test={len(test_x)} points")

        except Exception as e:
            print(f"Error processing row {i} (original index {original_row_idx}): {e}")
            continue

def main():

    # predictions_with_indices_path = "./data_splits/test_predictions_with_indices.csv"
    # predictions_with_indices_path = "./data_splits/train_predictions_with_indices.csv"
    # predictions_with_indices_path = "./data_splits/test_predictions_final.csv"
    predictions_with_indices_path = "./data_splits/test_predictions_two_stage.csv"
    # save_predictions_with_indices_to_csv(
    #     results['tangent'],
    #     results['phi'],
    #     test_with_indices,
    #     predictions_with_indices_path
    # )
    # Generate comparison plots
    save_test_indices_vs_original_pcl_plots(
        test_csv_path=predictions_with_indices_path,
        original_csv_path="/Users/farhang/Downloads/fls_all_with_phi.csv",
        output_dir="./test_indices_vs_original",
        specific_row_idx=31
    )

    # # process a specific row
    # save_test_indices_vs_original_pcl_plots(
    #     test_csv_path=predictions_with_indices_path,
    #     original_csv_path="/Users/farhang/Downloads/fls_all_with_phis_long.csv",
    #     output_dir="./test_indices_vs_original",
    #     specific_row_idx=31
    # )

if __name__ == "__main__":
    main()