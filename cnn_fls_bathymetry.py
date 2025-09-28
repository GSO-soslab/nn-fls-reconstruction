import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split, Subset
import pandas as pd
import numpy as np
import math
import os
from contextlib import nullcontext
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import csv
# ==================== UTILITIES ====================

def get_device():
    """Get the best available device."""
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')

def create_valid_mask(data, invalid_values=[-20.0]):
    """Convert NaNs to -20.0 and create mask for valid values."""
    data_modified = data.clone()
    data_modified[torch.isnan(data_modified)] = -20.0

    valid_mask = torch.ones_like(data_modified, dtype=torch.bool)
    for val in invalid_values:
        valid_mask &= (data_modified != val)

    return data_modified, valid_mask

def ensure_dir(path):
    """Ensure directory exists."""
    os.makedirs(path, exist_ok=True)

def reshape_predictions(predictions, prediction_type):
    """
    Reshape predictions based on type.

    Args:
        predictions: Model output tensor
        prediction_type: 'phi', 'tangent', or 'combined'
    """
    if predictions.dim() == 1:
        predictions = predictions.unsqueeze(0)
        squeeze_output = True
    else:
        squeeze_output = False

    batch_size = predictions.size(0)

    if prediction_type == 'phi':
        # Phi-only mode: [batch_size, 2672] -> phis [batch_size, 4, 668]
        phis = torch.zeros(batch_size, 4, 668)
        for i in range(4):
            phis[:, i, :] = predictions[:, i::4]
        tangents = torch.zeros_like(phis)  # No tangent predictions

    elif prediction_type == 'tangent':
        # Tangent-only mode: [batch_size, 2672] -> tangents [batch_size, 4, 668]
        tangents = torch.zeros(batch_size, 4, 668)
        for i in range(4):
            tangents[:, i, :] = predictions[:, i::4]
        phis = torch.zeros_like(tangents)  # No phi predictions

    elif prediction_type == 'combined':
        # Combined mode: [batch_size, 5336] -> both tangents and phis
        reshaped = predictions.view(batch_size, 668, 8)
        tangents = reshaped[:, :, :4].transpose(1, 2)  # [batch, 4, 668]
        phis = reshaped[:, :, 4:].transpose(1, 2)      # [batch, 4, 668]

    else:
        raise ValueError(f"Unknown prediction_type: {prediction_type}")

    if squeeze_output:
        tangents = tangents.squeeze(0)
        phis = phis.squeeze(0)

    return tangents, phis

def evaluate_single_row(prediction, ground_truth):
    """Evaluate single row with comprehensive metrics for angles and flags."""
    valid_mask = (ground_truth != -10.0) & (ground_truth != -20.0)
    flag_10_mask = (ground_truth == -10.0)
    flag_20_mask = (ground_truth == -20.0)

    results = {
        'total_points': len(ground_truth),
        'valid_points': valid_mask.sum().item(),
        'flag_10_points': flag_10_mask.sum().item(),
        'flag_20_points': flag_20_mask.sum().item()
    }

    # Overall accuracy
    exact_matches = (prediction == ground_truth).sum().item()
    results['overall_accuracy'] = exact_matches / len(ground_truth)

    # Angle regression metrics (valid positions only)
    if valid_mask.any():
        pred_valid = prediction[valid_mask]
        gt_valid = ground_truth[valid_mask]

        valid_finite = torch.isfinite(pred_valid) & torch.isfinite(gt_valid)
        if valid_finite.any():
            pred_clean = pred_valid[valid_finite]
            gt_clean = gt_valid[valid_finite]

            results['angle_mse'] = nn.functional.mse_loss(pred_clean, gt_clean).item()
            results['angle_mae'] = nn.functional.l1_loss(pred_clean, gt_clean).item()
            results['angle_rmse'] = math.sqrt(results['angle_mse'])
            results['clean_points'] = valid_finite.sum().item()
        else:
            results.update({'angle_mse': float('inf'), 'angle_mae': float('inf'),
                          'angle_rmse': float('inf'), 'clean_points': 0})
    else:
        results.update({'angle_mse': 0.0, 'angle_mae': 0.0, 'angle_rmse': 0.0, 'clean_points': 0})

    # Flag classification accuracy
    results['flag_10_acc'] = 1.0 if not flag_10_mask.any() else (prediction[flag_10_mask] == -10.0).float().mean().item()
    results['flag_20_acc'] = 1.0 if not flag_20_mask.any() else (prediction[flag_20_mask] == -20.0).float().mean().item()

    flag_mask = flag_10_mask | flag_20_mask
    if flag_mask.any():
        flag_correct = ((prediction[flag_10_mask] == -10.0).sum() +
                       (prediction[flag_20_mask] == -20.0).sum()).item()
        results['flag_acc'] = flag_correct / flag_mask.sum().item()
    else:
        results['flag_acc'] = 1.0

    return results

def load_terrain_coordinates():
    """Load terrain chunks for visualization (optional)."""
    try:
        chunks = []
        current_chunk = []

        with open("/Users/farhang/Downloads/fls_2d_terrain.csv", 'r') as file:
            for line in file:
                line = line.strip()
                if line:
                    current_chunk.append(line)
                else:
                    if current_chunk:
                        chunks.append(current_chunk)
                        current_chunk = []
            if current_chunk:
                chunks.append(current_chunk)

        terrain_data = {}
        for chunk_idx, chunk in enumerate(chunks):
            chunk_data = []
            for line in chunk:
                parts = line.split(',') if ',' in line else line.split()
                chunk_data.append([float(x.strip()) for x in parts])

            if chunk_data:
                df = pd.DataFrame(chunk_data, columns=['timestamp', 'x', 'z', 'tangent_angle', 'incident_angle', 'normal_angle'])
                terrain_data[chunk_idx] = {
                    'x': df['x'].values, 'z': df['z'].values,
                    'tangent': df['tangent_angle'].values, 'normal': df['normal_angle'].values
                }

        return terrain_data
    except FileNotFoundError:
        print("Warning: Terrain file not found, skipping terrain data")
        return {}

def save_predictions_to_csv(tangent_results, phi_results, original_csv_path, output_csv_path):
    """
    Save predictions maintaining correct structure:
    Column 1: timestamp
    Columns 2-669: intensities (668 columns) - KEEP ORIGINAL
    Columns 670-3341: tangents (2672 columns) - REPLACE WITH PREDICTIONS
    Columns 3342-6013: phis (2672 columns) - REPLACE WITH PREDICTIONS
    """
    # Read the original CSV data
    original_data = []
    with open(original_csv_path, 'r', newline='') as f:
        reader = csv.reader(f)
        for row in reader:
            # Convert strings to floats where applicable
            processed_row = []
            for val in row:
                try:
                    processed_row.append(float(val))
                except ValueError:
                    processed_row.append(val)
            original_data.append(processed_row)

    # Create output data by copying original data
    output_data = [row[:] for row in original_data]  # Deep copy

    for row_idx in tangent_results.keys():
        if row_idx < len(output_data) and row_idx in phi_results:
            pred_tangents = tangent_results[row_idx]['pred_tangents']  # Shape: [4, 668]
            pred_phis = phi_results[row_idx]['pred_phis']              # Shape: [4, 668]

            # KEEP timestamp (column 0) and intensities (columns 1-668) unchanged

            # Replace tangent columns (columns 669-3340, which is 2672 columns)
            tangent_start = 669  # After timestamp + 668 intensities
            tangent_flat = pred_tangents.flatten()  # Convert [4,668] to [2672]

            for i, val in enumerate(tangent_flat):
                col_idx = tangent_start + i
                if col_idx < len(output_data[row_idx]):
                    output_data[row_idx][col_idx] = float(val)

            # Replace phi columns (columns 3341-6012, which is 2672 columns)
            phi_start = 669 + 2672  # After timestamp + intensities + tangents
            phi_flat = pred_phis.flatten()  # Convert [4,668] to [2672]

            for i, val in enumerate(phi_flat):
                col_idx = phi_start + i
                if col_idx < len(output_data[row_idx]):
                    output_data[row_idx][col_idx] = float(val)

    # Write the predictions to CSV
    with open(output_csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        for row in output_data:
            writer.writerow(row)

    print(f"Predictions saved to {output_csv_path}")
    return output_data

# def save_splits_to_csv(csv_file, base_dir="./data_splits"):
#     """Save actual dataset splits to separate CSV files"""
#     os.makedirs(base_dir, exist_ok=True)

#     # Load the original CSV
#     df = pd.read_csv(csv_file, header=None)

#     # Create reproducible splits with fixed seed
#     train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42)
#     val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

#     # Save the actual data to CSV files
#     train_df.to_csv(f"{base_dir}/train_data.csv", index=False, header=False, na_rep='NaN')
#     val_df.to_csv(f"{base_dir}/val_data.csv", index=False, header=False, na_rep='NaN')
#     test_df.to_csv(f"{base_dir}/test_data.csv", index=False, header=False, na_rep='NaN')

#     print(f"Data splits saved to {base_dir}/")
#     print(f"Train: {len(train_df)} samples")
#     print(f"Val: {len(val_df)} samples")
#     print(f"Test: {len(test_df)} samples")

#     return f"{base_dir}/train_data.csv", f"{base_dir}/val_data.csv", f"{base_dir}/test_data.csv"

def save_splits_to_csv(csv_file, base_dir="./data_splits"):
    """Save actual dataset splits to separate CSV files"""
    os.makedirs(base_dir, exist_ok=True)

    # Read the raw file as text lines to preserve exact format
    with open(csv_file, 'r') as f:
        lines = f.readlines()

    # Remove any trailing newlines but preserve the data exactly
    lines = [line.rstrip('\n\r') for line in lines]

    # Create indices for splitting (same random seed logic)
    np.random.seed(42)
    indices = np.arange(len(lines))

    # Split indices instead of dataframe
    train_indices, temp_indices = train_test_split(indices, test_size=0.3, random_state=42)
    val_indices, test_indices = train_test_split(temp_indices, test_size=1/3, random_state=42)

    # Write lines directly without pandas processing
    with open(f"{base_dir}/train_data.csv", 'w') as f:
        for idx in train_indices:
            f.write(lines[idx] + '\n')

    with open(f"{base_dir}/val_data.csv", 'w') as f:
        for idx in val_indices:
            f.write(lines[idx] + '\n')

    with open(f"{base_dir}/test_data.csv", 'w') as f:
        for idx in test_indices:
            f.write(lines[idx] + '\n')

    print(f"Data splits saved to {base_dir}/")
    print(f"Train: {len(train_indices)} samples")
    print(f"Val: {len(val_indices)} samples")
    print(f"Test: {len(test_indices)} samples")

    return f"{base_dir}/train_data.csv", f"{base_dir}/val_data.csv", f"{base_dir}/test_data.csv"

def save_splits_with_indices_to_csv(csv_file, base_dir="./data_splits"):
    """Create test_data_with_indices.csv from existing split by recreating the same split"""
    os.makedirs(base_dir, exist_ok=True)

    # Read the raw file as text lines to preserve exact format
    with open(csv_file, 'r') as f:
        lines = f.readlines()

    # Remove any trailing newlines but preserve the data exactly
    lines = [line.rstrip('\n\r') for line in lines]

    # Create indices for splitting (same random seed logic)
    np.random.seed(42)
    indices = np.arange(len(lines))

    # Split indices instead of dataframe - EXACT same logic as original function
    train_indices, temp_indices = train_test_split(indices, test_size=0.3, random_state=42)
    val_indices, test_indices = train_test_split(temp_indices, test_size=1/3, random_state=42)

    # Write ONLY test split with original indices prepended to each line
    #Uncomment to save for test
    with open(f"{base_dir}/test_data_with_indices.csv", 'w') as f:
        for idx in test_indices:
            # Prepend original line index to the line content
            f.write(f"{idx},{lines[idx]}\n")

    #uncomment to save for train
    # with open(f"{base_dir}/train_data_with_indices.csv", 'w') as f:
    #     for idx in train_indices:
    #         f.write(f"{idx},{lines[idx]}\n")

    print(f"Created test_data_with_indices.csv with {len(test_indices)} samples")

    #uncomment to save for train
    # return f"{base_dir}/train_data_with_indices.csv"

    #uncomment to save for test
    return f"{base_dir}/test_data_with_indices.csv"



def save_test_indices_vs_original_pcl_plots(test_csv_path, original_csv_path, output_dir="./test_indices_vs_original",
                                          range_resolution=0.05988024, intensity_threshold=0.1, azimuth=0.0):
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

    print("First 5 test indices:", test_indices[:5])

    # Plot only these specific indices
    for i, original_row_idx in enumerate(test_indices):
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
            orig_x, orig_z = extract_pcl_points_from_row(original_row_data, range_resolution, intensity_threshold, azimuth, has_indices=True)
            test_x, test_z = extract_pcl_points_from_row(test_row_data, range_resolution, intensity_threshold, azimuth, has_indices=True)

            # Plot comparison
            if len(orig_x) > 0 or len(test_x) > 0:
                plt.figure(figsize=(15, 8))

                # Original data (blue)
                if len(orig_x) > 0:
                    plt.scatter(orig_x, orig_z, c='blue', s=3, alpha=0.8, label=f'Original Row {original_row_idx}')

                # Test data (red)
                if len(test_x) > 0:
                    plt.scatter(test_x, test_z, c='red', s=2, alpha=0.6, label=f'Test Split Row {original_row_idx}')

                plt.xlabel('X (meters)')
                plt.ylabel('Z (meters)')
                plt.title(f'Original vs Test Split - Row Index {original_row_idx}')
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.axis('equal')

                # Add lines from sensor to show measurement directions (optional)
                plt.axhline(y=0, color='gray', linestyle='--', alpha=0.3)
                plt.axvline(x=0, color='gray', linestyle='--', alpha=0.3)

                plt.savefig(f"{output_dir}/row_index_{int(original_row_idx):04d}.png", dpi=300, bbox_inches='tight')
                plt.close()

                # if i < 5:  # Print first few for verification
                #     print(f"Plotted row index {original_row_idx}: Original={len(orig_x)}, Test={len(test_x)} points")

        except Exception as e:
            print(f"Error processing row {i} (original index {original_row_idx}): {e}")
            continue

    print(f"Finished plotting {len(test_indices)} test rows")



def save_predictions_with_indices_to_csv(tangent_results, phi_results, test_csv_path, output_csv_path):
    """
    Save predictions with original indices for comparison plotting.
    This creates a CSV with predictions that can be compared against original data.
    Uses pure text processing instead of pandas.
    """
    # Read the test CSV data
    test_data = []
    with open(test_csv_path, 'r', newline='') as f:
        reader = csv.reader(f)
        for row in reader:
            # Convert strings to floats where applicable, keep original index as string/int
            processed_row = []
            for i, val in enumerate(row):
                if i == 0:  # Keep original index as is
                    processed_row.append(val)
                else:
                    try:
                        processed_row.append(float(val))
                    except ValueError:
                        processed_row.append(val)
            test_data.append(processed_row)

    print(f"Creating predictions CSV with {len(test_data)} test samples...")

    # Create output data by copying test data
    output_data = [row[:] for row in test_data]  # Deep copy

    # Replace predictions for each test row
    for i in range(len(test_data)):
        if i in tangent_results and i in phi_results:
            # Get predictions for this test sample
            pred_tangents = tangent_results[i]['pred_tangents']  # Shape: [4, 668]
            pred_phis = phi_results[i]['pred_phis']              # Shape: [4, 668]

            # The test CSV has structure: [original_index, timestamp, intensities(668), tangents(2672), phis(2672)]
            # Keep original_index (col 0), timestamp (col 1), and intensities (cols 2-669) unchanged

            # Replace tangent columns (cols 670-3341, accounting for the extra index column)
            tangent_start = 670  # After index + timestamp + 668 intensities
            tangent_flat = pred_tangents.flatten()  # Convert [4,668] to [2672]

            for j, val in enumerate(tangent_flat):
                col_idx = tangent_start + j
                if col_idx < len(output_data[i]):
                    output_data[i][col_idx] = float(val)

            # Replace phi columns (cols 3342-6013, accounting for the extra index column)
            phi_start = 670 + 2672  # After index + timestamp + intensities + tangents
            phi_flat = pred_phis.flatten()  # Convert [4,668] to [2672]

            for j, val in enumerate(phi_flat):
                col_idx = phi_start + j
                if col_idx < len(output_data[i]):
                    output_data[i][col_idx] = float(val)

    # Write the predictions with indices to CSV
    with open(output_csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        for row in output_data:
            writer.writerow(row)

    print(f"Predictions with indices saved to {output_csv_path}")
    return output_data

def extract_pcl_points_from_row(row_data, range_resolution, intensity_threshold, azimuth, has_indices=False):
    """Extract x,z points using correct range calculation - pure text processing"""

    # Split CSV row into values
    if isinstance(row_data, str):
        values = row_data.split(',')
    else:
        # Handle case where it's already a list
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
        # Skip first column (index) in test data
        intensities = numeric_values[2:670]  # columns 2-669
        phis = numeric_values[3342:6014]     # columns 3342-6013
    else:
        intensities = numeric_values[1:669]  # columns 1-668
        phis = numeric_values[3341:6013]     # columns 3341-6012

    x_points = []
    z_points = []

    for point_idx in range(668):
        reverse_idx = 667 - point_idx

        # Check if we have valid intensity data
        if point_idx < len(intensities):
            intensity = intensities[point_idx]
        else:
            continue

        # if not pd.isna(intensity) and intensity > intensity_threshold:

        # Range calculation using the full range span
        max_range = 40.0
        min_range = 0.5
        range_val = min_range + (reverse_idx / 668) * (max_range - min_range)

        phi_start_idx = point_idx * 4

        for beam_idx in range(4):
            phi_idx = phi_start_idx + beam_idx

            # Check if phi index is within bounds
            if phi_idx < len(phis):
                phi_rad = phis[phi_idx]
            else:
                continue

            if not pd.isna(phi_rad) and phi_rad not in [-10.0, -20.0]:
                # Calculate 3D coordinates
                x = range_val * np.cos(azimuth) * np.cos(phi_rad)
                y = range_val * np.sin(azimuth) * np.cos(phi_rad)
                z = range_val * np.sin(phi_rad)

                x_points.append(x)
                z_points.append(z)

    return x_points, z_points

# ==================== MODEL ARCHITECTURE ====================

class ResidualBlock1D(nn.Module):
    def __init__(self, channels, dropout_rate=0.0):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(channels, channels, 3, padding=1),
            nn.InstanceNorm1d(channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Conv1d(channels, channels, 3, padding=1),
            nn.InstanceNorm1d(channels),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.block(x) + x)

class IntensityToBathymetryUNet1D(nn.Module):
    def __init__(self, prediction_type='phi', dropout_rate=0.1):
        super().__init__()
        self.prediction_type = prediction_type

        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv1d(1, 64, 7, padding=3),
            nn.InstanceNorm1d(64), nn.ReLU()
        )
        self.enc2 = nn.Sequential(
            nn.Conv1d(64, 128, 5, stride=2, padding=2),
            nn.InstanceNorm1d(128), nn.ReLU()
        )
        self.enc3 = nn.Sequential(
            nn.Conv1d(128, 256, 3, stride=2, padding=1),
            nn.InstanceNorm1d(256), nn.ReLU()
        )

        # Bottleneck
        self.residual_blocks = nn.ModuleList([
            ResidualBlock1D(256, dropout_rate) for _ in range(5)
        ])

        # Decoder with skip connections
        self.dec1 = nn.Sequential(
            nn.ConvTranspose1d(256, 128, 3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm1d(128), nn.ReLU()
        )
        self.dec2 = nn.Sequential(
            nn.ConvTranspose1d(256, 64, 5, stride=2, padding=2, output_padding=1),
            nn.InstanceNorm1d(64), nn.ReLU()
        )

        # Final upsampling
        self.final_upsample = nn.Sequential(
            nn.ConvTranspose1d(128, 32, 7, stride=4, padding=3, output_padding=3),
            nn.ReLU(),
        )

        # Dual heads for classification and regression
        self.classifier = nn.Conv1d(32, 3, 3, padding=1)  # 3 classes: valid, -10, -20

        # Output dimensions based on prediction type
        if prediction_type == 'combined':
            self.regressor = nn.Conv1d(32, 2, 3, padding=1)  # tangent + phi values
        else:
            self.regressor = nn.Conv1d(32, 1, 3, padding=1)   # single angle values

    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size, 1, 668)

        # Encoder with skip connections
        e1 = self.enc1(x)      # [B, 64, 668]
        e2 = self.enc2(e1)     # [B, 128, 334]
        e3 = self.enc3(e2)     # [B, 256, 167]

        # Bottleneck
        b = e3
        for block in self.residual_blocks:
            b = block(b)

        # Decoder with skip connections
        d1 = self.dec1(b)                    # [B, 128, 334]
        d1 = torch.cat([d1, e2], dim=1)      # [B, 256, 334]

        d2 = self.dec2(d1)                   # [B, 64, 668]
        d2 = torch.cat([d2, e1], dim=1)      # [B, 128, 668]

        # Final features
        features = self.final_upsample(d2)   # [B, 32, 2672]

        class_logits = self.classifier(features)  # [B, 3, 2672]
        angle_pred = self.regressor(features)     # [B, 1, 2672] or [B, 2, 2672]

        return class_logits, angle_pred.squeeze(1) if angle_pred.size(1) == 1 else angle_pred

# ==================== DATA HANDLING ====================

class BathymetryDataset(Dataset):
    def __init__(self, csv_file, prediction_type='phi'):
        self.data = pd.read_csv(csv_file)
        self.prediction_type = prediction_type

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        # Extract intensities (always the same)
        intensities = torch.tensor(row.iloc[1:669].values, dtype=torch.float32)
        intensities_processed, _ = create_valid_mask(intensities)

        # Extract target data based on prediction type
        if self.prediction_type == 'phi':
            target_data = torch.tensor(row.iloc[3341:6013].values, dtype=torch.float32)
        elif self.prediction_type == 'tangent':
            target_data = torch.tensor(row.iloc[669:3341].values, dtype=torch.float32)
        elif self.prediction_type == 'combined':
            tangents = torch.tensor(row.iloc[669:3341].values, dtype=torch.float32)
            phis = torch.tensor(row.iloc[3341:6013].values, dtype=torch.float32)
            # Combine tangents and phis for combined training
            tangents_processed, _ = create_valid_mask(tangents)
            phis_processed, _ = create_valid_mask(phis)
            target_data = torch.stack([
                tangents_processed.view(4, 668),
                phis_processed.view(4, 668)
            ], dim=0).view(8, 668).transpose(0, 1).contiguous().view(-1)
        else:
            raise ValueError(f"Unknown prediction_type: {self.prediction_type}")

        # Process target data for non-combined types
        if self.prediction_type != 'combined':
            target_processed, _ = create_valid_mask(target_data)
            ground_truth = target_processed.view(4, 668).contiguous().view(-1)
        else:
            ground_truth = target_data

        return intensities_processed, ground_truth

def create_data_splits(dataset, test_indices=None, train_ratio=0.7, val_ratio=0.2, seed=42):
    """Split dataset into train/validation/test sets."""
    torch.manual_seed(seed)
    dataset_size = len(dataset)

    if test_indices is not None:
        remaining_indices = [i for i in range(dataset_size) if i not in test_indices]
        remaining_size = len(remaining_indices)
        train_size = int(train_ratio * remaining_size / (train_ratio + val_ratio))

        np.random.seed(seed)
        np.random.shuffle(remaining_indices)
        train_indices = remaining_indices[:train_size]
        val_indices = remaining_indices[train_size:]

        return (Subset(dataset, train_indices),
                Subset(dataset, val_indices),
                Subset(dataset, test_indices))
    else:
        train_size = int(train_ratio * dataset_size)
        val_size = int(val_ratio * dataset_size)
        test_size = dataset_size - train_size - val_size
        return random_split(dataset, [train_size, val_size, test_size])

# ==================== TRAINING & EVALUATION ====================

class SequenceBathymetryLoss(nn.Module):
    def __init__(self, reg_weight=1.0):
        super().__init__()
        self.reg_weight = reg_weight

    def forward(self, class_logits, angle_preds, targets):
        batch_size, seq_len = targets.shape

        # Create class labels: 0=valid, 1=flag_-10, 2=flag_-20
        valid_mask = (targets != -10) & (targets != -20)
        flag_10_mask = (targets == -10)
        flag_20_mask = (targets == -20)

        class_labels = torch.zeros_like(targets, dtype=torch.long)
        class_labels[flag_10_mask] = 1
        class_labels[flag_20_mask] = 2
        print("Classs Label shape",class_labels.shape)

        # Classification loss for all positions
        class_logits = class_logits.permute(0, 2, 1)  # [B, 2672, 3]
        class_loss = F.cross_entropy(
            class_logits.reshape(-1, 3),
            class_labels.reshape(-1)
        )
        print("Classs Logit shape",class_logits.shape)

        # Regression loss only for valid positions
        reg_loss = F.mse_loss(angle_preds[valid_mask], targets[valid_mask]) if valid_mask.any() else torch.tensor(0.0, device=targets.device)

        return class_loss + self.reg_weight * reg_loss

def run_epoch(model, dataloader, device, optimizer=None):
    """Run one epoch of training or validation."""
    is_training = optimizer is not None
    model.train() if is_training else model.eval()

    total_loss = 0.0
    valid_batches = 0
    context = nullcontext() if is_training else torch.no_grad()
    criterion = SequenceBathymetryLoss()

    with context:
        for intensities, ground_truth in dataloader:
            intensities, ground_truth = intensities.to(device), ground_truth.to(device)

            if is_training:
                optimizer.zero_grad()

            class_logits, angle_preds = model(intensities)
            loss = criterion(class_logits, angle_preds, ground_truth)

            if loss.item() > 0 and torch.isfinite(loss):
                if is_training:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                total_loss += loss.item()
                valid_batches += 1

    return total_loss / max(valid_batches, 1)

def train_model(model, train_loader, val_loader, num_epochs=300, model_name="bathymetry"):
    """Train the model with early stopping."""
    device = get_device()
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=1e-2, total_steps=num_epochs,
        pct_start=0.3, anneal_strategy='cos'
    )

    print(f"Training {model_name} on {device}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    best_val_loss = float('inf')
    patience_counter = 0
    patience = 30

    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        train_loss = run_epoch(model, train_loader, device, optimizer)
        val_loss = run_epoch(model, val_loader, device)
        scheduler.step()

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        # Early stopping with model-specific filename
        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), f'best_bathymetry_model_{model_name}.pth')
        else:
            patience_counter += 1

        if epoch % 10 == 0 or epoch < 5:
            print(f"Epoch {epoch+1:3d}/{num_epochs} | Train: {train_loss:.6f} | Val: {val_loss:.6f} | "
                  f"Best: {best_val_loss:.6f} | LR: {optimizer.param_groups[0]['lr']:.2e}")

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

    # Load best model
    model.load_state_dict(torch.load(f'best_bathymetry_model_{model_name}.pth', map_location=device))
    print(f"Training complete. Best validation loss: {best_val_loss:.6f}")

    # Save training curves with model-specific name
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss', alpha=0.8)
    plt.plot(val_losses, label='Validation Loss', alpha=0.8)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'{model_name.title()} Training and Validation Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f'training_curves_{model_name}.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Training curves saved: training_curves_{model_name}.png")

def evaluate_model(model, test_loader):
    """Evaluate model and return test loss."""
    device = get_device()
    model.to(device).eval()
    test_loss = run_epoch(model, test_loader, device)
    return test_loss

# ==================== INFERENCE & TESTING ====================

def predict_with_model(model, intensities, device):
    """Make predictions with trained model."""
    model.eval()
    if intensities.dim() == 1:
        intensities = intensities.unsqueeze(0)

    with torch.no_grad():
        class_logits, angle_preds = model(intensities.to(device))

    # Combine classification and regression outputs
    class_preds = torch.argmax(class_logits, dim=1).cpu()
    angle_preds = angle_preds.cpu()

    prediction = angle_preds.clone()

    class_preds_np = class_preds[0].numpy()  # shape: [2672]
    df = pd.DataFrame(class_preds_np, columns=['class_prediction'])
    df.to_csv("class_predictions_after_argmax.csv", index=False)

    prediction[class_preds == 1] = -10.0  # Flag -10 positions
    prediction[class_preds == 2] = -20.0  # Flag -20 positions

    return prediction, class_logits.cpu(), angle_preds

def test_specific_rows(model, dataset, num_test_rows, output_dir, prediction_type):
    """Test model on specific rows and save results."""
    ensure_dir(output_dir)
    device = get_device()
    model.to(device).eval()

    test_indices = list(range(min(num_test_rows, len(dataset))))
    results = {}

    print(f"\nTesting {len(test_indices)} samples for {prediction_type}...")

    with torch.no_grad():
        for i in test_indices:
            intensities, ground_truth_raw = dataset[i]
            actual_idx = dataset.indices[i] if hasattr(dataset, 'indices') else i

            # Make prediction
            prediction, class_logits, angle_preds = predict_with_model(model, intensities, device)
            prediction = prediction.squeeze(0)

            # Get predicted classes
            class_preds = torch.argmax(class_logits, dim=1).squeeze(0)  # [2672]

            # Create ground truth classes
            gt_classes = torch.zeros_like(ground_truth_raw, dtype=torch.long)
            gt_classes[ground_truth_raw == -10.0] = 1
            gt_classes[ground_truth_raw == -20.0] = 2

            # Reshape classes for each beam (4 beams)
            pred_classes_reshaped = torch.zeros(4, 668)
            gt_classes_reshaped = torch.zeros(4, 668)
            for j in range(4):
                pred_classes_reshaped[j, :] = class_preds[j::4]
                gt_classes_reshaped[j, :] = gt_classes[j::4]

            # Evaluate
            row_metrics = evaluate_single_row(prediction, ground_truth_raw)

            # Print clean summary
            print(f"Sample {i:2d} (CSV row {actual_idx:4d}) | "
                  f"Angle RMSE: {row_metrics['angle_rmse']:6.3f} | "
                  f"Flag Acc: {row_metrics['flag_acc']:5.1%} | "
                  f"Valid: {row_metrics['clean_points']:3d}/{row_metrics['valid_points']:3d}")

            # Reshape for visualization - USE THE PREDICTION_TYPE PARAMETER
            pred_tangents, pred_phis = reshape_predictions(prediction, prediction_type=prediction_type)
            gt_processed, _ = create_valid_mask(ground_truth_raw)
            gt_tangents, gt_phis = reshape_predictions(gt_processed, prediction_type=prediction_type)

            # Store results with separate class predictions for phi and tangent
            results[i] = {
                'intensities': intensities.numpy(),
                'pred_tangents': pred_tangents.numpy(),
                'pred_phis': pred_phis.numpy(),
                'gt_tangents': gt_tangents.numpy(),
                'gt_phis': gt_phis.numpy(),
                'classes_phi': class_logits.numpy(),
                'pred_classes_phi': pred_classes_reshaped.numpy() if prediction_type == 'phi' else torch.zeros(4, 668).numpy(),
                'gt_classes_phi': gt_classes_reshaped.numpy() if prediction_type == 'phi' else torch.zeros(4, 668).numpy(),
                'pred_classes_tangent': pred_classes_reshaped.numpy() if prediction_type == 'tangent' else torch.zeros(4, 668).numpy(),
                'gt_classes_tangent': gt_classes_reshaped.numpy() if prediction_type == 'tangent' else torch.zeros(4, 668).numpy(),
                'metrics': row_metrics,
                'prediction_type': prediction_type
            }

            # Save CSV
            data = {'pixel_idx': range(668), 'intensities': intensities.numpy()}
            for j in range(4):
                if prediction_type in ['phi', 'combined']:
                    data[f'pred_phi_{j}'] = pred_phis.numpy()[j]
                    data[f'gt_phi_{j}'] = gt_phis.numpy()[j]
                if prediction_type in ['tangent', 'combined']:
                    data[f'pred_tangent_{j}'] = pred_tangents.numpy()[j]
                    data[f'gt_tangent_{j}'] = gt_tangents.numpy()[j]

            pd.DataFrame(data).to_csv(f'{output_dir}/row_{actual_idx}_predictions.csv', index=False)

    print(f"Results saved to {output_dir}/")
    return results

def visualize_predictions(results, terrain_data=None, show_plots=True):
    """Create visualization plots for predictions."""
    if not show_plots:
        return

    for i, result in results.items():
        fig, axes = plt.subplots(4, 4, figsize=(16, 9), sharex=True)

        pred_tangents = result['pred_tangents']
        pred_phis = result['pred_phis']
        first_column = pred_phis[3, 108]
        print("Predicted Phis--------", first_column)
        gt_tangents = result['gt_tangents']
        gt_phis = result['gt_phis']
        # range_val = min_range + (reverse_idx / 668) * (max_range - min_range)

        for j in range(4):
            # Tangents (disabled but structure preserved)
            axes[0, j].scatter(range(668), pred_tangents[j], s=9, alpha=0.7, label='pred')
            axes[0, j].scatter(range(668), gt_tangents[j], s=2, marker='x', alpha=0.7, label='gt')
            axes[0, j].set_title(f"Tangent {j+1}")
            if j == 0: axes[0, j].legend()

            # Phis
            axes[1, j].scatter(range(668), pred_phis[j], s=9, alpha=0.7, label='pred')
            axes[1, j].scatter(range(668), gt_phis[j], s=2, marker='x', alpha=0.7, label='gt')
            axes[1, j].set_title(f"Phi {j+1}")
            if j == 0: axes[1, j].legend()

            # Row 2: Classes comparison
            axes[2, j].scatter(range(668), result['pred_classes_phi'][j], s=9, alpha=0.7, label='pred class', color='red')
            axes[2, j].scatter(range(668), result['gt_classes_phi'][j], s=2, marker='x', alpha=0.7, label='gt class', color='blue')
            axes[2, j].set_title(f"Classes Phi {j+1}")
            axes[2, j].set_ylim(-0.5, 2.5)
            axes[2, j].set_yticks([0, 1, 2])
            if j == 0: axes[2, j].legend()

            axes[3, j].scatter(range(668), result['pred_classes_tangent'][j], s=9, alpha=0.7, label='pred class', color='red')
            axes[3, j].scatter(range(668), result['gt_classes_tangent'][j], s=2, marker='x', alpha=0.7, label='gt class', color='blue')
            axes[3, j].set_title(f"Classes Tangents {j+1}")
            axes[3, j].set_ylim(-0.5, 2.5)
            axes[3, j].set_yticks([0, 1, 2])
            if j == 0: axes[3, j].legend()

        plt.suptitle(f"Sample {i} - Predictions vs Ground Truth")
        plt.tight_layout()
        plt.show()

def visualize_terrain_comparison(results, terrain_data):
    """Create terrain-style visualization with real coordinates and angles."""
    if not terrain_data:
        print("No terrain data available for spatial visualization")
        return

    print(f"\n{'='*50}")
    print("TERRAIN SPATIAL VISUALIZATION")
    print(f"{'='*50}")

# ==================== MAIN EXECUTION ====================

def main():
    # Setup
    csv_file = '/Users/farhang/Downloads/fls_all_with_phi.csv'

    # Create datasets for different prediction types
    phi_dataset = BathymetryDataset(csv_file, prediction_type='phi')
    tangent_dataset = BathymetryDataset(csv_file, prediction_type='tangent')

    print(f"Phi Dataset loaded: {len(phi_dataset)} samples")
    print(f"Tangent Dataset loaded: {len(tangent_dataset)} samples")

    # Handle data splits
    splits_dir = "./data_splits"
    if os.path.exists(f"{splits_dir}/train_data.csv"):
        print("Using existing CSV splits...")
        train_csv = f"{splits_dir}/train_data.csv"
        val_csv = f"{splits_dir}/val_data.csv"
        test_csv = f"{splits_dir}/test_data.csv"
    else:
        print("Creating new splits and saving to CSV...")
        train_csv, val_csv, test_csv = save_splits_to_csv(csv_file, splits_dir)



    # Create datasets for each split and prediction type
    datasets = {}
    for pred_type in ['phi', 'tangent']:
        datasets[pred_type] = {
            'train': BathymetryDataset(train_csv, prediction_type=pred_type),
            'val': BathymetryDataset(val_csv, prediction_type=pred_type),
            'test': BathymetryDataset(test_csv, prediction_type=pred_type)
        }

    print(f"Phi splits - Train: {len(datasets['phi']['train'])}, Val: {len(datasets['phi']['val'])}, Test: {len(datasets['phi']['test'])}")
    print(f"Tangent splits - Train: {len(datasets['tangent']['train'])}, Val: {len(datasets['tangent']['val'])}, Test: {len(datasets['tangent']['test'])}")

    # Create data loaders
    loaders = {}
    for pred_type in ['phi', 'tangent']:
        loaders[pred_type] = {
            'train': DataLoader(datasets[pred_type]['train'], batch_size=32, shuffle=True),
            'val': DataLoader(datasets[pred_type]['val'], batch_size=32, shuffle=False),
            'test': DataLoader(datasets[pred_type]['test'], batch_size=32, shuffle=False)
        }

    # Initialize and train models
    models = {}
    for pred_type in ['phi', 'tangent']:
        print(f"\nInitializing {pred_type.upper()} network...")
        model = IntensityToBathymetryUNet1D(prediction_type=pred_type)

        # Training (uncomment to train)
        print(f"Training {pred_type.upper()} network...")
        train_model(model, loaders[pred_type]['train'], loaders[pred_type]['val'],
                   num_epochs=100, model_name=pred_type)

        # Load existing model
        try:
            model.load_state_dict(torch.load(f'best_bathymetry_model_{pred_type}.pth', map_location=get_device()))
            print(f"Loaded existing {pred_type.upper()} model")
        except FileNotFoundError:
            print(f"No existing {pred_type.upper()} model found")

        models[pred_type] = model

    # Evaluation
    print(f"\n{'='*50}")
    print("MODEL EVALUATION")
    print(f"{'='*50}")

    for pred_type in ['phi', 'tangent']:
        test_loss = evaluate_model(models[pred_type], loaders[pred_type]['test'])
        print(f"{pred_type.title()} Test Loss: {test_loss:.6f}")

    # Test specific rows
    results = {}
    for pred_type in ['phi', 'tangent']:
        print(f"\n{'='*50}")
        print(f"DETAILED TESTING - {pred_type.upper()} NETWORK")
        print(f"{'='*50}")

        results[pred_type] = test_specific_rows(
            models[pred_type],
            datasets[pred_type]['test'],
            # num_test_rows=len(datasets[pred_type]['test']),
            num_test_rows= 1,
            output_dir=f'test_outputs_{pred_type}',
            prediction_type=pred_type
        )

    # Visualization (uncomment to show plots)
    print("Visualizing PHI predictions...")
    visualize_predictions(results['phi'], show_plots=True)
    # print("Visualizing TANGENT predictions...")
    # visualize_predictions(results['tangent'], show_plots=True)

    # # Save models
    # ensure_dir('./models')
    # for pred_type in ['phi', 'tangent']:
    #     model_path = f'./models/bathymetry_cnn_model_{pred_type}.pth'
    #     torch.save(models[pred_type].state_dict(), model_path)
    #     print(f"{pred_type.title()} model saved: {model_path}")

    # # Save predictions to CSV
    # original_csv_path = "/Users/farhang/Downloads/fls_all_with_phi.csv"
    # output_csv_path = "/Users/farhang/Downloads/fls_2d_terrain_prediction_output.csv"
    # save_predictions_to_csv(results['tangent'], results['phi'], original_csv_path, output_csv_path)

    # test_with_indices = save_splits_with_indices_to_csv(csv_file, splits_dir)
    # train_with_indices = save_splits_with_indices_to_csv(csv_file, splits_dir)

    # predictions_with_indices_path = "./data_splits/test_predictions_with_indices.csv"
    # save_predictions_with_indices_to_csv(
    #     results['tangent'],
    #     results['phi'],
    #     test_with_indices,
    #     predictions_with_indices_path
    # )

    # predictions_with_indices_path = "./data_splits/train_predictions_with_indices.csv"
    # save_predictions_with_indices_to_csv(
    #     results['tangent'],
    #     results['phi'],
    #     train_with_indices,
    #     predictions_with_indices_path
    # )

    # # Generate comparison plots
    # save_test_indices_vs_original_pcl_plots(
    #     test_csv_path=predictions_with_indices_path,
    #     original_csv_path="/Users/farhang/Downloads/fls_all_with_phi.csv",
    #     output_dir="./test_indices_vs_original"
    # )

if __name__ == "__main__":
    main()