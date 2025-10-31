from xml.parsers.expat import model
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split, Subset, WeightedRandomSampler

import pandas as pd
import numpy as np
import math
import os
from contextlib import nullcontext
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib

import csv

from check_separability import visualize_class_separability
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

def create_balanced_sampler(dataset):
    # Get all labels from dataset
    labels = []
    for i in range(len(dataset)):
        _, label = dataset[i]  # Assuming dataset returns (data, label)

        # Handle different label formats
        if torch.is_tensor(label):
            if label.numel() == 1:
                labels.append(label.item())
            else:
                labels.append(label.argmax().item())  # For one-hot encoded
        else:
            labels.append(int(label))

    labels = torch.tensor(labels)

    # Count samples per class
    class_counts = torch.bincount(labels)

    # Weight inversely proportional to class frequency
    class_weights = 1.0 / class_counts.float()

    # Assign weight to each sample
    sample_weights = class_weights[labels]

    # Create sampler
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )

    return sampler

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

def prepare_3class_data(dataset):
    """Prepare data for XGBoost with spatial context"""
    X_list = []
    y_list = []

    for i in range(len(dataset)):
        intensities, targets = dataset[i]

        X = intensities.numpy()  # [668]
        y = targets.numpy()      # [2672]

        # Create class labels
        y_classes = np.zeros(len(y), dtype=int)
        y_classes[y == -10] = 1
        y_classes[y == -20] = 2

        # For each of 2672 predictions, create feature vector with spatial context
        for j in range(2672):
            intensity_idx = j // 4

            # Feature: intensity at this position + neighbors (5-point window)
            start = max(0, intensity_idx - 2)
            end = min(668, intensity_idx + 3)
            features = X[start:end]

            # Pad to fixed size (5 features)
            if len(features) < 5:
                features = np.pad(features, (0, 5 - len(features)), mode='edge')

            X_list.append(features)
            y_list.append(y_classes[j])

    return np.array(X_list), np.array(y_list)

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

def save_splits_with_indices_to_csv(csv_file, base_dir="./data_splits", split_type='test'):
    """Create data_with_indices.csv from existing split by recreating the same split"""
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

    # Select the right indices based on split_type
    if split_type == 'train':
        selected_indices = train_indices
    elif split_type == 'val':
        selected_indices = val_indices
    else:  # 'test'
        selected_indices = test_indices

    # Write selected split with original indices prepended
    output_file = f"{base_dir}/{split_type}_data_with_indices.csv"
    with open(output_file, 'w') as f:
        for idx in selected_indices:
            f.write(f"{idx},{lines[idx]}\n")

    print(f"Created {split_type}_data_with_indices.csv with {len(selected_indices)} samples")
    return output_file



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
            nn.LeakyReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Conv1d(channels, channels, 3, padding=1),
            nn.InstanceNorm1d(channels),
        )
        self.relu = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.block(x) + x)


# ==================== MLP INTENSITY-BASED MODELS ====================

class IntensityToValidityMLP(nn.Module):
    """
    Lightweight MLP that learns intensity → valid/invalid mapping.
    Generalizes better than hardcoded threshold.
    """
    def __init__(self, hidden_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, intensities):
        """
        Args:
            intensities: [B, 668] pixel intensities
        Returns:
            validity_logits: [B, 2672] logits for each beam (4 per pixel)
        """
        batch_size = intensities.size(0)

        # Expand: each pixel has 4 beams
        intensities_expanded = intensities.repeat_interleave(4, dim=1)  # [B, 2672]

        # Reshape for MLP: [B*2672, 1]
        intensities_flat = intensities_expanded.reshape(-1, 1)

        # Predict validity logits
        logits_flat = self.net(intensities_flat)  # [B*2672, 1]

        # Reshape back: [B, 2672]
        validity_logits = logits_flat.reshape(batch_size, -1)

        return validity_logits


class MLPCombinedModel(nn.Module):
    """
    Combined MLP model for validity classification and angle regression.
    """
    def __init__(self, prediction_type='phi', dropout_rate=0.1):
        super().__init__()

        # Lightweight classifier (learns threshold function)
        self.validity_classifier = IntensityToValidityMLP(hidden_dim=32)

        # Angle regressor (your existing model)
        self.angle_regressor = RegressorCNN(prediction_type, dropout_rate)

    def forward(self, intensities, training=True):
        validity_logits = self.validity_classifier(intensities)  # Keep as logits
        validity_probs = torch.sigmoid(validity_logits)
        angle_predictions = self.angle_regressor(intensities)

        if training:
            final_predictions = validity_probs * angle_predictions + (1 - validity_probs) * (-20.0)
        else:
            final_predictions = angle_predictions.clone()
            final_predictions[validity_probs < 0.5] = -20.0

        return final_predictions, validity_logits, angle_predictions  # Return logits, not probs

class MLPCombinedLoss(nn.Module):
    def __init__(self, alpha=10.0, beta=1.0):  # Changed: alpha >> beta
        super().__init__()
        self.alpha = alpha  # INCREASE classification importance
        self.beta = beta    # DECREASE regression importance

    def forward(self, final_preds, validity_probs, angle_preds, targets):
        target_validity = ((targets != -10) & (targets != -20)).float()

        # Add class weights to BCE
        pos_weight = torch.tensor([30.0]).to(targets.device)  # Weight valid class more
        cls_loss = F.binary_cross_entropy_with_logits(
            validity_probs,
            target_validity,
            pos_weight=pos_weight
        )

        valid_mask = target_validity.bool()

        if valid_mask.any():
            reg_loss = F.mse_loss(angle_preds[valid_mask], targets[valid_mask])
        else:
            reg_loss = torch.tensor(0.0, device=targets.device)

        total_loss = self.alpha * cls_loss + self.beta * reg_loss

        return total_loss, cls_loss, reg_loss


def train_mlp_model(model, train_loader, val_loader, num_epochs=50):
    """
    Train the combined MLP model
    """
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    model.to(device)

    # Separate optimizers for classifier and regressor
    optimizer = torch.optim.AdamW([
        {'params': model.validity_classifier.parameters(), 'lr': 1e-3},
        {'params': model.angle_regressor.parameters(), 'lr': 1e-4}
    ], weight_decay=1e-4)

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=[1e-3, 1e-3],
        total_steps=num_epochs,
        pct_start=0.3
    )

    criterion = MLPCombinedLoss(alpha=1.0, beta=100.0)

    print(f"\n{'='*60}")
    print("TRAINING MLP BATHYMETRY MODEL")
    print(f"{'='*60}")
    print(f"Device: {device}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_cls_loss = 0.0
        train_reg_loss = 0.0

        for intensities, targets in train_loader:
            intensities = intensities.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()

            # final_preds, validity_probs, angle_preds = model(intensities, training=True)
            # loss, cls_loss, reg_loss = criterion(final_preds, validity_probs, angle_preds, targets)

            final_preds, validity_logits, angle_preds = model(intensities, training=True)
            loss, cls_loss, reg_loss = criterion(final_preds, validity_logits, angle_preds, targets)

            if torch.isfinite(loss):
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                train_loss += loss.item()
                train_cls_loss += cls_loss.item()
                train_reg_loss += reg_loss.item()

        # Validation
        model.eval()
        val_loss = 0.0
        val_cls_loss = 0.0
        val_reg_loss = 0.0

        with torch.no_grad():
            for intensities, targets in val_loader:
                intensities = intensities.to(device)
                targets = targets.to(device)

                final_preds, validity_probs, angle_preds = model(intensities)
                loss, cls_loss, reg_loss = criterion(final_preds, validity_probs, angle_preds, targets)

                if torch.isfinite(loss):
                    val_loss += loss.item()
                    val_cls_loss += cls_loss.item()
                    val_reg_loss += reg_loss.item()

        train_loss /= len(train_loader)
        val_loss /= len(val_loader)

        scheduler.step()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_mlp_model.pth')

        if epoch % 10 == 0 or epoch < 5:
            print(f"Epoch {epoch+1:3d}/{num_epochs} | "
                  f"Train: {train_loss:.4f} (cls:{train_cls_loss/len(train_loader):.4f}, "
                  f"reg:{train_reg_loss/len(train_loader):.4f}) | "
                  f"Val: {val_loss:.4f} | Best: {best_val_loss:.4f}")

    model.load_state_dict(torch.load('best_mlp_model.pth', map_location=device))
    print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}\n")
    return model

# ==================== CLASSIFIER CNN (INDEPENDENT) ====================

class ClassifierCNN(nn.Module):
    """Independent CNN for classification only"""
    def __init__(self, dropout_rate=0.1):
        super().__init__()

        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv1d(1, 64, 15, padding=7),
            nn.InstanceNorm1d(64),
            nn.LeakyReLU()
        )
        self.enc2 = nn.Sequential(
            nn.Conv1d(64, 128, 11, stride=2, padding=5),
            nn.InstanceNorm1d(128),
            nn.LeakyReLU()
        )
        self.enc3 = nn.Sequential(
            nn.Conv1d(128, 256, 9, stride=2, padding=4),
            nn.InstanceNorm1d(256),
            nn.LeakyReLU()
        )

        # Bottleneck
        self.residual_blocks = nn.ModuleList([
            ResidualBlock1D(256, dropout_rate) for _ in range(5)
        ])

        # Decoder with skip connections
        self.dec1 = nn.Sequential(
            nn.ConvTranspose1d(256, 128, 9, stride=2, padding=4, output_padding=1),
            nn.InstanceNorm1d(128),
            nn.LeakyReLU()
        )
        self.dec2 = nn.Sequential(
            nn.ConvTranspose1d(256, 64, 11, stride=2, padding=5, output_padding=1),
            nn.InstanceNorm1d(64),
            nn.LeakyReLU()
        )

        # Final upsampling
        self.final_upsample = nn.Sequential(
            nn.ConvTranspose1d(128, 32, 15, stride=4, padding=7, output_padding=3),
            nn.LeakyReLU(),
        )

        # Classification head
        self.classifier = nn.Conv1d(32, 3, 3, padding=1)  # 3 classes: valid, -10, -20

    def forward(self, x):
        """
        Args:
            x: [B, 668] or [B, 1, 668] intensity input
        Returns:
            class_logits: [B, 3, 2672]
        """
        if x.dim() == 2:
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

        # Classification output
        class_logits = self.classifier(features)  # [B, 3, 2672]

        return class_logits


class SimpleClassifier(nn.Module):
    """The one that worked - 21.9% Valid recall"""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=11, padding=5)
        self.conv2 = nn.Conv1d(32, 3, kernel_size=3, padding=1)
        self.upsample = nn.Upsample(size=2672, mode='linear', align_corners=False)

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        x = F.relu(self.conv1(x))
        x = self.conv2(x)
        x = self.upsample(x)
        return x

# ==================== REGRESSOR CNN (INDEPENDENT) ====================

class RegressorCNN(nn.Module):
    """Independent CNN for regression only"""
    def __init__(self, prediction_type='phi', dropout_rate=0.1):
        super().__init__()
        self.prediction_type = prediction_type

        # Encoder (separate from classifier)
        self.enc1 = nn.Sequential(
            nn.Conv1d(1, 64, 15, padding=7),
            nn.InstanceNorm1d(64),
            nn.LeakyReLU()
        )
        self.enc2 = nn.Sequential(
            nn.Conv1d(64, 128, 11, stride=2, padding=5),
            nn.InstanceNorm1d(128),
            nn.LeakyReLU()
        )
        self.enc3 = nn.Sequential(
            nn.Conv1d(128, 256, 9, stride=2, padding=4),
            nn.InstanceNorm1d(256),
            nn.LeakyReLU()
        )

        # Bottleneck (separate from classifier)
        self.residual_blocks = nn.ModuleList([
            ResidualBlock1D(256, dropout_rate) for _ in range(5)
        ])

        # Decoder with skip connections (separate from classifier)
        self.dec1 = nn.Sequential(
            nn.ConvTranspose1d(256, 128, 9, stride=2, padding=4, output_padding=1),
            nn.InstanceNorm1d(128),
            nn.LeakyReLU()
        )
        self.dec2 = nn.Sequential(
            nn.ConvTranspose1d(256, 64, 11, stride=2, padding=5, output_padding=1),
            nn.InstanceNorm1d(64),
            nn.LeakyReLU()
        )

        # Final upsampling (separate from classifier)
        self.final_upsample = nn.Sequential(
            nn.ConvTranspose1d(128, 32, 15, stride=4, padding=7, output_padding=3),
            nn.LeakyReLU(),
        )

        # Regression head
        if prediction_type == 'combined':
            self.regressor = nn.Conv1d(32, 2, 3, padding=1)  # tangent + phi
        else:
            self.regressor = nn.Conv1d(32, 1, 3, padding=1)  # single angle

    def forward(self, x):
        """
        Args:
            x: [B, 668] or [B, 1, 668] intensity input
        Returns:
            angle_pred: [B, 2672] or [B, 2, 2672]
        """
        if x.dim() == 2:
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

        # Regression output
        angle_pred = self.regressor(features)
        return angle_pred.squeeze(1) if angle_pred.size(1) == 1 else angle_pred

# ==================== COMBINED MODEL (FOR INFERENCE) ====================

class CombinedBathymetryModel(nn.Module):
    """Wrapper that uses both trained models for inference"""
    def __init__(self, classifier_model, regressor_model):
        super().__init__()
        self.classifier = classifier_model
        self.regressor = regressor_model
        self.prediction_type = regressor_model.prediction_type

    def forward(self, x):
        """
        Args:
            x: [B, 668] intensity input
        Returns:
            class_logits: [B, 3, 2672]
            angle_pred: [B, 2672] or [B, 2, 2672]
        """
        class_logits = self.classifier(x)
        angle_pred = self.regressor(x)
        return class_logits, angle_pred


# ==================== SEPARATE LOSS FUNCTIONS ====================

class ClassifierLoss(nn.Module):
    """Loss for classifier training only"""
    def __init__(self, class_weights=None):
        super().__init__()
        if class_weights is None:
           class_weights = torch.tensor([100.0, 30.0, 1.0])
        self.register_buffer('class_weights', class_weights)

    def forward(self, class_logits, targets):
        """
        Args:
            class_logits: [B, 3, 2672]
            targets: [B, 2672] - ground truth angles with flags
        """
        # Create class labels: 0=valid, 1=flag_-10, 2=flag_-20
        flag_10_mask = (targets == -10)
        flag_20_mask = (targets == -20)

        class_labels = torch.zeros_like(targets, dtype=torch.long)
        class_labels[flag_10_mask] = 1
        class_labels[flag_20_mask] = 2

        # Classification loss
        class_logits = class_logits.permute(0, 2, 1)  # [B, 2672, 3]
        class_logits_flat = class_logits.reshape(-1, 3)
        class_labels_flat = class_labels.reshape(-1)

        # Focal Loss implementation
        gamma = 3.0  # Focusing parameter (higher = more focus on hard examples)

        # Get class weights
        alpha = self.class_weights.to(class_labels_flat.device)[class_labels_flat]

        # Calculate probabilities
        probs = F.softmax(class_logits_flat, dim=1)
        pt = probs.gather(1, class_labels_flat.unsqueeze(1)).squeeze(1)

        # Focal weight: down-weight easy examples
        focal_weight = alpha * (1 - pt) ** gamma

        # Cross entropy loss
        log_probs = F.log_softmax(class_logits_flat, dim=1)
        nll_loss = -log_probs.gather(1, class_labels_flat.unsqueeze(1)).squeeze(1)

        # Apply focal weighting
        focal_loss = (focal_weight * nll_loss).mean()

        return focal_loss

class RegressorLoss(nn.Module):
    """Loss for regressor training only"""
    def __init__(self):
        super().__init__()

    def forward(self, angle_preds, targets):
        """
        Args:
            angle_preds: [B, 2672] or [B, 2, 2672]
            targets: [B, 2672] - ground truth angles with flags
        """
        # Only compute loss on valid positions (exclude flags)
        valid_mask = (targets != -10) & (targets != -20)

        if not valid_mask.any():
            return torch.tensor(0.0, device=targets.device)

        # Regression loss only for valid positions
        if angle_preds.dim() == 3:  # Combined mode [B, 2, 2672]
            angle_preds = angle_preds.reshape(angle_preds.size(0), -1)

        reg_loss = F.mse_loss(angle_preds[valid_mask], targets[valid_mask])
        return 100 * reg_loss

# ==================== TRAINING FUNCTIONS ====================




def train_xgboost_3class(train_dataset, val_dataset):
    """Train XGBoost for 3-class classification"""
    print("Preparing 3-class data for XGBoost...")
    X_train, y_train = prepare_3class_data(train_dataset)
    X_val, y_val = prepare_3class_data(val_dataset)

    print(f"\nTraining samples: {len(X_train)}")
    print(f"  Valid (0): {np.sum(y_train==0)} ({np.sum(y_train==0)/len(y_train)*100:.1f}%)")
    print(f"  No Return (1): {np.sum(y_train==1)} ({np.sum(y_train==1)/len(y_train)*100:.1f}%)")
    print(f"  Invalid (2): {np.sum(y_train==2)} ({np.sum(y_train==2)/len(y_train)*100:.1f}%)")

    # XGBoost multiclass
    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=8,
        learning_rate=0.1,
        objective='multi:softmax',
        num_class=3,
        eval_metric='mlogloss',
        random_state=42,
        tree_method='hist',
        device='cuda' if torch.cuda.is_available() else 'cpu',
        # Class weights for imbalance
        scale_pos_weight=None  # Not used for multiclass
    )

    print("\nTraining XGBoost...")
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=50
    )

    # Evaluate
    y_pred = model.predict(X_val)
    print("\n" + "="*70)
    print("XGBoost Classification Report:")
    print("="*70)
    print(classification_report(y_val, y_pred,
                                target_names=['Valid', 'No Return (-10)', 'Invalid (-20)']))

    return model


def train_random_forest_3class(train_dataset, val_dataset):
    """Train Random Forest for 3-class classification"""
    print("Preparing 3-class data for Random Forest...")
    X_train, y_train = prepare_3class_data(train_dataset)
    X_val, y_val = prepare_3class_data(val_dataset)

    print(f"\nTraining samples: {len(X_train)}")

    # Random Forest with class weights
    class_weights = {0: 100.0, 1: 30.0, 2: 1.0}  # Same as your focal loss weights

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=10,
        min_samples_leaf=5,
        class_weight=class_weights,
        random_state=42,
        n_jobs=-1,
        verbose=2
    )

    print("\nTraining Random Forest...")
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_val)
    print("\n" + "="*70)
    print("Random Forest Classification Report:")
    print("="*70)
    print(classification_report(y_val, y_pred,
                                target_names=['Valid', 'No Return (-10)', 'Invalid (-20)']))

    return model

def train_classifier_epoch(model, dataloader, device, optimizer):
    """Train classifier for one epoch"""
    model.train()
    criterion = ClassifierLoss()

    total_loss = 0.0
    valid_batches = 0

    for intensities, ground_truth in dataloader:
        intensities, ground_truth = intensities.to(device), ground_truth.to(device)

        optimizer.zero_grad()
        class_logits = model(intensities)
        loss = criterion(class_logits, ground_truth)

        if loss.item() > 0 and torch.isfinite(loss):
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
            valid_batches += 1

    return total_loss / max(valid_batches, 1)


def train_regressor_epoch(model, dataloader, device, optimizer):
    """Train regressor for one epoch"""
    model.train()
    criterion = RegressorLoss()

    total_loss = 0.0
    valid_batches = 0

    for intensities, ground_truth in dataloader:
        intensities, ground_truth = intensities.to(device), ground_truth.to(device)

        optimizer.zero_grad()
        angle_preds = model(intensities)
        loss = criterion(angle_preds, ground_truth)

        if loss.item() > 0 and torch.isfinite(loss):
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
            valid_batches += 1

    return total_loss / max(valid_batches, 1)


def validate_classifier(model, dataloader, device):
    """Validate classifier"""
    model.to(device).eval()
    criterion = ClassifierLoss()

    total_loss = 0.0
    valid_batches = 0

    with torch.no_grad():
        for intensities, ground_truth in dataloader:
            intensities, ground_truth = intensities.to(device), ground_truth.to(device)

            class_logits = model(intensities)
            loss = criterion(class_logits, ground_truth)

            if loss.item() > 0 and torch.isfinite(loss):
                total_loss += loss.item()
                valid_batches += 1

    return total_loss / max(valid_batches, 1)


def validate_regressor(model, dataloader, device):
    """Validate regressor"""
    model.eval()
    criterion = RegressorLoss()

    total_loss = 0.0
    valid_batches = 0

    with torch.no_grad():
        for intensities, ground_truth in dataloader:
            intensities, ground_truth = intensities.to(device), ground_truth.to(device)

            angle_preds = model(intensities)
            loss = criterion(angle_preds, ground_truth)

            if loss.item() > 0 and torch.isfinite(loss):
                total_loss += loss.item()
                valid_batches += 1

    return total_loss / max(valid_batches, 1)

# ==================== MAIN TRAINING PIPELINE ====================

def train_classifier_model(model, train_loader, val_loader, num_epochs=50, model_name="classifier"):
    """Train classifier model independently"""
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-6, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=1e-4, total_steps=num_epochs,
        pct_start=0.3, anneal_strategy='cos'
    )

    print(f"\n{'='*60}")
    print(f"TRAINING CLASSIFIER CNN")
    print(f"{'='*60}")
    print(f"Device: {device}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    best_val_loss = float('inf')
    patience_counter = 0
    patience = 50

    for epoch in range(num_epochs):
        train_loss = train_classifier_epoch(model, train_loader, device, optimizer)
        val_loss = validate_classifier(model, val_loader, device)
        scheduler.step()

        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), f'best_{model_name}_classifier.pth')
        else:
            patience_counter += 1

        if epoch % 10 == 0 or epoch < 5:
            print(f"Epoch {epoch+1:3d}/{num_epochs} | Train: {train_loss:.6f} | "
                  f"Val: {val_loss:.6f} | Best: {best_val_loss:.6f}")

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

    model.load_state_dict(torch.load(f'best_{model_name}_classifier.pth', map_location=device))
    print(f"Classifier training complete. Best val loss: {best_val_loss:.6f}\n")
    return model


def train_regressor_model(model, train_loader, val_loader, num_epochs=50, model_name="regressor"):
    """Train regressor model independently"""
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=1e-2, total_steps=num_epochs,
        pct_start=0.3, anneal_strategy='cos'
    )

    print(f"\n{'='*60}")
    print(f"TRAINING REGRESSOR CNN")
    print(f"{'='*60}")
    print(f"Device: {device}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    best_val_loss = float('inf')
    patience_counter = 0
    patience = 30

    for epoch in range(num_epochs):
        train_loss = train_regressor_epoch(model, train_loader, device, optimizer)
        val_loss = validate_regressor(model, val_loader, device)
        scheduler.step()

        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), f'best_{model_name}_regressor.pth')
        else:
            patience_counter += 1

        if epoch % 10 == 0 or epoch < 5:
            print(f"Epoch {epoch+1:3d}/{num_epochs} | Train: {train_loss:.6f} | "
                  f"Val: {val_loss:.6f} | Best: {best_val_loss:.6f}")

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

    model.load_state_dict(torch.load(f'best_{model_name}_regressor.pth', map_location=device))
    print(f"Regressor training complete. Best val loss: {best_val_loss:.6f}\n")
    return model

# ==================== DATA HANDLING ====================

class BathymetryDataset(Dataset):
    def __init__(self, csv_file, prediction_type='phi'):
        self.data = pd.read_csv(csv_file)
        self.prediction_type = prediction_type

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        # print(self.data.columns)
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

    # def __getitem__(self, idx):
    #     row = self.data.iloc[idx]

    #     # Extract intensities (always the same)
    #     intensities = torch.tensor(row.iloc[1:669].values, dtype=torch.float32)
    #     intensities_processed, _ = create_valid_mask(intensities)

    #     # Add range information as additional feature
    #     ranges = torch.linspace(0.5, 40.0, 668)  # Range from min_range to max_range

    #     # Stack as 2-channel input [2, 668]
    #     features = torch.stack([intensities_processed, ranges], dim=0)

    #     # Extract target data based on prediction type
    #     if self.prediction_type == 'phi':
    #         target_data = torch.tensor(row.iloc[3341:6013].values, dtype=torch.float32)
    #     elif self.prediction_type == 'tangent':
    #         target_data = torch.tensor(row.iloc[669:3341].values, dtype=torch.float32)
    #     elif self.prediction_type == 'combined':
    #         tangents = torch.tensor(row.iloc[669:3341].values, dtype=torch.float32)
    #         phis = torch.tensor(row.iloc[3341:6013].values, dtype=torch.float32)
    #         # Combine tangents and phis for combined training
    #         tangents_processed, _ = create_valid_mask(tangents)
    #         phis_processed, _ = create_valid_mask(phis)
    #         target_data = torch.stack([
    #             tangents_processed.view(4, 668),
    #             phis_processed.view(4, 668)
    #         ], dim=0).view(8, 668).transpose(0, 1).contiguous().view(-1)
    #     else:
    #         raise ValueError(f"Unknown prediction_type: {self.prediction_type}")

    #     # Process target data for non-combined types
    #     if self.prediction_type != 'combined':
    #         target_processed, _ = create_valid_mask(target_data)
    #         ground_truth = target_processed.view(4, 668).contiguous().view(-1)
    #     else:
    #         ground_truth = target_data

    #     return features, ground_truth

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

# ==================== INFERENCE & TESTING ====================

def predict_with_model(model, intensities, device):
    """Make predictions with trained model."""
    model.to(device).eval()
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

# def test_specific_rows_modular(combined_model, dataset, num_test_rows, output_dir, prediction_type):
#     """Test combined model (classifier + regressor) on specific rows"""
#     ensure_dir(output_dir)
#     device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
#     combined_model.classifier.to(device).eval()
#     combined_model.regressor.to(device).eval()

#     test_indices = list(range(min(num_test_rows, len(dataset))))
#     results = {}

#     print(f"\nTesting {len(test_indices)} samples for {prediction_type}...")

#     with torch.no_grad():
#         for i in test_indices:
#             intensities, ground_truth_raw = dataset[i]
#             actual_idx = dataset.indices[i] if hasattr(dataset, 'indices') else i

#             # Make prediction with combined model
#             intensities_batch = intensities.unsqueeze(0).to(device)
#             class_logits, angle_preds = combined_model(intensities_batch)

#             # Get predicted classes
#             class_preds = torch.argmax(class_logits, dim=1).squeeze(0).cpu()  # [2672]
#             angle_preds = angle_preds.squeeze(0).cpu()  # [2672]

#             # Combine classification and regression outputs
#             prediction = angle_preds.clone()
#             prediction[class_preds == 1] = -10.0  # Flag -10 positions
#             prediction[class_preds == 2] = -20.0  # Flag -20 positions

#             # Create ground truth classes
#             gt_classes = torch.zeros_like(ground_truth_raw, dtype=torch.long)
#             gt_classes[ground_truth_raw == -10.0] = 1
#             gt_classes[ground_truth_raw == -20.0] = 2

#             # Reshape classes for each beam (4 beams)
#             pred_classes_reshaped = torch.zeros(4, 668)
#             gt_classes_reshaped = torch.zeros(4, 668)
#             for j in range(4):
#                 pred_classes_reshaped[j, :] = class_preds[j::4]
#                 gt_classes_reshaped[j, :] = gt_classes[j::4]

#             # Evaluate
#             row_metrics = evaluate_single_row(prediction, ground_truth_raw)

#             # Print summary
#             print(f"Sample {i:2d} (CSV row {actual_idx:4d}) | "
#                   f"Angle RMSE: {row_metrics['angle_rmse']:6.3f} | "
#                   f"Flag Acc: {row_metrics['flag_acc']:5.1%} | "
#                   f"Valid: {row_metrics['clean_points']:3d}/{row_metrics['valid_points']:3d}")

#             # Reshape for visualization
#             pred_tangents, pred_phis = reshape_predictions(prediction, prediction_type=prediction_type)
#             gt_processed, _ = create_valid_mask(ground_truth_raw)
#             gt_tangents, gt_phis = reshape_predictions(gt_processed, prediction_type=prediction_type)

#             # Store results
#             results[i] = {
#                 'intensities': intensities.numpy(),
#                 'pred_tangents': pred_tangents.numpy(),
#                 'pred_phis': pred_phis.numpy(),
#                 'gt_tangents': gt_tangents.numpy(),
#                 'gt_phis': gt_phis.numpy(),
#                 'classes_phi': class_logits.cpu().numpy(),
#                 'pred_classes_phi': pred_classes_reshaped.numpy() if prediction_type == 'phi' else torch.zeros(4, 668).numpy(),
#                 'gt_classes_phi': gt_classes_reshaped.numpy() if prediction_type == 'phi' else torch.zeros(4, 668).numpy(),
#                 'pred_classes_tangent': pred_classes_reshaped.numpy() if prediction_type == 'tangent' else torch.zeros(4, 668).numpy(),
#                 'gt_classes_tangent': gt_classes_reshaped.numpy() if prediction_type == 'tangent' else torch.zeros(4, 668).numpy(),
#                 'metrics': row_metrics,
#                 'prediction_type': prediction_type
#             }

#             # Save CSV
#             import pandas as pd
#             data = {'pixel_idx': range(668), 'intensities': intensities.numpy()}
#             for j in range(4):
#                 if prediction_type in ['phi', 'combined']:
#                     data[f'pred_phi_{j}'] = pred_phis.numpy()[j]
#                     data[f'gt_phi_{j}'] = gt_phis.numpy()[j]
#                 if prediction_type in ['tangent', 'combined']:
#                     data[f'pred_tangent_{j}'] = pred_tangents.numpy()[j]
#                     data[f'gt_tangent_{j}'] = gt_tangents.numpy()[j]

#             pd.DataFrame(data).to_csv(f'{output_dir}/row_{actual_idx}_predictions.csv', index=False)

#     print(f"Results saved to {output_dir}/")
#     return results

def test_specific_rows_modular(combined_model, test_dataset, num_test_rows, output_dir, prediction_type):
    """Test with XGBoost classifier + PyTorch regressor"""

    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

    # Check if classifier is XGBoost or PyTorch
    is_xgboost = hasattr(combined_model.classifier, 'predict')

    if not is_xgboost:
        combined_model.classifier.to(device)
        combined_model.classifier.eval()

    combined_model.regressor.to(device)
    combined_model.regressor.eval()

    results = {}

    with torch.no_grad():
        for i in range(min(num_test_rows, len(test_dataset))):
            intensities, ground_truth = test_dataset[i]
            intensities_np = intensities.numpy()  # [668]
            intensities_tensor = intensities.unsqueeze(0).to(device)  # [1, 668]

            # Classification
            if is_xgboost:
                # Create features for each of 2672 predictions (same as training)
                X_features = []
                for j in range(2672):
                    intensity_idx = j // 4

                    # 5-point window (same as training)
                    start = max(0, intensity_idx - 2)
                    end = min(668, intensity_idx + 3)
                    features = intensities_np[start:end]

                    # Pad to 5 features
                    if len(features) < 5:
                        features = np.pad(features, (0, 5 - len(features)), mode='edge')

                    X_features.append(features)

                X_features = np.array(X_features)  # [2672, 5]
                class_preds = combined_model.classifier.predict(X_features)  # [2672]
                class_preds = torch.tensor(class_preds, device=device)
            else:
                # PyTorch CNN
                class_logits = combined_model.classifier(intensities_tensor)
                class_preds = torch.argmax(class_logits, dim=1).squeeze(0)

            # Regression
            angle_pred = combined_model.regressor(intensities_tensor).squeeze(0)  # [2672]

            # Combine: use regression for valid (class 0), flags for others
            final_pred = angle_pred.clone()
            final_pred[class_preds == 1] = -10.0
            final_pred[class_preds == 2] = -20.0

            results[i] = {
                f'pred_{prediction_type}s': final_pred.cpu().numpy().reshape(4, 668),
                f'true_{prediction_type}s': ground_truth.numpy().reshape(4, 668)
            }

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

# def main():
#     """Main training pipeline with separate classifier and regressor CNNs"""

#     MODELS_TO_TRAIN = ['phi']
#     # Setup
#     csv_file = '/Users/farhang/Downloads/fls_all_with_phi.csv'
#     # csv_file = '/Users/farhang/Downloads/fls_all_with_phis_long.csv'  # Updated CSV with extended phis

#     print(f"\n{'='*60}")
#     print("MODULAR BATHYMETRY CNN TRAINING")
#     print(f"{'='*60}\n")

#     # Handle data splits
#     splits_dir = "./data_splits"
#     if os.path.exists(f"{splits_dir}/train_data.csv"):
#         print("Using existing CSV splits...")
#         train_csv = f"{splits_dir}/train_data.csv"
#         val_csv = f"{splits_dir}/val_data.csv"
#         test_csv = f"{splits_dir}/test_data.csv"
#     else:
#         print("Creating new splits and saving to CSV...")
#         # Assuming you have save_splits_to_csv function
#         train_csv, val_csv, test_csv = save_splits_to_csv(csv_file, splits_dir)

#     # Create datasets for each split and prediction type
#     print("\nLoading datasets...")
#     datasets = {}
#     for pred_type in MODELS_TO_TRAIN:
#         datasets[pred_type] = {
#             'train': BathymetryDataset(train_csv, prediction_type=pred_type),
#             'val': BathymetryDataset(val_csv, prediction_type=pred_type),
#             'test': BathymetryDataset(test_csv, prediction_type=pred_type)
#         }

#     print(f"Phi splits - Train: {len(datasets['phi']['train'])}, "
#           f"Val: {len(datasets['phi']['val'])}, Test: {len(datasets['phi']['test'])}")
#     # print(f"Tangent splits - Train: {len(datasets['tangent']['train'])}, "
#     #       f"Val: {len(datasets['tangent']['val'])}, Test: {len(datasets['tangent']['test'])}")

#     # Create data loaders
#     print("\nCreating data loaders...")
#     loaders = {}
#     for pred_type in MODELS_TO_TRAIN:
#         loaders[pred_type] = {
#             'train': DataLoader(datasets[pred_type]['train'], batch_size=8, shuffle=True),
#             'val': DataLoader(datasets[pred_type]['val'], batch_size=8, shuffle=False),
#             'test': DataLoader(datasets[pred_type]['test'], batch_size=8, shuffle=False)
#         }

#         # train_sampler = create_balanced_sampler(datasets[pred_type]['train'])

#         # loaders[pred_type] = {
#         #     'train': DataLoader(datasets[pred_type]['train'], batch_size=8, sampler=train_sampler),  # Remove shuffle when using sampler
#         #     'val': DataLoader(datasets[pred_type]['val'], batch_size=8, shuffle=False),
#         #     'test': DataLoader(datasets[pred_type]['test'], batch_size=8, shuffle=False)
#         # }


#     # ==================== ANALYZE CLASS SEPARABILITY ====================

#     # print("\n" + "="*60)
#     # print("ANALYZING CLASS SEPARABILITY")
#     # print("="*60)

#     # visualize_class_separability(datasets['phi']['train'], num_samples_per_class=500)


#     # ==================== PHASE 1: TRAIN CLASSIFIERS ====================
#     print(f"\n{'='*60}")
#     print("PHASE 1: TRAINING CLASSIFIERS")
#     print(f"{'='*60}")

#     classifiers = {}
#     for pred_type in MODELS_TO_TRAIN:
#         print(f"\nTraining {pred_type.upper()} classifier...")
#         # classifier = ClassifierCNN(dropout_rate=0.1)
#         classifier = SimpleClassifier()
#         # Train or load existing classifier
#         classifier_path = f'best_{pred_type}_classifier.pth'
#         if os.path.exists(classifier_path):
#             device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
#             classifier.load_state_dict(torch.load(classifier_path, map_location=device))
#             print(f"Loaded existing {pred_type.upper()} classifier from {classifier_path}")
#         else:
#             classifier = train_classifier_model(
#                 classifier,
#                 loaders[pred_type]['train'],
#                 loaders[pred_type]['val'],
#                 num_epochs=120,
#                 model_name=pred_type
#             )

#         classifiers[pred_type] = classifier

#     # classifiers = {}
#     # for pred_type in MODELS_TO_TRAIN:
#     #     print(f"\nTraining {pred_type.upper()} classifier...")

#     #     # Check if XGBoost model already exists
#     #     xgb_path = f'xgboost_{pred_type}_classifier.pkl'
#     #     if os.path.exists(xgb_path):
#     #         print(f"Loading existing XGBoost classifier from {xgb_path}")
#     #         classifier = joblib.load(xgb_path)
#     #     else:
#     #         # Train new XGBoost classifier
#     #         classifier = train_xgboost_3class(
#     #             datasets[pred_type]['train'],
#     #             datasets[pred_type]['val']
#     #         )
#     #         joblib.dump(classifier, xgb_path)
#     #         print(f"Saved XGBoost classifier to {xgb_path}")

#     #     classifiers[pred_type] = classifier

#     # ==================== PHASE 2: TRAIN REGRESSORS ====================
#     print(f"\n{'='*60}")
#     print("PHASE 2: TRAINING REGRESSORS")
#     print(f"{'='*60}")

#     regressors = {}
#     for pred_type in MODELS_TO_TRAIN:
#         print(f"\nTraining {pred_type.upper()} regressor...")
#         regressor = RegressorCNN(prediction_type=pred_type, dropout_rate=0.1)

#         # Train or load existing regressor
#         regressor_path = f'best_{pred_type}_regressor.pth'
#         if os.path.exists(regressor_path):
#             device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
#             regressor.load_state_dict(torch.load(regressor_path, map_location=device))
#             print(f"Loaded existing {pred_type.upper()} regressor from {regressor_path}")
#         else:
#             regressor = train_regressor_model(
#                 regressor,
#                 loaders[pred_type]['train'],
#                 loaders[pred_type]['val'],
#                 num_epochs=10,
#                 model_name=pred_type
#             )

#         regressors[pred_type] = regressor

#     # ==================== PHASE 3: CREATE COMBINED MODELS ====================
#     print(f"\n{'='*60}")
#     print("PHASE 3: CREATING COMBINED MODELS")
#     print(f"{'='*60}\n")

#     combined_models = {}
#     for pred_type in MODELS_TO_TRAIN:
#         combined_models[pred_type] = CombinedBathymetryModel(
#             classifiers[pred_type],
#             regressors[pred_type]
#         )
#         print(f"Created combined {pred_type.upper()} model")

#     # ==================== EVALUATION ====================
#     print(f"\n{'='*60}")
#     print("MODEL EVALUATION")
#     print(f"{'='*60}\n")

#     device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

#     for pred_type in MODELS_TO_TRAIN:
#         classifiers[pred_type].to(device)
#         regressors[pred_type].to(device)

#     for pred_type in MODELS_TO_TRAIN:
#         # Evaluate classifier
#         classifier_val_loss = validate_classifier(
#             classifiers[pred_type],
#             loaders[pred_type]['test'],
#             device
#         )
#         print(f"{pred_type.title()} Classifier Test Loss: {classifier_val_loss:.6f}") # For xgboost this is not needed

#         # Evaluate regressor
#         regressor_val_loss = validate_regressor(
#             regressors[pred_type],
#             loaders[pred_type]['test'],
#             device
#         )
#         print(f"{pred_type.title()} Regressor Test Loss: {regressor_val_loss:.6f}")

#     # ==================== DETAILED TESTING ON SPECIFIC ROWS ====================
#     print(f"\n{'='*60}")
#     print("DETAILED TESTING ON TEST SET")
#     print(f"{'='*60}")

#     test_results = {}
#     for pred_type in MODELS_TO_TRAIN:
#         print(f"\nTesting {pred_type.upper()} network on TEST set...")

#         test_results[pred_type] = test_specific_rows_modular(
#             combined_models[pred_type],
#             datasets[pred_type]['test'],
#             num_test_rows=len(datasets[pred_type]['test']),
#             output_dir=f'test_outputs_{pred_type}',
#             prediction_type=pred_type
#         )

#     # ==================== INFERENCE ON TRAIN SET ====================
#     print(f"\n{'='*60}")
#     print("INFERENCE ON TRAIN SET")
#     print(f"{'='*60}")

#     train_results = {}
#     for pred_type in MODELS_TO_TRAIN:
#         print(f"\nRunning inference on TRAIN set for {pred_type.upper()}...")

#         train_results[pred_type] = test_specific_rows_modular(
#             combined_models[pred_type],
#             datasets[pred_type]['train'],
#             num_test_rows=len(datasets[pred_type]['train']),
#             output_dir=f'train_outputs_{pred_type}',
#             prediction_type=pred_type
#         )

#     # ==================== SAVE MODELS ====================
#     print(f"\n{'='*60}")
#     print("SAVING MODELS")
#     print(f"{'='*60}\n")

#     ensure_dir('./models')

#     for pred_type in MODELS_TO_TRAIN:
#         # Save classifier
#         classifier_path = f'./models/classifier_{pred_type}.pth'
#         torch.save(classifiers[pred_type].state_dict(), classifier_path)
#         print(f"Saved {pred_type.title()} classifier: {classifier_path}")

#         # Save regressor
#         regressor_path = f'./models/regressor_{pred_type}.pth'
#         torch.save(regressors[pred_type].state_dict(), regressor_path)
#         print(f"Saved {pred_type.title()} regressor: {regressor_path}")

#     # for pred_type in MODELS_TO_TRAIN:
#     #     # Save classifier (check if PyTorch or XGBoost)
#     #     if hasattr(classifiers[pred_type], 'state_dict'):
#     #         # PyTorch model
#     #         classifier_path = f'./models/classifier_{pred_type}.pth'
#     #         torch.save(classifiers[pred_type].state_dict(), classifier_path)
#     #         print(f"Saved {pred_type.title()} PyTorch classifier: {classifier_path}")
#     #     else:
#     #         # XGBoost model (already saved earlier with joblib)
#     #         print(f"XGBoost {pred_type.title()} classifier already saved as xgboost_{pred_type}_classifier.pkl")

#     #     # Save regressor (always PyTorch)
#     #     regressor_path = f'./models/regressor_{pred_type}.pth'
#     #     torch.save(regressors[pred_type].state_dict(), regressor_path)
#     #     print(f"Saved {pred_type.title()} regressor: {regressor_path}")


#     # ==================== SAVE PREDICTIONS TO CSV ====================
#     print(f"\n{'='*60}")
#     print("SAVING PREDICTIONS")
#     print(f"{'='*60}\n")

#     # Create test data with indices
#     test_with_indices = save_splits_with_indices_to_csv(csv_file, splits_dir, split_type='test')
#     train_with_indices = save_splits_with_indices_to_csv(csv_file, splits_dir, split_type='train')

#     train_predictions_path = "./data_splits/train_predictions_with_indices.csv"
#     test_predictions_path = "./data_splits/test_predictions_with_indices.csv"
#     # # Save test predictions
#     # test_predictions_path = "./data_splits/test_predictions_with_indices.csv"
#     # save_predictions_with_indices_to_csv(
#     #     results['phi'],
#     #     results['phi'],
#     #     test_with_indices,
#     #     test_predictions_path
#     # )
#     # print(f"Test predictions saved to {test_predictions_path}")

#     tangent_results_dummy_test = {}
#     for i, result in test_results['phi'].items():
#         tangent_results_dummy_test[i] = {
#             'pred_tangents': result['pred_phis'],
#             'true_tangents': result['true_phis']
#         }

#     # Save test predictions
#     save_predictions_with_indices_to_csv(
#         tangent_results_dummy_test,
#         test_results['phi'],
#         test_with_indices,
#         test_predictions_path
#     )

#     print(f"Test predictions saved to {test_predictions_path}")
#     # Create dummy tangents for train
#     tangent_results_dummy_train = {}
#     for i, result in train_results['phi'].items():
#         tangent_results_dummy_train[i] = {
#             'pred_tangents': result['pred_phis'],
#             'true_tangents': result['true_phis']
#         }

#     # Save train predictions
#     save_predictions_with_indices_to_csv(
#         tangent_results_dummy_train,
#         train_results['phi'],
#         train_with_indices,
#         train_predictions_path
#     )

#     print(f"Train predictions saved to {train_predictions_path}")

#     print(f"\n{'='*60}")
#     print("TRAINING COMPLETE!")
#     print(f"{'='*60}\n")

#     # Generate comparison plots
#     # save_test_indices_vs_original_pcl_plots(
#     #     test_csv_path=predictions_with_indices_path,
#     #     original_csv_path="/Users/farhang/Downloads/fls_all_with_phi.csv",
#     #     output_dir="./test_indices_vs_original"
#     # )

# if __name__ == "__main__":
#     main()

def find_optimal_threshold(dataset, num_samples=200):
    """Find optimal intensity threshold by testing different values"""

    print("\n" + "="*60)
    print("FINDING OPTIMAL THRESHOLD")
    print("="*60)

    # Collect intensity vs validity data
    intensity_valid_pairs = []

    for i in range(min(num_samples, len(dataset))):
        intensities, targets = dataset[i]
        targets_reshaped = targets.view(4, 668)

        for pixel_idx in range(668):
            intensity = intensities[pixel_idx].item()

            # Check if ANY beam at this pixel is valid
            pixel_targets = targets_reshaped[:, pixel_idx]
            has_valid = ((pixel_targets != -10) & (pixel_targets != -20)).any().item()
            has_neg10 = (pixel_targets == -10).any().item()
            has_neg20 = (pixel_targets == -20).any().item()

            # Store: (intensity, has_valid, has_-10, has_-20)
            intensity_valid_pairs.append((intensity, has_valid, has_neg10, has_neg20))

    # Test different thresholds
    thresholds = [5, 10, 15, 20, 25, 30, 40, 50, 75, 100, 103]

    best_threshold = None
    best_f1 = 0

    print(f"\nTesting {len(thresholds)} threshold values...")
    print(f"{'Threshold':<12} {'Valid F1':<12} {'(-20) F1':<12} {'Avg F1':<12}")
    print("-" * 60)

    for thresh in thresholds:
        # Calculate metrics for this threshold
        valid_tp = valid_fp = valid_fn = 0
        neg20_tp = neg20_fp = neg20_fn = 0

        for intensity, has_valid, has_neg10, has_neg20 in intensity_valid_pairs:
            predicted_valid = intensity >= thresh

            # Valid metrics
            if predicted_valid and has_valid:
                valid_tp += 1
            elif predicted_valid and not has_valid:
                valid_fp += 1
            elif not predicted_valid and has_valid:
                valid_fn += 1

            # -20 metrics
            predicted_neg20 = intensity < thresh
            if predicted_neg20 and has_neg20:
                neg20_tp += 1
            elif predicted_neg20 and not has_neg20:
                neg20_fp += 1
            elif not predicted_neg20 and has_neg20:
                neg20_fn += 1

        # Calculate F1 scores
        valid_precision = valid_tp / (valid_tp + valid_fp) if (valid_tp + valid_fp) > 0 else 0
        valid_recall = valid_tp / (valid_tp + valid_fn) if (valid_tp + valid_fn) > 0 else 0
        valid_f1 = 2 * valid_precision * valid_recall / (valid_precision + valid_recall) if (valid_precision + valid_recall) > 0 else 0

        neg20_precision = neg20_tp / (neg20_tp + neg20_fp) if (neg20_tp + neg20_fp) > 0 else 0
        neg20_recall = neg20_tp / (neg20_tp + neg20_fn) if (neg20_tp + neg20_fn) > 0 else 0
        neg20_f1 = 2 * neg20_precision * neg20_recall / (neg20_precision + neg20_recall) if (neg20_precision + neg20_recall) > 0 else 0

        avg_f1 = (valid_f1 + neg20_f1) / 2

        print(f"{thresh:<12.1f} {valid_f1:<12.3f} {neg20_f1:<12.3f} {avg_f1:<12.3f}")

        if avg_f1 > best_f1:
            best_f1 = avg_f1
            best_threshold = thresh

    print("-" * 60)
    print(f"\n✓ Optimal threshold: {best_threshold} (Avg F1: {best_f1:.3f})")

    return best_threshold

def main():
    csv_file = '/Users/farhang/Downloads/fls_all_with_phi.csv'

    print("\n" + "="*60)
    print("FINAL APPROACH: Regressor + Intensity Threshold")
    print("="*60)

    # Create splits
    splits_dir = "./data_splits"
    if not os.path.exists(f"{splits_dir}/train_data.csv"):
        train_csv, val_csv, test_csv = save_splits_to_csv(csv_file, splits_dir)
    else:
        train_csv = f"{splits_dir}/train_data.csv"
        val_csv = f"{splits_dir}/val_data.csv"
        test_csv = f"{splits_dir}/test_data.csv"

    datasets = {
        'train': BathymetryDataset(train_csv, prediction_type='phi'),
        'val': BathymetryDataset(val_csv, prediction_type='phi'),
        'test': BathymetryDataset(test_csv, prediction_type='phi')
    }

    INTENSITY_THRESHOLD = find_optimal_threshold(datasets['val'], num_samples=200)

    loaders = {
        'train': DataLoader(datasets['train'], batch_size=8, shuffle=True),
        'val': DataLoader(datasets['val'], batch_size=8, shuffle=False),
        'test': DataLoader(datasets['test'], batch_size=8, shuffle=False)
    }

    # Load or train regressor
    regressor = RegressorCNN(prediction_type='phi', dropout_rate=0.1)

    regressor_path = 'best_phi_regressor.pth'
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

    if os.path.exists(regressor_path):
        regressor.load_state_dict(torch.load(regressor_path, map_location=device))
        print(f"Loaded existing regressor from {regressor_path}")
    else:
        print("\nTraining regressor...")
        regressor = train_regressor_model(
            regressor,
            loaders['train'],
            loaders['val'],
            num_epochs=50,
            model_name='phi'
        )



    print(f"\nUsing intensity threshold: {INTENSITY_THRESHOLD}")

    # Test
    print("\n" + "="*60)
    print("TESTING WITH THRESHOLD")
    print("="*60)

    regressor.to(device).eval()

    test_results = {}
    train_results = {}

    # Test set inference
    with torch.no_grad():
        for i in range(len(datasets['test'])):
            intensities, ground_truth = datasets['test'][i]

            # Predict angles
            angle_pred = regressor(intensities.unsqueeze(0).to(device)).squeeze(0).cpu()

            # Apply intensity-based flagging
            final_pred = angle_pred.clone()

            for pixel_idx in range(668):
                intensity = intensities[pixel_idx].item()

                # If intensity below threshold, mark all 4 beams as -20
                if intensity < INTENSITY_THRESHOLD:
                    final_pred[pixel_idx*4:(pixel_idx+1)*4] = -20.0

            # Evaluate
            metrics = evaluate_single_row(final_pred, ground_truth)

            if i < 20:
                print(f"Sample {i:2d} | "
                      f"Angle RMSE: {metrics['angle_rmse']:6.3f} | "
                      f"Flag Acc: {metrics['flag_acc']:5.1%} | "
                      f"Valid: {metrics['clean_points']:3d}/{metrics['valid_points']:3d}")

            test_results[i] = {
                'pred_phis': final_pred.view(4, 668).numpy(),
                'true_phis': ground_truth.view(4, 668).numpy()
            }

    # Train set inference
    print("\n" + "="*60)
    print("RUNNING INFERENCE ON TRAIN SET")
    print("="*60)

    with torch.no_grad():
        for i in range(len(datasets['train'])):
            intensities, ground_truth = datasets['train'][i]

            angle_pred = regressor(intensities.unsqueeze(0).to(device)).squeeze(0).cpu()

            final_pred = angle_pred.clone()

            for pixel_idx in range(668):
                intensity = intensities[pixel_idx].item()
                if intensity < INTENSITY_THRESHOLD:
                    final_pred[pixel_idx*4:(pixel_idx+1)*4] = -20.0

            train_results[i] = {
                'pred_phis': final_pred.view(4, 668).numpy(),
                'true_phis': ground_truth.view(4, 668).numpy()
            }

    # Save predictions to CSV
    print("\n" + "="*60)
    print("SAVING PREDICTIONS TO CSV")
    print("="*60)

    test_with_indices = save_splits_with_indices_to_csv(csv_file, splits_dir, split_type='test')
    train_with_indices = save_splits_with_indices_to_csv(csv_file, splits_dir, split_type='train')

    # Create dummy tangent results (same as phi)
    tangent_results_test = {i: {'pred_tangents': r['pred_phis'], 'true_tangents': r['true_phis']}
                           for i, r in test_results.items()}
    tangent_results_train = {i: {'pred_tangents': r['pred_phis'], 'true_tangents': r['true_phis']}
                            for i, r in train_results.items()}

    # Save
    test_predictions_path = "./data_splits/test_predictions_final.csv"
    save_predictions_with_indices_to_csv(
        tangent_results_test,
        test_results,
        test_with_indices,
        test_predictions_path
    )
    print(f"Test predictions saved to {test_predictions_path}")

    train_predictions_path = "./data_splits/train_predictions_final.csv"
    save_predictions_with_indices_to_csv(
        tangent_results_train,
        train_results,
        train_with_indices,
        train_predictions_path
    )
    print(f"Train predictions saved to {train_predictions_path}")

    print("\n" + "="*60)
    print("COMPLETE!")
    print("="*60)
    print(f"  - Intensity threshold ({INTENSITY_THRESHOLD}) for valid/invalid classification")
    print("="*60)

if __name__ == "__main__":
    main()