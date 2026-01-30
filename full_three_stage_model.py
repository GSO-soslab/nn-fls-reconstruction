# full_three_stage_model.py

# Standard imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset

import pandas as pd
import numpy as np
import os
import csv

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score
import math

# ==============================================================================
# UTILITY FUNCTIONS (Copied from cnn_separate_fls_bathymetry_separate.py)
# ==============================================================================

def create_valid_mask(data, invalid_values=[-20.0]):
    """Convert NaNs to -20.0 and create mask for valid values."""
    data_modified = data.clone()
    data_modified[torch.isnan(data_modified)] = -20.0

    valid_mask = torch.ones_like(data_modified, dtype=torch.bool)
    for val in invalid_values:
        valid_mask &= (data_modified != val)

    return data_modified, valid_mask

def save_splits_to_csv(csv_file, base_dir="./data_splits"):
    """Save actual dataset splits to separate CSV files"""
    os.makedirs(base_dir, exist_ok=True)

    with open(csv_file, 'r') as f:
        lines = f.readlines()

    lines = [line.rstrip('\n\r') for line in lines]

    np.random.seed(42)
    indices = np.arange(len(lines))

    train_indices, temp_indices = train_test_split(indices, test_size=0.3, random_state=42)
    val_indices, test_indices = train_test_split(temp_indices, test_size=1/3, random_state=42)

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

    with open(csv_file, 'r') as f:
        lines = f.readlines()

    lines = [line.rstrip('\n\r') for line in lines]

    np.random.seed(42)
    indices = np.arange(len(lines))

    train_indices, temp_indices = train_test_split(indices, test_size=0.3, random_state=42)
    val_indices, test_indices = train_test_split(temp_indices, test_size=1/3, random_state=42)

    if split_type == 'train':
        selected_indices = train_indices
    elif split_type == 'val':
        selected_indices = val_indices
    else:  # 'test'
        selected_indices = test_indices

    output_file = f"{base_dir}/{split_type}_data_with_indices.csv"
    with open(output_file, 'w') as f:
        for idx in selected_indices:
            f.write(f"{idx},{lines[idx]}\n")

    print(f"Created {split_type}_data_with_indices.csv with {len(selected_indices)} samples")
    return output_file

def save_predictions_with_indices_to_csv(tangent_results, phi_results, test_csv_path, output_csv_path):
    """
    Save predictions with original indices for comparison plotting.
    This creates a CSV with predictions that can be compared against original data.
    Uses pure text processing instead of pandas.
    """
    test_data = []
    with open(test_csv_path, 'r', newline='') as f:
        reader = csv.reader(f)
        for row in reader:
            processed_row = []
            for i, val in enumerate(row):
                if i == 0:
                    processed_row.append(val)
                else:
                    try:
                        processed_row.append(float(val))
                    except ValueError:
                        processed_row.append(val)
            test_data.append(processed_row)

    print(f"Creating predictions CSV with {len(test_data)} test samples...")

    output_data = [row[:] for row in test_data]

    for i in range(len(test_data)):
        if i in tangent_results and i in phi_results:
            pred_tangents = tangent_results[i]['pred_tangents']
            pred_phis = phi_results[i]['pred_phis']

            tangent_start = 670
            tangent_flat = pred_tangents.flatten()

            for j, val in enumerate(tangent_flat):
                col_idx = tangent_start + j
                if col_idx < len(output_data[i]):
                    output_data[i][col_idx] = float(val)

            phi_start = 670 + 2672
            phi_flat = pred_phis.flatten()

            for j, val in enumerate(phi_flat):
                col_idx = phi_start + j
                if col_idx < len(output_data[i]):
                    output_data[i][col_idx] = float(val)

    with open(output_csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        for row in output_data:
            writer.writerow(row)

    print(f"Predictions with indices saved to {output_csv_path}")
    return output_data

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

    exact_matches = (prediction == ground_truth).sum().item()
    results['overall_accuracy'] = exact_matches / len(ground_truth)

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


# ==============================================================================
# DATASET CLASSES
# ==============================================================================

class BathymetryDataset(Dataset):
    def __init__(self, csv_file, prediction_type='phi'):
        # Assuming header=None if your CSV does not have a header
        self.data = pd.read_csv(csv_file, header=None)
        self.prediction_type = prediction_type

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        intensities = torch.tensor(row.iloc[1:669].values, dtype=torch.float32)
        intensities_processed, _ = create_valid_mask(intensities)

        if self.prediction_type == 'phi':
            target_data = torch.tensor(row.iloc[3341:6013].values, dtype=torch.float32)
        elif self.prediction_type == 'tangent':
            target_data = torch.tensor(row.iloc[669:3341].values, dtype=torch.float32)
        elif self.prediction_type == 'combined':
            tangents = torch.tensor(row.iloc[669:3341].values, dtype=torch.float32)
            phis = torch.tensor(row.iloc[3341:6013].values, dtype=torch.float32)
            tangents_processed, _ = create_valid_mask(tangents)
            phis_processed, _ = create_valid_mask(phis)
            target_data = torch.stack([
                tangents_processed.view(4, 668),
                phis_processed.view(4, 668)
            ], dim=0).view(8, 668).transpose(0, 1).contiguous().view(-1)
        else:
            raise ValueError(f"Unknown prediction_type: {self.prediction_type}")

        if self.prediction_type != 'combined':
            target_processed, _ = create_valid_mask(target_data)
            ground_truth = target_processed.view(4, 668).contiguous().view(-1)
        else:
            ground_truth = target_data

        return intensities_processed, ground_truth

class FilteredBathymetryDataset(Dataset):
    """
    Dataset that excludes -20 values
    Only returns samples where target is Valid or -10
    """
    def __init__(self, csv_file, prediction_type='phi'):
        self.data = pd.read_csv(csv_file, header=None)
        self.prediction_type = prediction_type

        print(f"\nFiltering dataset to exclude -20s...")
        valid_indices = []

        for idx in range(len(self.data)):
            row = self.data.iloc[idx]

            if prediction_type == 'phi':
                target_data = torch.tensor(row.iloc[3341:6013].values, dtype=torch.float32)
            else:
                target_data = torch.tensor(row.iloc[669:3341].values, dtype=torch.float32)

            target_processed, _ = create_valid_mask(target_data)

            has_valid_or_neg10 = ((target_processed != -20)).any()

            if has_valid_or_neg10:
                valid_indices.append(idx)

        self.valid_indices = valid_indices
        print(f"Kept {len(valid_indices)} / {len(self.data)} rows ({len(valid_indices)/len(self.data)*100:.1f}%)")

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        actual_idx = self.valid_indices[idx]
        row = self.data.iloc[actual_idx]

        intensities = torch.tensor(row.iloc[1:669].values, dtype=torch.float32)
        intensities_processed, _ = create_valid_mask(intensities)

        if self.prediction_type == 'phi':
            target_data = torch.tensor(row.iloc[3341:6013].values, dtype=torch.float32)
        else:
            target_data = torch.tensor(row.iloc[669:3341].values, dtype=torch.float32)

        target_processed, _ = create_valid_mask(target_data)
        ground_truth = target_processed.view(4, 668).contiguous().view(-1)

        non_neg20_mask = (ground_truth != -20)

        ground_truth_filtered = ground_truth[non_neg20_mask]

        intensities_expanded = intensities_processed.repeat_interleave(4)
        intensities_filtered = intensities_expanded[non_neg20_mask]

        labels = (ground_truth_filtered != -10).float() # 1 = Valid, 0 = -10

        return intensities_filtered, labels, ground_truth_filtered

# Custom collate function for variable-length sequences
def collate_fn(batch):
    intensities_list, labels_list, gt_list = zip(*batch)

    max_len = max(x.size(0) for x in intensities_list)

    intensities_padded = []
    labels_padded = []
    gt_padded = []
    masks = []

    for intens, labels, gt in zip(intensities_list, labels_list, gt_list):
        seq_len = intens.size(0)

        pad_len = max_len - seq_len
        intensities_padded.append(F.pad(intens, (0, pad_len), value=0))
        labels_padded.append(F.pad(labels, (0, pad_len), value=0))
        gt_padded.append(F.pad(gt, (0, pad_len), value=-20.0)) # Pad ground truth with -20

        mask = torch.cat([torch.ones(seq_len), torch.zeros(pad_len)])
        masks.append(mask)

    return (torch.stack(intensities_padded),
            torch.stack(labels_padded),
            torch.stack(gt_padded),
            torch.stack(masks).bool())


# ==============================================================================
# MODEL ARCHITECTURES (Copied/Modified)
# ==============================================================================


class RegressorCNN(nn.Module):
    """Independent CNN for regression only"""
    def __init__(self, prediction_type='phi', dropout_rate=0.1):
        super().__init__()
        self.prediction_type = prediction_type

        self.enc1 = nn.Sequential(
            nn.Conv1d(2, 64, 5, padding=2),
            nn.InstanceNorm1d(64),
            nn.LeakyReLU()
        )
        self.enc2 = nn.Sequential(
            nn.Conv1d(64, 128, 5, stride=2, padding=2),
            nn.InstanceNorm1d(128),
            nn.LeakyReLU()
        )
        self.enc3 = nn.Sequential(
            nn.Conv1d(128, 256, 5, stride=2, padding=2),
            nn.InstanceNorm1d(256),
            nn.LeakyReLU()
        )

        # Dilated bottleneck for larger receptive field
        self.bottleneck = nn.Sequential(
            nn.Conv1d(256, 256, 7, padding=3),
            nn.InstanceNorm1d(256),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Conv1d(256, 256, 7, padding=3),
            nn.InstanceNorm1d(256),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Conv1d(256, 256, 7, padding=3),
            nn.InstanceNorm1d(256),
        )

        # Bottleneck - increased residual blocks for better curvature learning
        self.residual_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(256, 256, 7, padding=3),
                nn.InstanceNorm1d(256),
                nn.LeakyReLU(inplace=True),
                nn.Dropout(dropout_rate),
                nn.Conv1d(256, 256, 7, padding=3),
                nn.InstanceNorm1d(256),
            ) for _ in range(8)
        ])

        self.dec1 = nn.Sequential(
            nn.ConvTranspose1d(256, 128, 5, stride=2, padding=2, output_padding=1),
            nn.InstanceNorm1d(128),
            nn.LeakyReLU()
        )
        self.dec2 = nn.Sequential(
            nn.ConvTranspose1d(256, 64, 5, stride=2, padding=2, output_padding=1),
            nn.InstanceNorm1d(64),
            nn.LeakyReLU()
        )
        self.final_upsample = nn.Sequential(
            nn.ConvTranspose1d(128, 32, 5, stride=4, padding=1, output_padding=1),
            nn.LeakyReLU(),
        )

        if prediction_type == 'combined':
            self.regressor = nn.Conv1d(32, 2, 3, padding=1)
        else:
            self.regressor = nn.Conv1d(32, 1, 3, padding=1)

    def forward(self, x):
        """
        Args:
            x: [B, 2, 668] intensity+range input
        Returns:
            angle_pred: [B, 2672] or [B, 2, 2672]
        """
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)

        b = e3
        for block in self.residual_blocks:
            b = block(b) + b  # Add residual connection

        # Apply dilated bottleneck
        b = self.bottleneck(b) + b

        d1 = self.dec1(b)
        d1 = torch.cat([d1, e2], dim=1)

        d2 = self.dec2(d1)
        d2 = torch.cat([d2, e1], dim=1)

        features = self.final_upsample(d2)
        angle_pred = self.regressor(features)

        return angle_pred.squeeze(1) if angle_pred.size(1) == 1 else angle_pred

# class Neg20Detector(nn.Module):
#     """
#     MLP to detect -20s based on intensity alone.
#     Outputs logits for binary classification (is_neg20 vs not_neg20).
#     """
#     def __init__(self, dropout_rate=0.1):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Linear(1, 64),
#             nn.BatchNorm1d(64),
#             nn.ReLU(),
#             nn.Dropout(dropout_rate),
#             nn.Linear(64, 32),
#             nn.BatchNorm1d(32),
#             nn.ReLU(),
#             nn.Dropout(dropout_rate),
#             nn.Linear(32, 1) # Logits for P(is_neg20)
#         )

#     def forward(self, intensities_expanded):
#         """
#         Args:
#             intensities_expanded: [B, 2672] - intensities for each beam
#         Returns:
#             neg20_logits: [B, 2672] - logits for P(is_neg20)
#         """
#         batch_size = intensities_expanded.size(0)
#         intensities_flat = intensities_expanded.reshape(-1, 1)
#         logits_flat = self.net(intensities_flat)
#         neg20_logits = logits_flat.reshape(batch_size, -1)
#         return neg20_logits

class Neg20DetectorCNN(nn.Module):
    """CNN-based -20 detector: 668 intensities -> 2672 outputs via network architecture"""
    def __init__(self, dropout_rate=0.1):
        super().__init__()

        # Encoder: process 668 intensities
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, padding=1),  # [B, 32, 668]
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),  # [B, 64, 668]
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Conv1d(64, 32, kernel_size=3, padding=1),  # [B, 32, 668]
            nn.BatchNorm1d(32),
            nn.ReLU(),
        )

        # Expand 668 -> 2672 (4x upsampling)
        self.upsample = nn.Sequential(
            nn.ConvTranspose1d(32, 16, kernel_size=8, stride=4, padding=2),  # [B, 16, 2672]
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Conv1d(16, 1, kernel_size=3, padding=1),  # [B, 1, 2672]
        )

    def forward(self, intensities):
        """
        Args:
            intensities: [B, 668] - raw pixel intensities
        Returns:
            neg20_logits: [B, 2672] - expanded to 2672 via transposed conv
        """
        # Convert to [B, 1, 668] format for Conv1d
        x = intensities.unsqueeze(1)  # [B, 1, 668]

        # Process with encoder
        x = self.encoder(x)  # [B, 32, 668]

        # Upsample to 2672
        x = self.upsample(x)  # [B, 1, 2672]

        logits = x.squeeze(1)  # [B, 2672]
        return logits

class ValidVsNeg10CNN(nn.Module):
    """CNN-based Valid vs -10 classifier without positional encoding"""
    def __init__(self, dropout_rate=0.1):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv1d(2, 32, kernel_size=5, padding=2),  # Input: 2 channels (intensity + range)
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2, 2),  # 2672 -> 1336
            nn.Dropout(dropout_rate),
            nn.Conv1d(32, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2, 2),  # 1336 -> 668
            nn.Dropout(dropout_rate),
            nn.Conv1d(64, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
        )

        # Upsample back to original resolution
        self.upsample = nn.Sequential(
            nn.ConvTranspose1d(32, 16, kernel_size=4, stride=2, padding=1),  # 668 -> 1336
            nn.ReLU(),
            nn.ConvTranspose1d(16, 1, kernel_size=4, stride=2, padding=1),   # 1336 -> 2672
        )

    def forward(self, features):
        """
        Args:
            features: [B, 2, 2672] (intensity + range)
        Returns:
            logits: [B, 2672]
        """
        # Features already in [B, 2, 2672] format - ready for Conv1d
        x = features  # [B, 2, 2672]

        # Apply conv layers with pooling
        x = self.net(x)  # [B, 32, 668]

        # Upsample back to original resolution
        logits = self.upsample(x).squeeze(1)  # [B, 2672]
        return logits

# ==============================================================================
# LOSS FUNCTION
# ==============================================================================

class FullThreeStageLoss(nn.Module):
    def __init__(self, alpha_neg20=2.0, alpha_valid_neg10=5.0, beta_reg=1.0):
        super().__init__()
        self.alpha_neg20 = alpha_neg20
        self.alpha_valid_neg10 = alpha_valid_neg10
        self.beta_reg = beta_reg

    def forward(self, final_preds, neg20_logits, valid_vs_neg10_logits, angle_preds, targets_full):

        # Stage 1: Binary Classification for -20 detection
        target_is_neg20 = (targets_full == -20).float()

        # Use pos_weight for -20 class (it's dominant)
        num_neg20 = target_is_neg20.sum()
        num_not_neg20 = (~target_is_neg20.bool()).sum()
        pos_weight_neg20 = (num_not_neg20 / (num_neg20 + 1e-8)).clamp(min=1.0, max=5.0)

        s1_loss = F.binary_cross_entropy_with_logits(
            neg20_logits,
            target_is_neg20,
            pos_weight=pos_weight_neg20.unsqueeze(0)
        )

        # Stage 2: Binary Classification for Valid vs -10 (only on non--20 data)
        non_neg20_mask = (targets_full != -20)

        if non_neg20_mask.any():
            target_is_valid = (targets_full != -10).float()
            num_valid = target_is_valid[non_neg20_mask].sum()
            num_neg10 = (1 - target_is_valid[non_neg20_mask]).sum()
            pos_weight_valid = (num_neg10 / (num_valid + 1e-8)).clamp(min=1.0, max=5.0)

            s2_loss = F.binary_cross_entropy_with_logits(
                valid_vs_neg10_logits[non_neg20_mask],
                target_is_valid[non_neg20_mask],
                pos_weight=pos_weight_valid.unsqueeze(0)
            )
        else:
            s2_loss = torch.tensor(0.0, device=targets_full.device, dtype=targets_full.dtype)

        # Stage 3: Regression Loss (only on Valid positions)
        valid_mask = (targets_full != -10) & (targets_full != -20)

        if valid_mask.any():
            reg_loss = F.mse_loss(angle_preds[valid_mask], targets_full[valid_mask])
        else:
            reg_loss = torch.tensor(0.0, device=targets_full.device, dtype=targets_full.dtype)

        total_loss = (self.alpha_neg20 * s1_loss +
                      self.alpha_valid_neg10 * s2_loss +
                      self.beta_reg * reg_loss)

        return total_loss, s1_loss, s2_loss, reg_loss

class FullThreeStageModelCNN(nn.Module):
    """
    All-CNN architecture with positional embeddings.
    1. CNN -20 Detector
    2. CNN Valid vs -10 Classifier
    3. CNN Angle Regressor
    """
    def __init__(self, prediction_type='phi', dropout_rate=0.1):
        super().__init__()
        self.neg20_detector = Neg20DetectorCNN(dropout_rate=dropout_rate)
        self.valid_vs_neg10_classifier = ValidVsNeg10CNN(dropout_rate=dropout_rate)
        self.angle_regressor = RegressorCNN(prediction_type, dropout_rate)

    def forward(self, intensities_full):
        """
        Args:
            intensities_full: [B, 668]
        Returns:
            final_predictions: [B, 2672]
            neg20_logits: [B, 2672]
            valid_vs_neg10_logits: [B, 2672]
            angle_preds: [B, 2672]
        """
        batch_size = intensities_full.size(0)
        device = intensities_full.device

        min_range = 0.5
        max_range = 40.0
        pixel_ranges = torch.linspace(min_range, max_range, 668, dtype=torch.float32, device=device)
        beam_ranges_expanded_2672 = pixel_ranges.repeat_interleave(4)

        # Prepare inputs
        regressor_input = torch.stack([
            intensities_full,
            pixel_ranges.unsqueeze(0).repeat(batch_size, 1)
        ], dim=1)

        intensities_expanded_2672 = intensities_full.repeat_interleave(4, dim=1)
        classifier_input = torch.stack([
            intensities_expanded_2672,
            beam_ranges_expanded_2672.unsqueeze(0).repeat(batch_size, 1)
        ], dim=1)

        # Stage 1: CNN -20 Detector (operates on 668 intensities, outputs 2672)
        neg20_logits = self.neg20_detector(intensities_full)
        neg20_probs = torch.sigmoid(neg20_logits)

        # Stage 2: CNN Valid vs -10 Classifier
        valid_vs_neg10_logits = self.valid_vs_neg10_classifier(classifier_input)
        valid_vs_neg10_probs = torch.sigmoid(valid_vs_neg10_logits)

        # Stage 3: CNN Angle Regressor
        angle_preds = self.angle_regressor(regressor_input)

        # Final blending - using hard thresholding for both training and inference
        # This ensures training-inference consistency

        # COMMENTED OUT: Old inconsistent implementation
        # if training:
        #     # Soft blending with probabilities (differentiable)
        #     effective_valid_prob = (1 - neg20_probs) * valid_vs_neg10_probs
        #     effective_neg10_prob = (1 - neg20_probs) * (1 - valid_vs_neg10_probs)
        #     final_predictions = (
        #         neg20_probs * (-20.0) +
        #         effective_valid_prob * angle_preds +
        #         effective_neg10_prob * (-10.0)
        #     )
        # else:
        #     # Hard thresholding (non-differentiable)
        #     final_predictions = angle_preds.clone()
        #     is_neg20 = (neg20_probs > 0.15)
        #     is_valid = (valid_vs_neg10_probs > 0.15)
        #     final_predictions[is_neg20] = -20.0
        #     non_neg20_mask = ~is_neg20
        #     final_predictions[non_neg20_mask & ~is_valid] = -10.0
        #     final_predictions[non_neg20_mask & is_valid] = angle_preds[non_neg20_mask & is_valid]

        # NEW: Consistent hard thresholding for both training and inference
        final_predictions = angle_preds.clone()
        is_neg20 = (neg20_probs > 0.78)  # Use 0.5 threshold (standard for binary classification)
        is_valid = (valid_vs_neg10_probs > 0.75)

        # Apply decisions in order: first -20, then -10, rest are angle predictions
        final_predictions[is_neg20] = -20.0
        non_neg20_mask = ~is_neg20
        final_predictions[non_neg20_mask & ~is_valid] = -10.0
        # Valid positions keep their angle predictions (already in final_predictions)

        return final_predictions, neg20_logits, valid_vs_neg10_logits, angle_preds

# ==============================================================================
# TRAINING FUNCTIONS
# ==============================================================================


def train_full_three_stage_model(model, train_loader, val_loader, num_epochs=50):
    device = torch.device('cuda' if torch.cuda.is_available() else
                          'mps' if torch.backends.mps.is_available() else
                          'cpu')
    model.to(device)

    # SEPARATE optimizers for each stage
    optimizer_stage1 = torch.optim.AdamW(model.neg20_detector.parameters(), lr=1e-4, weight_decay=1e-4)
    optimizer_stage2 = torch.optim.AdamW(model.valid_vs_neg10_classifier.parameters(), lr=1e-4, weight_decay=1e-4)
    optimizer_stage3 = torch.optim.AdamW(model.angle_regressor.parameters(), lr=1e-5, weight_decay=1e-4)

    total_steps = num_epochs * len(train_loader)
    if total_steps <= 0:
        raise ValueError("total_steps for scheduler must be > 0.")

    # SEPARATE schedulers for each stage
    scheduler_stage1 = torch.optim.lr_scheduler.OneCycleLR(optimizer_stage1, max_lr=1e-4, total_steps=total_steps, pct_start=0.3)
    scheduler_stage2 = torch.optim.lr_scheduler.OneCycleLR(optimizer_stage2, max_lr=1e-4, total_steps=total_steps, pct_start=0.3)
    scheduler_stage3 = torch.optim.lr_scheduler.OneCycleLR(optimizer_stage3, max_lr=1e-5, total_steps=total_steps, pct_start=0.3)

    criterion = FullThreeStageLoss(alpha_neg20=2.0, alpha_valid_neg10=5.0, beta_reg=2.0)

    print(f"\n{'='*60}")
    print("TRAINING FULL THREE-STAGE CNN MODEL")
    print(f"{'='*60}")
    print(f"Device: {device}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_s1_loss = 0.0
        train_s2_loss = 0.0
        train_reg_loss = 0.0
        step_count = 0

        for batch_idx, batch in enumerate(train_loader):
            intensities, ground_truth = batch
            intensities = intensities.to(device)
            ground_truth = ground_truth.to(device)

            # FORWARD PASS
            final_preds_padded, neg20_logits_padded, valid_vs_neg10_logits_padded, angle_preds_padded = model(intensities)

            # COMPUTE ALL LOSSES
            loss, s1_loss, s2_loss, reg_loss = criterion(
                final_preds_padded, neg20_logits_padded, valid_vs_neg10_logits_padded, angle_preds_padded, ground_truth
            )

            # STAGE 1: Update Neg20Detector only
            if torch.isfinite(s1_loss):
                optimizer_stage1.zero_grad()
                s1_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.neg20_detector.parameters(), max_norm=1.0)
                optimizer_stage1.step()
                scheduler_stage1.step()

            # STAGE 2: Update Transformer only
            if torch.isfinite(s2_loss):
                optimizer_stage2.zero_grad()
                s2_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.valid_vs_neg10_classifier.parameters(), max_norm=1.0)
                optimizer_stage2.step()
                scheduler_stage2.step()

            # STAGE 3: Update Regressor only
            if torch.isfinite(reg_loss):
                optimizer_stage3.zero_grad()
                reg_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.angle_regressor.parameters(), max_norm=1.0)
                optimizer_stage3.step()
                scheduler_stage3.step()

            train_loss += loss.item()
            train_s1_loss += s1_loss.item()
            train_s2_loss += s2_loss.item()
            train_reg_loss += reg_loss.item()
            step_count += 1

        if step_count == 0:
            print("Warning: No training steps performed this epoch.")
            avg_train_loss = avg_s1 = avg_s2 = avg_reg = float('inf')
        else:
            avg_train_loss = train_loss / step_count
            avg_s1 = train_s1_loss / step_count
            avg_s2 = train_s2_loss / step_count
            avg_reg = train_reg_loss / step_count

        # VALIDATION
        model.eval()
        val_loss = 0.0
        val_steps = 0

        with torch.no_grad():
            for batch in val_loader:
                intensities, ground_truth = batch
                intensities = intensities.to(device)
                ground_truth = ground_truth.to(device)

                final_preds_padded, neg20_logits_padded, valid_vs_neg10_logits_padded, angle_preds_padded = model(intensities)
                loss, _, _, _ = criterion(
                    final_preds_padded, neg20_logits_padded, valid_vs_neg10_logits_padded, angle_preds_padded, ground_truth
                )

                if torch.isfinite(loss):
                    val_loss += loss.item()
                    val_steps += 1

        if val_steps == 0:
            avg_val_loss = float('inf')
        else:
            avg_val_loss = val_loss / val_steps

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'best_full_three_stage_model.pth')

        if epoch % 10 == 0 or epoch < 5:
            print(f"Epoch {epoch+1:3d}/{num_epochs} | "
                  f"Train: {avg_train_loss:.4f} "
                  f"(s1:{avg_s1:.3f}, s2:{avg_s2:.3f}, reg:{avg_reg:.4f}) | "
                  f"Val: {avg_val_loss:.4f} | Best: {best_val_loss:.4f}")

    model.load_state_dict(torch.load('best_full_three_stage_model.pth', map_location=device))
    print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}\n")
    return model

# Post-process for 3-class metrics
def map_to_3_classes(values):
    mapped = np.zeros_like(values, dtype=int)
    mapped[values == -10] = 1 # (-10)
    mapped[values == -20] = 2 # (-20)
    # Anything else is Valid (0)
    return mapped

def run_inference_for_csv(dataset, model, device):
    results = {}
    for i in range(len(dataset)):
        intensities, ground_truth = dataset[i]
        final_pred_padded, _, _, _ = model(intensities.unsqueeze(0).to(device))
        final_pred = final_pred_padded.squeeze(0).cpu()

        results[i] = {
            'pred_phis': final_pred.view(4, 668).detach().numpy(),
            'true_phis': ground_truth.view(4, 668).detach().numpy()
        }
    return results

# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

def main():
    csv_file = '/home/farhang/fls_ws/src/fls_reconstruction/results/test/fls_all.csv'

    print("\n" + "="*60)
    print("FULL THREE-STAGE MODEL: NEURAL -20 + TRANSFORMER + REGRESSOR")
    print("="*60)

    # Create splits (always recreate to use fresh data)
    splits_dir = "./data_splits"
    if os.path.exists(splits_dir):
        import shutil
        shutil.rmtree(splits_dir)
    print("\nCreating data splits...")
    train_csv, val_csv, test_csv = save_splits_to_csv(csv_file, splits_dir)

    # Load FULL datasets (including -20s)
    print("\nLoading full datasets (with -20s)...")
    datasets = {
        'train': BathymetryDataset(train_csv, prediction_type='phi'),
        'val': BathymetryDataset(val_csv, prediction_type='phi'),
        'test': BathymetryDataset(test_csv, prediction_type='phi')
    }

    print(f"Train: {len(datasets['train'])} samples")
    print(f"Val: {len(datasets['val'])} samples")
    print(f"Test: {len(datasets['test'])} samples")

    # Create data loaders (no custom collate needed for full dataset)
    loaders = {
        'train': DataLoader(datasets['train'], batch_size=4, shuffle=True),
        'val': DataLoader(datasets['val'], batch_size=4, shuffle=False),
        'test': DataLoader(datasets['test'], batch_size=4, shuffle=False)
    }

    # Create or load model
    model = FullThreeStageModelCNN(prediction_type='phi', dropout_rate=0.1)

    model_path = 'best_full_three_stage_model.pth'
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    # Always train a new model (delete old one if exists)
    if os.path.exists(model_path):
        os.remove(model_path)

    print("\nTraining new full three-stage model...")
    model = train_full_three_stage_model(
        model,
        loaders['train'],
        loaders['val'],
        num_epochs=60
    )

    # Test
    print("\n" + "="*60)
    print("TESTING FULL THREE-STAGE MODEL")
    print("="*60)

    model.to(device).eval()

    all_final_preds = []
    all_ground_truths = []

    with torch.no_grad():
        for intensities, ground_truth in loaders['test']:
            intensities = intensities.to(device)

            final_pred_padded, _, _, _ = model(intensities)

            all_final_preds.extend(final_pred_padded.flatten().cpu().tolist())
            all_ground_truths.extend(ground_truth.flatten().cpu().tolist())



    true_labels_3class = map_to_3_classes(np.array(all_ground_truths))
    pred_labels_3class = map_to_3_classes(np.array(all_final_preds))

    print("\n" + "="*60)
    print("3-Class Classification Report (Full Test Set):")
    print("="*60)
    print(classification_report(true_labels_3class, pred_labels_3class,
                                target_names=['Valid', '(-10)', '(-20)']))

    # Generate CSVs
    print("\n" + "="*60)
    print("GENERATING PREDICTION CSVS")
    print("="*60)

    test_results = {}
    train_results = {}



    test_results = run_inference_for_csv(datasets['test'], model, device)
    train_results = run_inference_for_csv(datasets['train'], model, device)

    test_with_indices = save_splits_with_indices_to_csv(csv_file, splits_dir, split_type='test')
    train_with_indices = save_splits_with_indices_to_csv(csv_file, splits_dir, split_type='train')

    tangent_results_test = {i: {'pred_tangents': r['pred_phis'], 'true_tangents': r['true_phis']}
                           for i, r in test_results.items()}
    tangent_results_train = {i: {'pred_tangents': r['pred_phis'], 'true_tangents': r['true_phis']}
                            for i, r in train_results.items()}

    test_predictions_path = "./data_splits/test_predictions_full_three_stage.csv"
    save_predictions_with_indices_to_csv(
        tangent_results_test, test_results, test_with_indices, test_predictions_path
    )
    print(f" Test predictions saved to {test_predictions_path}")

    train_predictions_path = "./data_splits/train_predictions_full_three_stage.csv"
    save_predictions_with_indices_to_csv(
        tangent_results_train, train_results, train_with_indices, train_predictions_path
    )
    print(f"Train predictions saved to {train_predictions_path}")

    print("\n" + "="*60)
    print("COMPLETE!")
    print("="*60)
    print(f"\nModel saved: best_full_three_stage_model.pth")
    print(f"Test CSV: {test_predictions_path}")
    print(f"Train CSV: {train_predictions_path}")
    print("="*60)


if __name__ == "__main__":
    main()