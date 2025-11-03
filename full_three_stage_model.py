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
            nn.Conv1d(2, 64, 15, padding=7),
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
            nn.Sequential(
                nn.Conv1d(256, 256, 3, padding=1),
                nn.InstanceNorm1d(256),
                nn.LeakyReLU(inplace=True),
                nn.Dropout(dropout_rate),
                nn.Conv1d(256, 256, 3, padding=1),
                nn.InstanceNorm1d(256),
            ) for _ in range(5)
        ])

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

        self.final_upsample = nn.Sequential(
            nn.ConvTranspose1d(128, 32, 15, stride=4, padding=7, output_padding=3),
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
        # if x.dim() == 2:
        #     batch_size = x.size(0)
        #     x = x.view(batch_size, 1, 668)

        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)

        b = e3
        for block in self.residual_blocks:
            b = block(b)

        d1 = self.dec1(b)
        d1 = torch.cat([d1, e2], dim=1)

        d2 = self.dec2(d1)
        d2 = torch.cat([d2, e1], dim=1)

        features = self.final_upsample(d2)

        angle_pred = self.regressor(features)
        return angle_pred.squeeze(1) if angle_pred.size(1) == 1 else angle_pred


class Neg20Detector(nn.Module):
    """
    MLP to detect -20s based on intensity alone.
    Outputs logits for binary classification (is_neg20 vs not_neg20).
    """
    def __init__(self, dropout_rate=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(32, 1) # Logits for P(is_neg20)
        )

    def forward(self, intensities_expanded):
        """
        Args:
            intensities_expanded: [B, 2672] - intensities for each beam
        Returns:
            neg20_logits: [B, 2672] - logits for P(is_neg20)
        """
        batch_size = intensities_expanded.size(0)
        intensities_flat = intensities_expanded.reshape(-1, 1)
        logits_flat = self.net(intensities_flat)
        neg20_logits = logits_flat.reshape(batch_size, -1)
        return neg20_logits


class PureTransformerBinaryClassifier(nn.Module):
    def __init__(self, d_model=128, nhead=8, num_layers=3, dropout=0.2): # d_model = 128
        super().__init__()

        self.input_embed = nn.Sequential(
            nn.Linear(2, d_model // 2), # 64
            nn.LayerNorm(d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, d_model) # 128
        )

        self.pos_embed = nn.Parameter(torch.randn(1, 2672, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, # 128
            nhead=nhead,
            dim_feedforward=d_model * 4, # 512
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2), # 128 -> 64
            nn.LayerNorm(d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, d_model // 4), # 64 -> 32
            nn.LayerNorm(d_model // 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 4, 1) # 32 -> 1
        )

    def forward(self, features, src_key_padding_mask=None): # Input is now 'features'
        batch_size, seq_len, _ = features.shape # Features shape [B, seq_len, 2]

        x = self.input_embed(features) # Process 2 features per token
        x = x + self.pos_embed[:, :seq_len, :]

        x = self.transformer(x, src_key_padding_mask=src_key_padding_mask)

        logits = self.classifier(x).squeeze(-1)

        return logits


class FullThreeStageModel(nn.Module):
    """
    Combines all components for a complete 3-class prediction pipeline.
    1. Neural -20 Detector
    2. Pure Transformer for Valid vs -10
    3. Regressor for angles
    """
    def __init__(self, prediction_type='phi', dropout_rate=0.1):
        super().__init__()
        self.neg20_detector = Neg20Detector(dropout_rate=dropout_rate)
        self.valid_vs_neg10_classifier = PureTransformerBinaryClassifier(
            d_model=128, # Pass the new d_model
            nhead=8,
            num_layers=3, # Pass the new num_layers
            dropout=dropout_rate
        )
        self.angle_regressor = RegressorCNN(prediction_type, dropout_rate)

    def forward(self, intensities_full, training=True):
        """
        Args:
            intensities_full: [B, 668] - raw pixel intensities
            training: boolean flag
        Returns:
            final_predictions: [B, 2672] - 3-class predictions (angle, -10, or -20)
            neg20_logits: [B, 2672] - raw logits for -20 detection
            valid_vs_neg10_logits: [B, 2672] - raw logits for Valid vs -10
            angle_preds: [B, 2672] - raw angle predictions
        """
        batch_size = intensities_full.size(0)
        device = intensities_full.device

        # --- Prepare Range Data (needed for both Regressor and Transformer) ---
        min_range = 0.5 # Ensure these are defined or passed to init
        max_range = 40.0
        pixel_ranges = torch.linspace(min_range, max_range, 668, dtype=torch.float32, device=device)
        beam_ranges_expanded_2672 = pixel_ranges.repeat_interleave(4) # [2672]

        # Create [B, 2, 668] input for RegressorCNN
        regressor_input = torch.stack([
            intensities_full,
            pixel_ranges.unsqueeze(0).repeat(batch_size, 1)
        ], dim=1) # Shape [B, 2, 668]

        # Create [B, 2672, 2] input for Transformer (intensity + range for each beam position)
        intensities_expanded_2672 = intensities_full.repeat_interleave(4, dim=1) # [B, 2672]
        transformer_input = torch.stack([
            intensities_expanded_2672,
            beam_ranges_expanded_2672.unsqueeze(0).repeat(batch_size, 1)
        ], dim=2) # Shape [B, 2672, 2]

        # --- Model Calls ---

        # 1. Neural -20 Detector (takes [B, 2672] expanded intensities)
        neg20_logits = self.neg20_detector(intensities_expanded_2672) # [B, 2672]
        neg20_probs = torch.sigmoid(neg20_logits)

        # Generate the src_key_padding_mask for the Transformer (True where position should be ignored)
        src_key_padding_mask = (neg20_probs > 0.5).to(device) # [B, 2672], True where -20

        # 2. Pure Transformer for Valid vs -10 (Pass the 2-channel input and the mask)
        valid_vs_neg10_logits = self.valid_vs_neg10_classifier(
            transformer_input, # Pass the 2-feature tensor [B, 2672, 2]
            src_key_padding_mask=src_key_padding_mask # Pass the mask
        )
        valid_vs_neg10_probs = torch.sigmoid(valid_vs_neg10_logits)

        # 3. Angle Regressor (Pass the [B, 2, 668] input)
        angle_preds = self.angle_regressor(regressor_input) # [B, 2672]

        # --- Final Prediction Blending (same logic as before) ---
        if training:
            # Soft blending for training
            final_predictions = (
                neg20_probs * (-20.0) +
                (1 - neg20_probs) * (
                    valid_vs_neg10_probs * angle_preds +
                    (1 - valid_vs_neg10_probs) * (-10.0)
                )
            )
        else:
            # Hard decisions for inference
            final_predictions = angle_preds.clone()

            is_neg20 = (neg20_probs > 0.5)
            is_valid = (valid_vs_neg10_probs > 0.5)

            final_predictions[is_neg20] = -20.0

            non_neg20_mask = ~is_neg20
            final_predictions[non_neg20_mask & ~is_valid] = -10.0
            final_predictions[non_neg20_mask & is_valid] = angle_preds[non_neg20_mask & is_valid]

        return final_predictions, neg20_logits, valid_vs_neg10_logits, angle_preds


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
        pos_weight_neg20 = (num_not_neg20 / (num_neg20 + 1e-8)).clamp(min=1.0, max=5.0) # Downweight dominant -20

        s1_loss = F.binary_cross_entropy_with_logits(
            neg20_logits,
            target_is_neg20,
            pos_weight=pos_weight_neg20.unsqueeze(0)
        )

        # Stage 2: Binary Classification for Valid vs -10 (only on non--20 data)
        non_neg20_mask = (targets_full != -20)
        s2_loss = torch.tensor(0.0, device=targets_full.device)

        if non_neg20_mask.any():
            target_is_valid = (targets_full != -10).float() # 1=Valid, 0=-10

            # Use pos_weight for Valid class (it's rarer among non--20)
            num_valid = target_is_valid[non_neg20_mask].sum()
            num_neg10 = (1 - target_is_valid[non_neg20_mask]).sum()
            pos_weight_valid = (num_neg10 / (num_valid + 1e-8)).clamp(min=1.0, max=5.0) # Upweight Valid

            s2_loss = F.binary_cross_entropy_with_logits(
                valid_vs_neg10_logits[non_neg20_mask],
                target_is_valid[non_neg20_mask],
                pos_weight=pos_weight_valid.unsqueeze(0)
            )

        # Stage 3: Regression Loss (only on Valid positions)
        valid_mask = (targets_full != -10) & (targets_full != -20)
        reg_loss = torch.tensor(0.0, device=targets_full.device)

        if valid_mask.any():
            reg_loss = F.mse_loss(angle_preds[valid_mask], targets_full[valid_mask])

        total_loss = (self.alpha_neg20 * s1_loss +
                      self.alpha_valid_neg10 * s2_loss +
                      self.beta_reg * reg_loss)

        return total_loss, s1_loss, s2_loss, reg_loss


# ==============================================================================
# TRAINING FUNCTIONS
# ==============================================================================

# def train_full_three_stage_model(model, train_loader, val_loader, num_epochs=50):
#     # choose device (include cuda, mps, cpu)
#     device = torch.device('cuda' if torch.cuda.is_available() else
#                           'mps' if torch.backends.mps.is_available() else
#                           'cpu')
#     model.to(device)

#     optimizer = torch.optim.AdamW([
#         {'params': model.neg20_detector.parameters(), 'lr': 1e-4},
#         {'params': model.valid_vs_neg10_classifier.parameters(), 'lr': 1e-4},
#         {'params': model.angle_regressor.parameters(), 'lr': 1e-5}
#     ], weight_decay=1e-4)

#     total_steps = num_epochs * len(train_loader)
#     if total_steps <= 0:
#         raise ValueError("total_steps for scheduler must be > 0. "
#                          "Check that train_loader is not empty and num_epochs > 0.")

#     scheduler = torch.optim.lr_scheduler.OneCycleLR(
#         optimizer,
#         max_lr=[1e-4, 1e-4, 1e-5],
#         total_steps=total_steps,
#         pct_start=0.3
#     )

#     criterion = FullThreeStageLoss(alpha_neg20=2.0, alpha_valid_neg10=5.0, beta_reg=1.0)

#     print(f"\n{'='*60}")
#     print("TRAINING FULL THREE-STAGE MODEL")
#     print(f"{'='*60}")
#     print(f"Device: {device}")
#     print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

#     best_val_loss = float('inf')

#     for epoch in range(num_epochs):
#         model.train()
#         train_loss = 0.0
#         train_s1_loss = 0.0
#         train_s2_loss = 0.0
#         train_reg_loss = 0.0
#         step_count = 0

#         for batch in train_loader:
#             # train_loader yields (intensities, ground_truth)
#             intensities, ground_truth = batch
#             intensities = intensities.to(device)
#             ground_truth = ground_truth.to(device)

#             optimizer.zero_grad()

#             final_preds_padded, neg20_logits_padded, valid_vs_neg10_logits_padded, angle_preds_padded = model(intensities, training=True)

#             # compute loss directly on these padded tensors; loss handles internal masks
#             loss, s1_loss, s2_loss, reg_loss = criterion(
#                 final_preds_padded, neg20_logits_padded, valid_vs_neg10_logits_padded, angle_preds_padded, ground_truth
#             )

#             if torch.isfinite(loss):
#                 loss.backward()
#                 torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
#                 optimizer.step()
#                 # step the scheduler once per optimizer.step
#                 scheduler.step()

#                 train_loss += loss.item()
#                 train_s1_loss += s1_loss.item()
#                 train_s2_loss += s2_loss.item()
#                 train_reg_loss += reg_loss.item()
#                 step_count += 1

#         # avoid division by zero if dataloader had 0 batches
#         if step_count == 0:
#             print("Warning: No training steps performed this epoch (train_loader empty?).")
#             avg_train_loss = avg_s1 = avg_s2 = avg_reg = float('inf')
#         else:
#             avg_train_loss = train_loss / step_count
#             avg_s1 = train_s1_loss / step_count
#             avg_s2 = train_s2_loss / step_count
#             avg_reg = train_reg_loss / step_count

#         # VALIDATION
#         model.eval()
#         val_loss = 0.0
#         val_steps = 0

#         with torch.no_grad():
#             for batch in val_loader:
#                 intensities, ground_truth = batch
#                 intensities = intensities.to(device)
#                 ground_truth = ground_truth.to(device)

#                 final_preds_padded, neg20_logits_padded, valid_vs_neg10_logits_padded, angle_preds_padded = model(intensities, training=True)

#                 # full-batch loss; FullThreeStageLoss internally handles ignoring -20/-10 etc
#                 loss, _, _, _ = criterion(
#                     final_preds_padded, neg20_logits_padded, valid_vs_neg10_logits_padded, angle_preds_padded, ground_truth
#                 )

#                 if torch.isfinite(loss):
#                     val_loss += loss.item()
#                     val_steps += 1

#         if val_steps == 0:
#             avg_val_loss = float('inf')
#         else:
#             avg_val_loss = val_loss / val_steps

#         if avg_val_loss < best_val_loss:
#             best_val_loss = avg_val_loss
#             torch.save(model.state_dict(), 'best_full_three_stage_model.pth')

#         if epoch % 10 == 0 or epoch < 5:
#             print(f"Epoch {epoch+1:3d}/{num_epochs} | "
#                   f"Train: {avg_train_loss:.4f} "
#                   f"(s1:{avg_s1:.3f}, s2:{avg_s2:.3f}, reg:{avg_reg:.4f}) | "
#                   f"Val: {avg_val_loss:.4f} | Best: {best_val_loss:.4f}")

#     # load best
#     model.load_state_dict(torch.load('best_full_three_stage_model.pth', map_location=device))
#     print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}\n")
#     return model

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

    criterion = FullThreeStageLoss(alpha_neg20=2.0, alpha_valid_neg10=5.0, beta_reg=1.0)

    print(f"\n{'='*60}")
    print("TRAINING FULL THREE-STAGE MODEL (SEPARATE LOSSES)")
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

        for batch in train_loader:
            intensities, ground_truth = batch
            intensities = intensities.to(device)
            ground_truth = ground_truth.to(device)

            # FORWARD PASS
            final_preds_padded, neg20_logits_padded, valid_vs_neg10_logits_padded, angle_preds_padded = model(intensities, training=True)

            # COMPUTE ALL LOSSES
            loss, s1_loss, s2_loss, reg_loss = criterion(
                final_preds_padded, neg20_logits_padded, valid_vs_neg10_logits_padded, angle_preds_padded, ground_truth
            )

            if torch.isfinite(s1_loss):
                # STAGE 1: Update Neg20Detector only
                optimizer_stage1.zero_grad()
                s1_loss.backward(retain_graph=True)
                torch.nn.utils.clip_grad_norm_(model.neg20_detector.parameters(), max_norm=1.0)
                optimizer_stage1.step()
                scheduler_stage1.step()

            if torch.isfinite(s2_loss):
                # STAGE 2: Update Transformer only
                optimizer_stage2.zero_grad()
                s2_loss.backward(retain_graph=True)
                torch.nn.utils.clip_grad_norm_(model.valid_vs_neg10_classifier.parameters(), max_norm=1.0)
                optimizer_stage2.step()
                scheduler_stage2.step()

            if torch.isfinite(reg_loss):
                # STAGE 3: Update Regressor only
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

                final_preds_padded, neg20_logits_padded, valid_vs_neg10_logits_padded, angle_preds_padded = model(intensities, training=True)
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
        final_pred_padded, _, _, _ = model(intensities.unsqueeze(0).to(device), training=False)
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
    csv_file = '/home/farhang/Downloads/fls_all_with_phi.csv'

    print("\n" + "="*60)
    print("FULL THREE-STAGE MODEL: NEURAL -20 + TRANSFORMER + REGRESSOR")
    print("="*60)

    # Create splits
    splits_dir = "./data_splits"
    if not os.path.exists(f"{splits_dir}/train_data.csv"):
        print("\nCreating data splits...")
        train_csv, val_csv, test_csv = save_splits_to_csv(csv_file, splits_dir)
    else:
        print("\nUsing existing data splits...")
        train_csv = f"{splits_dir}/train_data.csv"
        val_csv = f"{splits_dir}/val_data.csv"
        test_csv = f"{splits_dir}/test_data.csv"

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
        'train': DataLoader(datasets['train'], batch_size=8, shuffle=True),
        'val': DataLoader(datasets['val'], batch_size=8, shuffle=False),
        'test': DataLoader(datasets['test'], batch_size=8, shuffle=False)
    }

    # Create or load model
    model = FullThreeStageModel(prediction_type='phi', dropout_rate=0.1)

    model_path = 'best_full_three_stage_model.pth'
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    if os.path.exists(model_path):
        print(f"\nLoading existing model from {model_path}")
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
    else:
        print("\nTraining new full three-stage model...")
        model = train_full_three_stage_model(
            model,
            loaders['train'],
            loaders['val'],
            num_epochs=100
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

            final_pred_padded, _, _, _ = model(intensities, training=False)

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