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

# def reshape_predictions(predictions):
#     """
#     Reshape [batch_size, 2672] to phis [batch_size, 4, 668].

#     For tangent mode, uncomment the lines below and modify output dimensions:
#     # tangents = reshaped[:, :, :4].transpose(1, 2)  # [batch, 4, 668]
#     # phis = reshaped[:, :, 4:].transpose(1, 2)      # [batch, 4, 668]
#     """
#     if predictions.dim() == 1:
#         predictions = predictions.unsqueeze(0)
#         squeeze_output = True
#     else:
#         squeeze_output = False

#     batch_size = predictions.size(0)

#     # Current mode: phis only
#     phis = torch.zeros(batch_size, 4, 668)
#     for i in range(4):
#         phis[:, i, :] = predictions[:, i::4]

#     # For tangent + phi mode, use this instead:
#     # reshaped = predictions.view(batch_size, 668, 8)
#     # tangents = reshaped[:, :, :4].transpose(1, 2)
#     # phis = reshaped[:, :, 4:].transpose(1, 2)

#     tangents = torch.zeros_like(phis)  # Disabled for phi-only mode

#     if squeeze_output:
#         tangents = tangents.squeeze(0)
#         phis = phis.squeeze(0)

#     return tangents, phis

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
    original_df = pd.read_csv(original_csv_path)
    # print(f"Total columns: {len(original_df.columns)}")
    # print(f"Column names: {list(original_df.columns)}")
    output_df = original_df.copy()

    # print(f"Original CSV shape: {original_df.shape}")

    for row_idx in tangent_results.keys():
        if row_idx < len(output_df) and row_idx in phi_results:
            pred_tangents = tangent_results[row_idx]['pred_tangents']  # Shape: [4, 668]
            pred_phis = phi_results[row_idx]['pred_phis']              # Shape: [4, 668]

            # KEEP timestamp (column 0) and intensities (columns 1-668) unchanged

            # Replace tangent columns (columns 669-3340, which is 2672 columns)
            tangent_start = 669  # After timestamp + 668 intensities
            tangent_flat = pred_tangents.flatten()  # Convert [4,668] to [2672]

            for i, val in enumerate(tangent_flat):
                if tangent_start + i < len(output_df.columns):
                    output_df.iloc[row_idx, tangent_start + i] = val

            # Replace phi columns (columns 3341-6012, which is 2672 columns)
            phi_start = 669 + 2672  # After timestamp + intensities + tangents
            phi_flat = pred_phis.flatten()  # Convert [4,668] to [2672]

            for i, val in enumerate(phi_flat):
                if phi_start + i < len(output_df.columns):
                    output_df.iloc[row_idx, phi_start + i] = val

    output_df.to_csv(output_csv_path, index=False)
    print(f"Predictions saved to {output_csv_path}")
    return output_df

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

class IntensityToBathymetryPhiUNet1D(nn.Module):
    def __init__(self, dropout_rate=0.1):
        super().__init__()
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
        self.regressor = nn.Conv1d(32, 1, 3, padding=1)   # angle values

        # For tangent + phi mode, change output dimensions:
        # self.regressor = nn.Conv1d(32, 2, 3, padding=1)  # tangent + phi values

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
        angle_pred = self.regressor(features)     # [B, 1, 2672]

        first_logits = class_logits[0]  # shape: [3, 2672]

        # ##################
        # first_logits = class_logits[0]
        # # Move to CPU and convert to NumPy
        # first_logits_np = first_logits.detach().cpu().numpy()  # shape: [3, 2672]

        # # Create a DataFrame where each row is one class (3 rows, 2672 columns)
        # df = pd.DataFrame(first_logits_np)

        # # Save to CSV
        # df.to_csv("class_logits_first_sample.csv", index=False)

        # # Print the first 10 values (flattened)
        # print("First 10 class logits of the first sample:")
        # print(first_logits_np.flatten()[:10])
        # ##################

        return class_logits, angle_pred.squeeze(1)

class IntensityToBathymetryTangentsUNet1D(nn.Module):
    def __init__(self, dropout_rate=0.1):
        super().__init__()
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
        self.regressor = nn.Conv1d(32, 1, 3, padding=1)   # angle values

        # For tangent + phi mode, change output dimensions:
        # self.regressor = nn.Conv1d(32, 2, 3, padding=1)  # tangent + phi values

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
        angle_pred = self.regressor(features)     # [B, 1, 2672]

        first_logits = class_logits[0]  # shape: [3, 2672]

        # ##################
        # first_logits = class_logits[0]
        # # Move to CPU and convert to NumPy
        # first_logits_np = first_logits.detach().cpu().numpy()  # shape: [3, 2672]

        # # Create a DataFrame where each row is one class (3 rows, 2672 columns)
        # df = pd.DataFrame(first_logits_np)

        # # Save to CSV
        # df.to_csv("class_logits_first_sample.csv", index=False)

        # # Print the first 10 values (flattened)
        # print("First 10 class logits of the first sample:")
        # print(first_logits_np.flatten()[:10])
        # ##################

        return class_logits, angle_pred.squeeze(1)

# ==================== DATA HANDLING ====================

class BathymetryPhiDataset(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        # Extract data: timestamps(0), intensities(1-668), tangents(669-3340), phis(3341-6012)
        intensities = torch.tensor(row.iloc[1:669].values, dtype=torch.float32)
        # tangents = torch.tensor(row.iloc[669:3341].values, dtype=torch.float32)
        phis = torch.tensor(row.iloc[3341:6013].values, dtype=torch.float32)

        # Process data (NaN → -20.0)
        intensities_processed, _ = create_valid_mask(intensities)
        # tangents_processed, _ = create_valid_mask(tangents)
        phis_processed, _ = create_valid_mask(phis)

        # Current mode: phis only
        phis_processed = phis_processed.view(4, 668)
        ground_truth = phis_processed.contiguous().view(-1)

        # For tangent + phi mode, use:
        # tangents = tangents_processed.view(4, 668)
        # phis = phis_processed.view(4, 668)
        # ground_truth = torch.stack([tangents, phis], dim=0).view(8, 668).transpose(0, 1).contiguous().view(-1)

        return intensities_processed, ground_truth

class BathymetryTangentDataset(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        # Extract data: timestamps(0), intensities(1-668), tangents(669-3340), phis(3341-6012)
        intensities = torch.tensor(row.iloc[1:669].values, dtype=torch.float32)
        tangents = torch.tensor(row.iloc[669:3341].values, dtype=torch.float32)
        # phis = torch.tensor(row.iloc[3341:6013].values, dtype=torch.float32)  # Not needed

        # Process data (NaN → -20.0)
        intensities_processed, _ = create_valid_mask(intensities)
        tangents_processed, _ = create_valid_mask(tangents)

        # Use tangents as ground truth
        tangents_processed = tangents_processed.view(4, 668)
        ground_truth = tangents_processed.contiguous().view(-1)

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

        # Classification loss for all positions
        class_logits = class_logits.permute(0, 2, 1)  # [B, 2672, 3]
        class_loss = F.cross_entropy(
            class_logits.reshape(-1, 3),
            class_labels.reshape(-1)
        )

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

# def train_model(model, train_loader, val_loader, num_epochs=300, model_name="bathymetry"):
#     """Train the model with early stopping."""
#     device = get_device()
#     model.to(device)

#     optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
#     scheduler = torch.optim.lr_scheduler.OneCycleLR(
#         optimizer, max_lr=1e-2, total_steps=num_epochs,
#         pct_start=0.3, anneal_strategy='cos'
#     )

#     print(f"Training on {device}")
#     print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

#     best_val_loss = float('inf')
#     patience_counter = 0
#     patience = 60

#     train_losses = []
#     val_losses = []

#     for epoch in range(num_epochs):
#         train_loss = run_epoch(model, train_loader, device, optimizer)
#         val_loss = run_epoch(model, val_loader, device)
#         scheduler.step()

#         train_losses.append(train_loss)
#         val_losses.append(val_loss)

#         # Early stopping
#         if val_loss < best_val_loss - 1e-6:
#             best_val_loss = val_loss
#             patience_counter = 0
#             torch.save(model.state_dict(), 'best_bathymetry_model.pth')
#         else:
#             patience_counter += 1

#         if epoch % 10 == 0 or epoch < 5:
#             print(f"Epoch {epoch+1:3d}/{num_epochs} | Train: {train_loss:.6f} | Val: {val_loss:.6f} | "
#                   f"Best: {best_val_loss:.6f} | LR: {optimizer.param_groups[0]['lr']:.2e}")

#         if patience_counter >= patience:
#             print(f"Early stopping at epoch {epoch+1}")
#             break

#     # Load best model
#     model.load_state_dict(torch.load('best_bathymetry_model.pth', map_location=device))
#     print(f"Training complete. Best validation loss: {best_val_loss:.6f}")

#     plt.figure(figsize=(10, 6))
#     plt.plot(train_losses, label='Training Loss', alpha=0.8)
#     plt.plot(val_losses, label='Validation Loss', alpha=0.8)
#     plt.xlabel('Epoch')
#     plt.ylabel('Loss')
#     plt.title('Training and Validation Loss')
#     plt.legend()
#     plt.grid(True, alpha=0.3)
#     plt.savefig('training_curves.png', dpi=300, bbox_inches='tight')
#     plt.close()
#     print(f"Training curves saved: training_curves.png")

def train_model(model, train_loader, val_loader, num_epochs=300, model_name=None):
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

# def test_specific_rows(model, dataset, num_test_rows=5, output_dir='test_outputs'):
#     """Test model on specific rows and save results."""
#     ensure_dir(output_dir)
#     device = get_device()
#     model.to(device).eval()

#     test_indices = list(range(min(num_test_rows, len(dataset))))
#     results = {}

#     print(f"\nTesting {len(test_indices)} samples...")

#     with torch.no_grad():
#         for i in test_indices:
#             intensities, ground_truth_raw = dataset[i]
#             actual_idx = dataset.indices[i] if hasattr(dataset, 'indices') else i

#             # Make prediction
#             prediction, class_logits, angle_preds = predict_with_model(model, intensities, device)

#             prediction = prediction.squeeze(0)

#             ########
#             # Get predicted classes
#             class_preds = torch.argmax(class_logits, dim=1).squeeze(0)  # [2672]

#             # Create ground truth classes
#             gt_classes = torch.zeros_like(ground_truth_raw, dtype=torch.long)
#             gt_classes[ground_truth_raw == -10.0] = 1
#             gt_classes[ground_truth_raw == -20.0] = 2

#             # Reshape classes for each phi (4 beams)
#             pred_classes_reshaped = torch.zeros(4, 668)
#             gt_classes_reshaped = torch.zeros(4, 668)
#             for j in range(4):
#                 pred_classes_reshaped[j, :] = class_preds[j::4]
#                 gt_classes_reshaped[j, :] = gt_classes[j::4]
#             ##########

#             # Evaluate
#             row_metrics = evaluate_single_row(prediction, ground_truth_raw)

#             # Print clean summary
#             print(f"Sample {i:2d} (CSV row {actual_idx:4d}) | "
#                   f"Angle RMSE: {row_metrics['angle_rmse']:6.3f} | "
#                   f"Flag Acc: {row_metrics['flag_acc']:5.1%} | "
#                   f"Valid: {row_metrics['clean_points']:3d}/{row_metrics['valid_points']:3d}")

#             # Reshape for visualization
#             pred_tangents, pred_phis = reshape_predictions(prediction)
#             gt_processed, _ = create_valid_mask(ground_truth_raw)
#             gt_tangents, gt_phis = reshape_predictions(gt_processed)

#             # Store results
#             results[i] = {
#                 'intensities': intensities.numpy(),
#                 'pred_tangents': pred_tangents.numpy(),
#                 'pred_phis': pred_phis.numpy(),
#                 'gt_tangents': gt_tangents.numpy(),
#                 'gt_phis': gt_phis.numpy(),
#                 'classes_phi': class_logits.numpy(),
#                 'pred_classes': pred_classes_reshaped.numpy(),
#                 'gt_classes': gt_classes_reshaped.numpy(),
#                 'metrics': row_metrics
#             }

#             # Save CSV
#             data = {'pixel_idx': range(668), 'intensities': intensities.numpy()}
#             for j in range(4):
#                 data[f'pred_phi_{j}'] = pred_phis.numpy()[j]
#                 data[f'gt_phi_{j}'] = gt_phis.numpy()[j]
#                 # For tangent mode, add:
#                 data[f'pred_tangent_{j}'] = pred_tangents.numpy()[j]
#                 data[f'gt_tangent_{j}'] = gt_tangents.numpy()[j]

#             pd.DataFrame(data).to_csv(f'{output_dir}/row_{actual_idx}_predictions.csv', index=False)

#     print(f"Results saved to {output_dir}/")
#     return results

# def test_specific_rows(model, dataset, num_test_rows=5, output_dir='test_outputs', prediction_type=None):
#     """Test model on specific rows and save results."""
#     ensure_dir(output_dir)
#     device = get_device()
#     model.to(device).eval()

#     test_indices = list(range(min(num_test_rows, len(dataset))))
#     results = {}

#     print(f"\nTesting {len(test_indices)} samples for {prediction_type}...")

#     with torch.no_grad():
#         for i in test_indices:
#             intensities, ground_truth_raw = dataset[i]
#             actual_idx = dataset.indices[i] if hasattr(dataset, 'indices') else i

#             # Make prediction
#             prediction, class_logits, angle_preds = predict_with_model(model, intensities, device)
#             prediction = prediction.squeeze(0)

#             # Get predicted classes
#             class_preds = torch.argmax(class_logits, dim=1).squeeze(0)  # [2672]

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

#             # Print clean summary
#             print(f"Sample {i:2d} (CSV row {actual_idx:4d}) | "
#                   f"Angle RMSE: {row_metrics['angle_rmse']:6.3f} | "
#                   f"Flag Acc: {row_metrics['flag_acc']:5.1%} | "
#                   f"Valid: {row_metrics['clean_points']:3d}/{row_metrics['valid_points']:3d}")

#             # Reshape for visualization - USE THE PREDICTION_TYPE PARAMETER
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
#                 'classes_phi': class_logits.numpy(),
#                 'pred_classes': pred_classes_reshaped.numpy(),
#                 'gt_classes': gt_classes_reshaped.numpy(),
#                 'metrics': row_metrics,
#                 'prediction_type': prediction_type
#             }

#             # Save CSV
#             data = {'pixel_idx': range(668), 'intensities': intensities.numpy()}
#             for j in range(4):
#                 if prediction_type in ['phi', 'combined']:
#                     data[f'pred_phi_{j}'] = pred_phis.numpy()[j]
#                     data[f'gt_phi_{j}'] = gt_phis.numpy()[j]
#                 if prediction_type in ['tangent', 'combined']:
#                     data[f'pred_tangent_{j}'] = pred_tangents.numpy()[j]
#                     data[f'gt_tangent_{j}'] = gt_tangents.numpy()[j]
#                 else:
#                     print("No valid Prediction type passed to inference")

#             pd.DataFrame(data).to_csv(f'{output_dir}/row_{actual_idx}_predictions.csv', index=False)

#     print(f"Results saved to {output_dir}/")
#     return results

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
        # fig, axes = plt.subplots(2, 4, figsize=(16, 6), sharex=True, sharey=True)
        # fig, axes = plt.subplots(3, 4, figsize=(16, 9), sharex=True)
        fig, axes = plt.subplots(4, 4, figsize=(16, 9), sharex=True)

        pred_tangents = result['pred_tangents']
        pred_phis = result['pred_phis']
        gt_tangents = result['gt_tangents']
        gt_phis = result['gt_phis']
        # classes_phi = result['classes_phi']


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

    # for i in range(len(results)):
    #     result = results[i]
    #     chunk_idx = i * 10  # Simple mapping - adjust as needed

    #     if chunk_idx not in terrain_data:
    #         print(f"No terrain data for chunk {chunk_idx}, skipping sample {i}")
    #         continue

    #     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    #     # Get real coordinates and angles from terrain data
    #     x_coords = terrain_data[chunk_idx]['x']
    #     z_coords = terrain_data[chunk_idx]['z']
    #     real_tangents = terrain_data[chunk_idx]['tangent']
    #     real_normals = terrain_data[chunk_idx]['normal']

    #     # Plot terrain points
    #     ax1.scatter(x_coords, z_coords, s=4, c='blue', alpha=0.6, label='Terrain Points')
    #     ax2.scatter(x_coords, z_coords, s=4, c='blue', alpha=0.6, label='Terrain Points')

    #     # Draw angle arrows at each point
    #     for j in range(0, len(x_coords), 1):
    #         x_pos, z_pos = x_coords[j], z_coords[j]

    #         # CNN predicted angles (left plot)
    #         # if j < result['pred_phis'].shape[1]:
    #         #     cnn_idx = j
    #         #     pred_phi = result['pred_phis'][0, cnn_idx]

    #         # if j < result['pred_phis'].shape[1] * result['pred_phis'].shape[0]:  # Total predictions available
    #         #     beam_idx = j // 668  # Which beam (0-3)
    #         #     pixel_idx = j % 668  # Which pixel within beam (0-667)
    #         #     if beam_idx < 4:  # Valid beam
    #         #         pred_phi = result['pred_phis'][beam_idx, pixel_idx]
    #         #         print(pred_phi)
    #         if j < result['pred_phis'].shape[1]:  # Only use beam 0's 668 pixels
    #             pixel_idx = j * 4  # Sample every 4th pixel: 0, 4, 8, 12...
    #             if pixel_idx < 668:  # Stay within beam bounds
    #                 pred_phi = result['pred_phis'][0, pixel_idx]  # Always beam 0
    #                 print(f"j={j}, pixel_idx={pixel_idx}, pred_phi={pred_phi}")
    #             ax1.arrow(x_pos, z_pos, math.cos(pred_phi)*0.2, math.sin(pred_phi)*0.2,
    #                         head_width=0.05, color='red', length_includes_head=True, alpha=0.8)

    #             # For tangent mode, uncomment:
    #             # pred_tang = result['pred_tangents'][0, cnn_idx]
    #             # if pred_tang not in [-10.0, -20.0]:
    #             #     ax1.arrow(x_pos, z_pos, math.cos(pred_tang)*0.2, math.sin(pred_tang)*0.2,
    #             #              head_width=0.05, color='green', length_includes_head=True, alpha=0.8)

    #         # Real terrain angles (right plot)
    #         ax2.arrow(x_pos, z_pos, math.cos(real_normals[j])*0.2, math.sin(real_normals[j])*0.2,
    #                  head_width=0.05, color='red', length_includes_head=True, alpha=0.8)
    #         ax2.arrow(x_pos, z_pos, math.cos(real_tangents[j])*0.2, math.sin(real_tangents[j])*0.2,
    #                  head_width=0.05, color='green', length_includes_head=True, alpha=0.8)

    #     # Mark sensor position
    #     ax1.plot(0, 0, 'x', markersize=10, color='black', markeredgewidth=2, label='Sensor')
    #     ax2.plot(0, 0, 'x', markersize=10, color='black', markeredgewidth=2, label='Sensor')

    #     # Configure plots
    #     ax1.set_title("CNN Predicted Angles")
    #     ax1.axis('equal')
    #     ax1.grid(True, alpha=0.3)
    #     ax1.legend()

    #     ax2.set_title("Real Terrain Angles")
    #     ax2.axis('equal')
    #     ax2.grid(True, alpha=0.3)
    #     ax2.legend()

    #     plt.suptitle(f"Sample {i} - Spatial Angle Comparison (Red=Phi/Normal, Green=Tangent)")
    #     plt.tight_layout()
    #     plt.show()

        # print(f"Visualized sample {i} using terrain chunk {chunk_idx}")

# ==================== MAIN EXECUTION ====================

def main():
    # Setup
    csv_file = '/Users/farhang/Downloads/fls_all_with_phi.csv'

    # dataset = BathymetryDataset(csv_file)

    phi_dataset = BathymetryPhiDataset(csv_file)
    tangent_dataset = BathymetryTangentDataset(csv_file)

    print(f"Phi Dataset loaded: {len(phi_dataset)} samples")
    print(f"Tangent Dataset loaded: {len(tangent_dataset)} samples")

    # Create data splits
    # train_dataset, val_dataset, test_dataset = create_data_splits(dataset)
    # print(f"Data splits - Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

    phi_train, phi_val, phi_test = create_data_splits(phi_dataset)
    tangent_train, tangent_val, tangent_test = create_data_splits(tangent_dataset, seed=42)

    print(f"Phi splits - Train: {len(phi_train)}, Val: {len(phi_val)}, Test: {len(phi_test)}")
    print(f"Tangent splits - Train: {len(tangent_train)}, Val: {len(tangent_val)}, Test: {len(tangent_test)}")

    # # Create data loaders
    # train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    # val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    # test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Create data loaders for BOTH
    phi_train_loader = DataLoader(phi_train, batch_size=32, shuffle=True)
    phi_val_loader = DataLoader(phi_val, batch_size=32, shuffle=False)
    phi_test_loader = DataLoader(phi_test, batch_size=32, shuffle=False)

    tangent_train_loader = DataLoader(tangent_train, batch_size=32, shuffle=True)
    tangent_val_loader = DataLoader(tangent_val, batch_size=32, shuffle=False)
    tangent_test_loader = DataLoader(tangent_test, batch_size=32, shuffle=False)

    # Initialize model
    # model = IntensityToBathymetryUNet1D()

    # Training (uncomment to train)
    # train_model(model, train_loader, val_loader, num_epochs=130)

    phi_model = IntensityToBathymetryPhiUNet1D()
    tangent_model = IntensityToBathymetryTangentsUNet1D()

    # print("Training PHI network...")
    # train_model(phi_model, phi_train_loader, phi_val_loader, num_epochs=60, model_name="phi")

    # print("Training TANGENT network...")
    # train_model(tangent_model, tangent_train_loader, tangent_val_loader, num_epochs=60,model_name="tangent")



    # Load existing model
    # try:
    #     model.load_state_dict(torch.load('best_bathymetry_model.pth', map_location=get_device()))
    #     print("Loaded existing trained model")
    # except FileNotFoundError:
    #     print("No existing model found - using untrained model")

    try:
        phi_model.load_state_dict(torch.load('best_bathymetry_model_phi.pth', map_location=get_device()))
        print("Loaded existing PHI model")
    except FileNotFoundError:
        print("No existing PHI model found")

    try:
        tangent_model.load_state_dict(torch.load('best_bathymetry_model_tangent.pth', map_location=get_device()))
        print("Loaded existing TANGENT model")
    except FileNotFoundError:
        print("No existing TANGENT model found")

    # Evaluation
    print(f"\n{'='*50}")
    print("MODEL EVALUATION")
    print(f"{'='*50}")

    # test_loss = evaluate_model(model, test_loader)
    # print(f"Test Loss: {test_loss:.6f}")

    phi_test_loss = evaluate_model(phi_model, phi_test_loader)
    tangent_test_loss = evaluate_model(tangent_model, tangent_test_loader)

    print(f"Phi Test Loss: {phi_test_loss:.6f}")
    print(f"Tangent Test Loss: {tangent_test_loss:.6f}")

    # Test specific rows
    # print(f"\n{'='*50}")
    # print("DETAILED TESTING")
    # print(f"{'='*50}")

    print(f"\n{'='*50}")
    print("DETAILED TESTING - PHI NETWORK")
    print(f"{'='*50}")

    phi_results = test_specific_rows(phi_model, phi_test, num_test_rows=5,
                                output_dir='test_outputs_phi',
                                prediction_type='phi')

    print(f"\n{'='*50}")
    print("DETAILED TESTING - TANGENT NETWORK")
    print(f"{'='*50}")
    tangent_results = test_specific_rows(tangent_model, tangent_test, num_test_rows=5,
                                   output_dir='test_outputs_tangents',
                                   prediction_type='tangent')

    # test_results = test_specific_rows(model, test_dataset, num_test_rows=10)

    # #Load terrain data and visualize
    # terrain_data = load_terrain_coordinates()
    # if terrain_data:
    #     print(f"Loaded terrain data: {len(terrain_data)} chunks")

    # # Standard prediction visualization
    # visualize_predictions(test_results, terrain_data, show_plots=True)

    # # Terrain spatial visualization (shows physical angle relationships)
    # visualize_terrain_comparison(test_results, terrain_data)

    # print("Visualizing PHI predictions...")
    # visualize_predictions(phi_results, show_plots=True)

    # print("Visualizing TANGENT predictions...")
    # visualize_predictions(tangent_results, show_plots=True)

    # # Save model
    # phi_model_path = 'bathymetry_cnn_model_phi.pth'
    # tangent_model_path = 'bathymetry_cnn_model_tangent.pth'

    # torch.save(phi_model.state_dict(), phi_model_path)
    # torch.save(tangent_model.state_dict(), tangent_model_path)

    # print(f"\nModels saved:")
    # print(f"- Phi model: {phi_model_path}")
    # print(f"- Tangent model: {tangent_model_path}")

    # model_path = 'bathymetry_cnn_model.pth'
    # torch.save(model.state_dict(), model_path)
    # print(f"\nModel saved: {model_path}")


    original_csv_path = "/Users/farhang/Downloads/fls_all_with_phi.csv"  # actual file path
    output_csv_path = "/Users/farhang/Downloads/fls_2d_terrain_prediction_output.csv"  #output file
    save_predictions_to_csv(tangent_results, phi_results, original_csv_path, output_csv_path)

if __name__ == "__main__":
    main()