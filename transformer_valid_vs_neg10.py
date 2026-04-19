import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset

import pandas as pd
import numpy as np
import os
from cnn_separate_fls_bathymetry_separate import BathymetryDataset, RegressorCNN

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import f1_score
from cnn_separate_fls_bathymetry_separate import (
    save_splits_to_csv,
    save_splits_with_indices_to_csv,
    save_predictions_with_indices_to_csv,
    create_valid_mask,
    evaluate_single_row
)


# ==================== FILTERED DATASET ====================

class FilteredBathymetryDataset(Dataset):
    """
    Dataset that excludes -20 values
    Only returns samples where target is Valid or -10
    """
    def __init__(self, csv_file, prediction_type='phi'):
        self.data = pd.read_csv(csv_file, header=None)
        self.prediction_type = prediction_type

        # Pre-filter to only keep rows with Valid or -10 data
        print(f"\nFiltering dataset to exclude -20s...")
        valid_indices = []

        for idx in range(len(self.data)):
            row = self.data.iloc[idx]

            if prediction_type == 'phi':
                target_data = torch.tensor(row.iloc[3341:6013].values, dtype=torch.float32)
            else:
                target_data = torch.tensor(row.iloc[669:3341].values, dtype=torch.float32)

            target_processed, _ = create_valid_mask(target_data)

            # Check if this row has any Valid or -10 data (not all -20)
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

        # Extract intensities
        intensities = torch.tensor(row.iloc[1:669].values, dtype=torch.float32)
        intensities_processed, _ = create_valid_mask(intensities)

        # Extract targets
        if self.prediction_type == 'phi':
            target_data = torch.tensor(row.iloc[3341:6013].values, dtype=torch.float32)
        else:
            target_data = torch.tensor(row.iloc[669:3341].values, dtype=torch.float32)

        target_processed, _ = create_valid_mask(target_data)
        ground_truth = target_processed.view(4, 668).contiguous().view(-1)

        # Filter out -20 positions
        non_neg20_mask = (ground_truth != -20)

        # Keep only Valid and -10 positions
        ground_truth_filtered = ground_truth[non_neg20_mask]

        # Expand intensities to match (4 beams per pixel)
        intensities_expanded = intensities_processed.repeat_interleave(4)
        intensities_filtered = intensities_expanded[non_neg20_mask]

        # Create binary labels: 1 = Valid, 0 = -10
        labels = (ground_truth_filtered != -10).float()

        return intensities_filtered, labels, ground_truth_filtered


# ==================== PURE TRANSFORMER MODEL ====================

class PureTransformerBinaryClassifier(nn.Module):
    """
    Pure Transformer for Valid vs -10 classification
    No CNN, no threshold - just Transformer
    """
    def __init__(self, d_model=256, nhead=8, num_layers=4, dropout=0.2):
        super().__init__()

        # Input embedding
        self.input_embed = nn.Sequential(
            nn.Linear(1, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, d_model)
        )

        # Learnable positional encoding (variable sequence length)
        self.pos_embed = nn.Parameter(torch.randn(1, 1000, d_model))  # Max 1000 positions

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True  # Pre-norm architecture (more stable)
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, d_model // 4),
            nn.LayerNorm(d_model // 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 4, 1)
        )

    def forward(self, intensities):
        """
        Args:
            intensities: [B, seq_len] - variable length sequences
        Returns:
            logits: [B, seq_len] - classification logits
        """
        batch_size, seq_len = intensities.shape

        # Embed each intensity
        x = intensities.unsqueeze(-1)  # [B, seq_len, 1]
        x = self.input_embed(x)  # [B, seq_len, d_model]

        # Add positional encoding
        x = x + self.pos_embed[:, :seq_len, :]

        # Apply transformer
        x = self.transformer(x)  # [B, seq_len, d_model]

        # Classify each position
        logits = self.classifier(x).squeeze(-1)  # [B, seq_len]

        return logits


# ==================== TRAINING ====================

def collate_fn(batch):
    """
    Custom collate to handle variable-length sequences
    """
    intensities_list, labels_list, gt_list = zip(*batch)

    # Find max length in this batch
    max_len = max(x.size(0) for x in intensities_list)

    # Pad sequences
    intensities_padded = []
    labels_padded = []
    masks = []

    for intens, labels in zip(intensities_list, labels_list):
        seq_len = intens.size(0)

        # Pad
        pad_len = max_len - seq_len
        intensities_padded.append(F.pad(intens, (0, pad_len), value=0))
        labels_padded.append(F.pad(labels, (0, pad_len), value=0))

        # Mask (1 = real data, 0 = padding)
        mask = torch.cat([torch.ones(seq_len), torch.zeros(pad_len)])
        masks.append(mask)

    return (torch.stack(intensities_padded),
            torch.stack(labels_padded),
            torch.stack(masks).bool())


def train_pure_transformer(model, train_loader, val_loader, num_epochs=50):
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=5e-4,
        total_steps=num_epochs * len(train_loader),
        pct_start=0.3
    )

    # Focal loss for class imbalance
    def focal_loss(logits, targets, mask, alpha=3.0, gamma=2.0):
        probs = torch.sigmoid(logits)
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')

        # Focal weight
        pt = torch.where(targets == 1, probs, 1 - probs)
        focal_weight = alpha * (1 - pt) ** gamma

        # Apply mask (ignore padding)
        loss = (focal_weight * bce * mask.float()).sum() / mask.sum()
        return loss

    print(f"\n{'='*60}")
    print("TRAINING PURE TRANSFORMER (Valid vs -10)")
    print(f"{'='*60}")
    print(f"Device: {device}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0

        for intensities, labels, masks in train_loader:
            intensities = intensities.to(device)
            labels = labels.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()

            logits = model(intensities)
            loss = focal_loss(logits, labels, masks, alpha=3.0, gamma=2.0)

            if torch.isfinite(loss):
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()

                train_loss += loss.item()

        # Validation
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for intensities, labels, masks in val_loader:
                intensities = intensities.to(device)
                labels = labels.to(device)
                masks = masks.to(device)

                logits = model(intensities)
                loss = focal_loss(logits, labels, masks)

                if torch.isfinite(loss):
                    val_loss += loss.item()

        train_loss /= len(train_loader)
        val_loss /= len(val_loader)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_pure_transformer.pth')

        if epoch % 10 == 0 or epoch < 5:
            print(f"Epoch {epoch+1:3d}/{num_epochs} | "
                  f"Train: {train_loss:.4f} | Val: {val_loss:.4f} | Best: {best_val_loss:.4f}")

    model.load_state_dict(torch.load('best_pure_transformer.pth', map_location=device))
    print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}\n")
    return model

# Inference function
def run_full_inference(dataset, model, regressor, device, threshold_neg20=103.0):
    """
    Combines:
    - Threshold for -20 detection
    - Transformer for Valid vs -10
    - Regressor for angles
    """
    results = {}

    for i in range(len(dataset)):
        intensities, ground_truth = dataset[i]

        # Predict angles with regressor
        angle_pred = regressor(intensities.unsqueeze(0).to(device)).squeeze(0).cpu()

        # Expand intensities for 4 beams per pixel
        intensities_expanded = intensities.repeat_interleave(4)

        # Stage 1: Detect -20 using threshold
        is_neg20 = intensities_expanded < threshold_neg20

        # Stage 2: For non--20, use Transformer to classify Valid vs -10
        non_neg20_indices = ~is_neg20

        if non_neg20_indices.any():
            # Get intensities for non--20 positions
            intensities_non_neg20 = intensities_expanded[non_neg20_indices].unsqueeze(0).to(device)

            # Classify with Transformer
            with torch.no_grad():
                class_logits = model(intensities_non_neg20)
                probs = torch.sigmoid(class_logits).squeeze(0).cpu()
                is_valid = probs > 0.5  # 1 = Valid, 0 = -10

            # Create final prediction
            final_pred = angle_pred.clone()
            final_pred[is_neg20] = -20.0

            # For non--20 positions, set -10 where not valid
            non_neg20_preds = torch.full((non_neg20_indices.sum(),), -10.0)
            non_neg20_preds[is_valid] = angle_pred[non_neg20_indices][is_valid]
            final_pred[non_neg20_indices] = non_neg20_preds
        else:
            # All are -20
            final_pred = torch.full_like(angle_pred, -20.0)

        results[i] = {
            'pred_phis': final_pred.view(4, 668).detach().numpy(),
            'true_phis': ground_truth.view(4, 668).numpy()
        }

        if i < 20:
            metrics = evaluate_single_row(final_pred, ground_truth)
            print(f"Sample {i:2d} | "
                    f"Angle RMSE: {metrics['angle_rmse']:6.3f} | "
                    f"Flag Acc: {metrics['flag_acc']:5.1%} | "
                    f"Valid: {metrics['clean_points']:3d}/{metrics['valid_points']:3d}")

    return results

# ==================== MAIN ====================

def main():
    csv_file = '/Users/farhang/Downloads/fls_all_with_phi.csv'

    print("\n" + "="*60)
    print("PURE TRANSFORMER: Valid vs -10 ONLY")
    print("="*60)

    # Create splits
    splits_dir = "./data_splits"
    if not os.path.exists(f"{splits_dir}/train_data.csv"):
        train_csv, val_csv, test_csv = save_splits_to_csv(csv_file, splits_dir)
    else:
        train_csv = f"{splits_dir}/train_data.csv"
        val_csv = f"{splits_dir}/val_data.csv"
        test_csv = f"{splits_dir}/test_data.csv"

    # Load filtered datasets (no -20s)
    print("\nLoading filtered datasets...")
    datasets = {
        'train': FilteredBathymetryDataset(train_csv, prediction_type='phi'),
        'val': FilteredBathymetryDataset(val_csv, prediction_type='phi'),
        'test': FilteredBathymetryDataset(test_csv, prediction_type='phi')
    }

    # Create data loaders with custom collate
    loaders = {
        'train': DataLoader(datasets['train'], batch_size=16, shuffle=True, collate_fn=collate_fn),
        'val': DataLoader(datasets['val'], batch_size=16, shuffle=False, collate_fn=collate_fn),
        'test': DataLoader(datasets['test'], batch_size=16, shuffle=False, collate_fn=collate_fn)
    }

    # Create model
    model = PureTransformerBinaryClassifier(d_model=256, nhead=8, num_layers=4, dropout=0.2)

    model_path = 'best_pure_transformer.pth'
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

    if os.path.exists(model_path):
        print(f"\nLoading existing model from {model_path}")
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
    else:
        print("\nTraining new pure transformer...")
        model = train_pure_transformer(model, loaders['train'], loaders['val'], num_epochs=50)

    # Evaluate
    print("\n" + "="*60)
    print("EVALUATION ON FILTERED TEST SET")
    print("="*60)

    model.eval()
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    model.to(device)

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for intensities, labels, masks in loaders['test']:
            intensities = intensities.to(device)
            # Keep labels and masks on CPU

            logits = model(intensities)
            probs = torch.sigmoid(logits).cpu()  # Move to CPU immediately
            preds = (probs > 0.5).float()

            # Now everything is on CPU
            for pred, label, mask in zip(preds, labels, masks):
                pred_valid = pred[mask]
                label_valid = label[mask]

                all_preds.extend(pred_valid.tolist())
                all_labels.extend(label_valid.tolist())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Calculate metrics

    print("\nBinary Classification: Valid (1) vs -10 (0)")
    print(classification_report(all_labels, all_preds, target_names=['(-10)', 'Valid']))

    cm = confusion_matrix(all_labels, all_preds)
    print(f"\nConfusion Matrix:")
    print(f"              Predicted")
    print(f"              -10    Valid")
    print(f"Actual -10    {cm[0,0]:<6} {cm[0,1]:<6}")
    print(f"       Valid  {cm[1,0]:<6} {cm[1,1]:<6}")

    f1_neg10 = f1_score(all_labels, all_preds, pos_label=0)
    f1_valid = f1_score(all_labels, all_preds, pos_label=1)

    print(f"\nF1 Scores:")
    print(f"  (-10):  {f1_neg10:.3f}")
    print(f"  Valid:  {f1_valid:.3f}")

    print("\n" + "="*60)
    print("COMPLETE!")
    print("="*60)

    print("\n" + "="*60)
    print("RUNNING FULL INFERENCE FOR CSV GENERATION")
    print("="*60)

    # We need to run inference on FULL test set (including -20s)
    # and combine Transformer predictions with -20 detection
    # Load full datasets (with -20s)
    full_datasets = {
        'train': BathymetryDataset(train_csv, prediction_type='phi'),
        'test': BathymetryDataset(test_csv, prediction_type='phi')
    }

    # Load regressor for angle prediction
    regressor = RegressorCNN(prediction_type='phi', dropout_rate=0.1)
    regressor_path = 'best_phi_regressor.pth'

    if os.path.exists(regressor_path):
        regressor.load_state_dict(torch.load(regressor_path, map_location=device))
        print(f"Loaded regressor from {regressor_path}")
    else:
        print("WARNING: No regressor found, angle predictions will be poor!")

    regressor.to(device).eval()

    # Run inference on test set
    print("\nTest set inference...")
    test_results = run_full_inference(full_datasets['test'], model, regressor, device)

    # Run inference on train set
    print("\nTrain set inference...")
    train_results = run_full_inference(full_datasets['train'], model, regressor, device)

    # Save to CSV
    print("\n" + "="*60)
    print("SAVING PREDICTIONS TO CSV")
    print("="*60)

    test_with_indices = save_splits_with_indices_to_csv(csv_file, splits_dir, split_type='test')
    train_with_indices = save_splits_with_indices_to_csv(csv_file, splits_dir, split_type='train')

    # Create dummy tangent results
    tangent_results_test = {i: {'pred_tangents': r['pred_phis'], 'true_tangents': r['true_phis']}
                           for i, r in test_results.items()}
    tangent_results_train = {i: {'pred_tangents': r['pred_phis'], 'true_tangents': r['true_phis']}
                            for i, r in train_results.items()}

    # Save
    test_predictions_path = "./data_splits/test_predictions_pure_transformer.csv"
    save_predictions_with_indices_to_csv(
        tangent_results_test,
        test_results,
        test_with_indices,
        test_predictions_path
    )
    print(f"✓ Test predictions saved to {test_predictions_path}")

    train_predictions_path = "./data_splits/train_predictions_pure_transformer.csv"
    save_predictions_with_indices_to_csv(
        tangent_results_train,
        train_results,
        train_with_indices,
        train_predictions_path
    )
    print(f"✓ Train predictions saved to {train_predictions_path}")

    print("\n" + "="*60)
    print("COMPLETE!")
    print("="*60)
    print(f"\nModel saved: best_pure_transformer.pth")
    print(f"Test CSV: {test_predictions_path}")
    print(f"Train CSV: {train_predictions_path}")
    print("="*60)

if __name__ == "__main__":
    main()
