import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import pandas as pd
import numpy as np
import os
import csv

from cnn_separate_fls_bathymetry_separate import (
    BathymetryDataset,
    RegressorCNN,
    evaluate_single_row,
    save_splits_to_csv,
    save_splits_with_indices_to_csv,
    save_predictions_with_indices_to_csv,
    create_valid_mask
)

class TwoStageClassifier(nn.Module):
    """
    Stage 1: Neural network for -20 detection (no threshold)
    Stage 2: Transformer for Valid vs -10 classification
    """
    def __init__(self, prediction_type='phi', dropout_rate=0.1):
        super().__init__()

        # Stage 1: MLP for -20 detection (learns from intensity)
        self.neg20_detector = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(32, 1)  # Logits for -20 probability
        )

        # Stage 2: Transformer for Valid vs -10 (uses spatial context)
        self.transformer_classifier = TransformerValidVsNeg10(
            input_dim=668,
            d_model=128,
            nhead=8,
            num_layers=3,
            dropout=dropout_rate
        )

        # Angle regressor
        self.angle_regressor = RegressorCNN(prediction_type, dropout_rate)

    def forward(self, intensities, training=True):
        """
        Args:
            intensities: [B, 668]
        Returns:
            final_predictions: [B, 2672]
            neg20_probs: [B, 2672] - probability of being -20
            valid_probs: [B, 2672] - probability of Valid (vs -10) for non--20
        """
        batch_size = intensities.size(0)

        # Expand intensities to all 4 beams
        intensities_expanded = intensities.repeat_interleave(4, dim=1)  # [B, 2672]

        # Stage 1: Detect -20 using neural network (NO threshold)
        intensities_flat = intensities_expanded.reshape(-1, 1)  # [B*2672, 1]
        neg20_logits_flat = self.neg20_detector(intensities_flat)  # [B*2672, 1]
        neg20_logits = neg20_logits_flat.reshape(batch_size, -1)  # [B, 2672]
        neg20_probs = torch.sigmoid(neg20_logits)

        # Stage 2: For non--20 positions, classify Valid vs -10 using Transformer
        valid_vs_neg10_logits = self.transformer_classifier(intensities)  # [B, 2672]
        valid_probs = torch.sigmoid(valid_vs_neg10_logits)

        # Predict angles
        angle_predictions = self.angle_regressor(intensities)  # [B, 2672]

        if training:
            # Soft combination for gradients
            final_predictions = (
                neg20_probs * (-20.0) +  # -20 contribution
                (1 - neg20_probs) * (
                    valid_probs * angle_predictions +  # Valid angle contribution
                    (1 - valid_probs) * (-10.0)  # -10 contribution
                )
            )
        else:
            # Hard decisions for inference
            final_predictions = angle_predictions.clone()

            # First: mark -20s (learned decision)
            is_neg20 = (neg20_probs > 0.5)
            final_predictions[is_neg20] = -20.0

            # Second: among remaining, classify Valid vs -10
            is_not_neg20 = ~is_neg20
            is_neg10 = is_not_neg20 & (valid_probs < 0.5)
            final_predictions[is_neg10] = -10.0

        return final_predictions, neg20_probs, valid_probs

class TransformerValidVsNeg10(nn.Module):
    def __init__(self, input_dim=668, d_model=256, nhead=8, num_layers=4, dropout=0.1):  # Changed
        super().__init__()

        # Richer input embedding
        self.input_proj = nn.Sequential(
            nn.Linear(1, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, d_model)
        )

        self.pos_encoding = nn.Parameter(torch.randn(1, input_dim, d_model))

        # Larger transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,  # 256 instead of 128
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)  # 4 layers

        # Deeper output projection
        self.output_proj = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, d_model // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 4, 4)
        )

    def forward(self, intensities):
        """
        Args:
            intensities: [B, 668]
        Returns:
            logits: [B, 2672] - logits for Valid (vs -10)
        """
        batch_size = intensities.size(0)

        # Project each intensity value
        x = intensities.unsqueeze(-1)  # [B, 668, 1]
        x = self.input_proj(x)  # [B, 668, d_model]

        # Add positional encoding
        x = x + self.pos_encoding

        # Apply transformer (captures spatial relationships)
        x = self.transformer(x)  # [B, 668, d_model]

        # Project to 4 outputs per pixel
        logits_per_pixel = self.output_proj(x)  # [B, 668, 4]

        # Reshape to [B, 2672]
        logits = logits_per_pixel.reshape(batch_size, -1)

        return logits


class TwoStageLoss(nn.Module):
    def __init__(self, alpha_stage1=2.0, alpha_stage2=8.0, beta=1.0):  # Increased stage2
        super().__init__()
        self.alpha_stage1 = alpha_stage1
        self.alpha_stage2 = alpha_stage2
        self.beta = beta

    def forward(self, final_preds, neg20_probs, valid_probs, angle_preds, targets):
        # Stage 1: unchanged
        target_is_neg20 = (targets == -20).float()
        stage1_loss = F.binary_cross_entropy(neg20_probs, target_is_neg20)

        # Stage 2: ADD CLASS WEIGHT FOR VALID
        non_neg20_mask = (targets != -20)

        if non_neg20_mask.any():
            target_is_valid = (targets != -10).float()

            # Count class distribution
            num_valid = target_is_valid[non_neg20_mask].sum()
            num_neg10 = (1 - target_is_valid[non_neg20_mask]).sum()

            # Weight Valid class more (it's rarer)
            pos_weight = (num_neg10 / (num_valid + 1e-8)).clamp(min=1.0, max=5.0)

            # Weighted BCE
            stage2_loss = F.binary_cross_entropy_with_logits(
                torch.logit(valid_probs[non_neg20_mask] + 1e-8),
                target_is_valid[non_neg20_mask],
                pos_weight=pos_weight.unsqueeze(0)
            )
        else:
            stage2_loss = torch.tensor(0.0, device=targets.device)

        # Rest unchanged
        valid_mask = (targets != -10) & (targets != -20)
        if valid_mask.any():
            reg_loss = F.mse_loss(angle_preds[valid_mask], targets[valid_mask])
        else:
            reg_loss = torch.tensor(0.0, device=targets.device)

        total_loss = (self.alpha_stage1 * stage1_loss +
                     self.alpha_stage2 * stage2_loss +
                     self.beta * reg_loss)

        return total_loss, stage1_loss, stage2_loss, reg_loss


def train_two_stage_model(model, train_loader, val_loader, num_epochs=50):
    """Train two-stage model"""
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    model.to(device)

    optimizer = torch.optim.AdamW([
        {'params': model.transformer_classifier.parameters(), 'lr': 1e-4},
        {'params': model.angle_regressor.parameters(), 'lr': 1e-5},
        {'params': model.neg20_detector.parameters(), 'lr': 1e-4}
    ], weight_decay=1e-4)

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=[1e-4, 1e-4, 1e-2],
        total_steps=num_epochs,
        pct_start=0.3
    )

    criterion = TwoStageLoss(alpha_stage1=2.0, alpha_stage2=5.0, beta=1.0)

    print(f"\n{'='*60}")
    print("TRAINING TWO-STAGE TRANSFORMER MODEL")
    print(f"{'='*60}")
    print(f"Device: {device}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_s1_loss = 0.0
        train_s2_loss = 0.0
        train_reg_loss = 0.0

        for intensities, targets in train_loader:
            intensities = intensities.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()

            final_preds, neg20_probs, valid_probs = model(intensities, training=True)
            angle_preds = model.angle_regressor(intensities)

            loss, s1_loss, s2_loss, reg_loss = criterion(
                final_preds, neg20_probs, valid_probs, angle_preds, targets
            )

            if torch.isfinite(loss):
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                train_loss += loss.item()
                train_s1_loss += s1_loss.item()
                train_s2_loss += s2_loss.item()
                train_reg_loss += reg_loss.item()

        # Validation
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for intensities, targets in val_loader:
                intensities = intensities.to(device)
                targets = targets.to(device)

                final_preds, neg20_probs, valid_probs = model(intensities, training=True)
                angle_preds = model.angle_regressor(intensities)

                loss, s1_loss, s2_loss, reg_loss = criterion(
                    final_preds, neg20_probs, valid_probs, angle_preds, targets
                )

                if torch.isfinite(loss):
                    val_loss += loss.item()

        train_loss /= len(train_loader)
        val_loss /= len(val_loader)

        scheduler.step()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_two_stage_model.pth')

        if epoch % 10 == 0 or epoch < 5:
            print(f"Epoch {epoch+1:3d}/{num_epochs} | "
                f"Train: {train_loss:.4f} "
                f"(s1:{train_s1_loss/len(train_loader):.3f}, "
                f"s2:{train_s2_loss/len(train_loader):.3f}, "
                f"reg:{train_reg_loss/len(train_loader):.4f}) | "
                f"Val: {val_loss:.4f}")

    model.load_state_dict(torch.load('best_two_stage_model.pth', map_location=device))
    print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")
    print(f"Training complete. Best val loss: {best_val_loss:.4f}\n")
    return model

def main():
    """
    Two-Stage Approach:
    Stage 1: Intensity threshold for -20 detection
    Stage 2: Transformer for Valid vs -10 classification
    """
    csv_file = '/Users/farhang/Downloads/fls_all_with_phi.csv'

    print("\n" + "="*60)
    print("TWO-STAGE TRANSFORMER APPROACH")
    print("Stage 1: Threshold for -20 detection")
    print("Stage 2: Transformer for Valid vs -10")
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

    # Load datasets
    print("\nLoading datasets...")
    datasets = {
        'train': BathymetryDataset(train_csv, prediction_type='phi'),
        'val': BathymetryDataset(val_csv, prediction_type='phi'),
        'test': BathymetryDataset(test_csv, prediction_type='phi')
    }

    print(f"Train: {len(datasets['train'])} samples")
    print(f"Val: {len(datasets['val'])} samples")
    print(f"Test: {len(datasets['test'])} samples")

    # Create data loaders
    loaders = {
        'train': DataLoader(datasets['train'], batch_size=8, shuffle=True),
        'val': DataLoader(datasets['val'], batch_size=8, shuffle=False),
        'test': DataLoader(datasets['test'], batch_size=8, shuffle=False)
    }

    # Create or load model
    model = TwoStageClassifier(prediction_type='phi', dropout_rate=0.1)

    model_path = 'best_two_stage_model.pth'
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

    if os.path.exists(model_path):
        print(f"\nLoading existing model from {model_path}")
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
    else:
        print("\nTraining new two-stage model...")
        model = train_two_stage_model(
            model,
            loaders['train'],
            loaders['val'],
            num_epochs=50
        )

    # Test
    print("\n" + "="*60)
    print("TESTING TWO-STAGE MODEL")
    print("="*60)

    model.to(device).eval()

    test_results = {}
    train_results = {}

    # Test set inference
    print("\nRunning inference on test set...")
    with torch.no_grad():
        for i in range(len(datasets['test'])):
            intensities, ground_truth = datasets['test'][i]

            final_pred, neg20_probs, valid_probs = model(
                intensities.unsqueeze(0).to(device),
                training=False
            )
            final_pred = final_pred.squeeze(0).cpu()

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

            final_pred, neg20_probs, valid_probs = model(
                intensities.unsqueeze(0).to(device),
                training=False
            )
            final_pred = final_pred.squeeze(0).cpu()

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
    test_predictions_path = "./data_splits/test_predictions_two_stage.csv"
    save_predictions_with_indices_to_csv(
        tangent_results_test,
        test_results,
        test_with_indices,
        test_predictions_path
    )
    print(f"✓ Test predictions saved to {test_predictions_path}")

    train_predictions_path = "./data_splits/train_predictions_two_stage.csv"
    save_predictions_with_indices_to_csv(
        tangent_results_train,
        train_results,
        train_with_indices,
        train_predictions_path
    )
    print(f"✓ Train predictions saved to {train_predictions_path}")

    # Print summary
    print("\n" + "="*60)
    print("COMPLETE!")
    print("="*60)
    print(f" Transformer with {sum(p.numel() for p in model.transformer_classifier.parameters()):,} parameters")
    print(f"\nOutput files:")
    print(f"  - {test_predictions_path}")
    print(f"  - {train_predictions_path}")
    print("\nThis approach:")
    print("="*60)


if __name__ == "__main__":
    main()
