import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import csv

def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')

def create_valid_mask(data, invalid_values=[-20.0]):
    data_modified = data.clone()
    data_modified[torch.isnan(data_modified)] = -20.0
    valid_mask = torch.ones_like(data_modified, dtype=torch.bool)
    for val in invalid_values:
        valid_mask &= (data_modified != val)
    return data_modified, valid_mask

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

class IntensityToBathymetryUNet1D(nn.Module):
    def __init__(self, prediction_type='phi', dropout_rate=0.1):
        super().__init__()
        self.prediction_type = prediction_type

        self.enc1 = nn.Sequential(
            nn.Conv1d(1, 64, 7, padding=3),
            nn.InstanceNorm1d(64), nn.LeakyReLU()
        )
        self.enc2 = nn.Sequential(
            nn.Conv1d(64, 128, 5, stride=2, padding=2),
            nn.InstanceNorm1d(128), nn.LeakyReLU()
        )
        self.enc3 = nn.Sequential(
            nn.Conv1d(128, 256, 3, stride=2, padding=1),
            nn.InstanceNorm1d(256), nn.LeakyReLU()
        )

        self.residual_blocks = nn.ModuleList([
            ResidualBlock1D(256, dropout_rate) for _ in range(5)
        ])

        self.dec1 = nn.Sequential(
            nn.ConvTranspose1d(256, 128, 3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm1d(128), nn.LeakyReLU()
        )
        self.dec2 = nn.Sequential(
            nn.ConvTranspose1d(256, 64, 5, stride=2, padding=2, output_padding=1),
            nn.InstanceNorm1d(64), nn.LeakyReLU()
        )

        self.final_upsample = nn.Sequential(
            nn.ConvTranspose1d(128, 32, 7, stride=4, padding=3, output_padding=3),
            nn.LeakyReLU(),
        )

        self.classifier = nn.Conv1d(32, 3, 3, padding=1)

        if prediction_type == 'combined':
            self.regressor = nn.Conv1d(32, 2, 3, padding=1)
        else:
            self.regressor = nn.Conv1d(32, 1, 3, padding=1)

    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size, 1, 668)

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

        class_logits = self.classifier(features)
        angle_pred = self.regressor(features)

        return class_logits, angle_pred.squeeze(1) if angle_pred.size(1) == 1 else angle_pred

def predict_with_model(model, intensities, device):
    model.eval()
    if intensities.dim() == 1:
        intensities = intensities.unsqueeze(0)

    with torch.no_grad():
        class_logits, angle_preds = model(intensities.to(device))

    class_preds = torch.argmax(class_logits, dim=1).cpu()
    angle_preds = angle_preds.cpu()
    prediction = angle_preds.clone()

    prediction[class_preds == 1] = -10.0
    prediction[class_preds == 2] = -20.0

    return prediction

def reshape_predictions(predictions, prediction_type):
    if predictions.dim() == 1:
        predictions = predictions.unsqueeze(0)

    batch_size = predictions.size(0)

    if prediction_type == 'phi':
        phis = torch.zeros(batch_size, 4, 668)
        for i in range(4):
            phis[:, i, :] = predictions[:, i::4]
        tangents = torch.zeros_like(phis)

    elif prediction_type == 'tangent':
        tangents = torch.zeros(batch_size, 4, 668)
        for i in range(4):
            tangents[:, i, :] = predictions[:, i::4]
        phis = torch.zeros_like(tangents)

    return tangents.squeeze(0), phis.squeeze(0)

def generate_full_predictions(original_csv_path, output_csv_path):
    device = get_device()
    print(f"Using device: {device}")

    # Load models
    phi_model = IntensityToBathymetryUNet1D(prediction_type='phi')
    tangent_model = IntensityToBathymetryUNet1D(prediction_type='tangent')

    phi_model.load_state_dict(torch.load('best_bathymetry_model_phi.pth', map_location=device))
    tangent_model.load_state_dict(torch.load('best_bathymetry_model_tangent.pth', map_location=device))

    phi_model.to(device).eval()
    tangent_model.to(device).eval()
    print("Models loaded successfully")

    # Read and process CSV
    original_data = []
    with open(original_csv_path, 'r', newline='') as f:
        reader = csv.reader(f)
        for row in reader:
            processed_row = []
            for val in row:
                try:
                    processed_row.append(float(val))
                except ValueError:
                    processed_row.append(val)
            original_data.append(processed_row)

    print(f"Processing {len(original_data)} rows...")

    # Generate predictions
    output_data = [row[:] for row in original_data]

    with torch.no_grad():
        for idx in range(len(original_data)):
            row = original_data[idx]

            # Extract intensities
            intensities = torch.tensor(row[1:669], dtype=torch.float32)
            intensities_processed, _ = create_valid_mask(intensities)

            # Get predictions
            phi_prediction = predict_with_model(phi_model, intensities_processed, device)
            tangent_prediction = predict_with_model(tangent_model, intensities_processed, device)

            # Reshape predictions
            pred_tangents, _ = reshape_predictions(tangent_prediction, 'tangent')
            _, pred_phis = reshape_predictions(phi_prediction, 'phi')

            # Replace in output data
            # Tangent columns (669-3340)
            tangent_flat = pred_tangents.flatten()
            for i, val in enumerate(tangent_flat):
                col_idx = 669 + i
                if col_idx < len(output_data[idx]):
                    output_data[idx][col_idx] = float(val)

            # Phi columns (3341-6012)
            phi_flat = pred_phis.flatten()
            for i, val in enumerate(phi_flat):
                col_idx = 3341 + i
                if col_idx < len(output_data[idx]):
                    output_data[idx][col_idx] = float(val)

            if idx % 100 == 0:
                print(f"Processed {idx}/{len(original_data)} rows")

    # Save output
    with open(output_csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        for row in output_data:
            writer.writerow(row)

    print(f"Full predictions saved to {output_csv_path}")

if __name__ == "__main__":
    original_csv = "/Users/farhang/Downloads/fls_all_with_phis_long.csv"
    output_csv = "full_predictions.csv"

    generate_full_predictions(original_csv, output_csv)
